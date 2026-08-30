"""End-to-end YOLO → DINO → hierarchical retrieval pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from classifier.dino_backbone import DINOBackbone
from classifier.projection_head import HierarchicalProjectionHead
from detector.obb_utils import clip_xyxy
from detector.yolo_model import Detection, YOLOModel
from .decision_logic import ClassificationDecision, DecisionLogic
from .faiss_retriever import Candidate, FAISSRetriever


@dataclass(frozen=True)
class Prediction:
    bbox: tuple[float, float, float, float]
    polygon: tuple[tuple[float, float], ...] | None
    detection_confidence: float
    decision: ClassificationDecision
    genus_candidates: tuple[Candidate, ...]
    species_candidates: tuple[Candidate, ...]
    gate_probability: float | None = None
    gate_accepted: bool | None = None
    rejection_stage: str | None = None
    crop_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision"] = self.decision.to_dict()
        value["genus_candidates"] = [candidate.to_dict() for candidate in self.genus_candidates]
        value["species_candidates"] = [candidate.to_dict() for candidate in self.species_candidates]
        return value


class DiatomDINOPipeline:
    def __init__(
        self,
        *,
        detector: YOLOModel,
        backbone: DINOBackbone,
        projection_head: HierarchicalProjectionHead,
        genus_index: FAISSRetriever,
        species_index: FAISSRetriever,
        decision_logic: DecisionLogic,
        transform: Any,
        artifact_gate: Any | None = None,
        detector_confidence: float = 0.25,
        detector_iou: float = 0.7,
        detector_image_size: int = 1024,
        top_k: int = 5,
        crop_padding: float = 0.05,
        save_crops: bool = False,
        output_dir: str | Path | None = None,
    ) -> None:
        self.detector = detector
        self.backbone = backbone
        self.projection_head = projection_head.to(backbone.device)
        self.genus_index = genus_index
        self.species_index = species_index
        self.decision_logic = decision_logic
        self.transform = transform
        self.artifact_gate = artifact_gate
        self.detector_confidence = float(detector_confidence)
        self.detector_iou = float(detector_iou)
        self.detector_image_size = detector_image_size
        self.top_k = int(top_k)
        self.crop_padding = float(crop_padding)
        self.save_crops = bool(save_crops)
        self.output_dir = Path(output_dir) if output_dir else None
        if self.save_crops and self.output_dir is None:
            raise ValueError("output_dir is required when save_crops=True")

    def _crop(self, image: Image.Image, detection: Detection) -> Image.Image:
        width, height = image.size
        x1, y1, x2, y2 = detection.xyxy
        padding_x = (x2 - x1) * self.crop_padding
        padding_y = (y2 - y1) * self.crop_padding
        clipped = clip_xyxy(
            (x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y),
            width,
            height,
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError(f"Detector produced an empty box: {tuple(clipped)}")
        return image.crop(tuple(float(value) for value in clipped))

    @torch.no_grad()
    def _embed_crops(self, crops: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        tensors = [self.transform(crop) for crop in crops]
        batch = torch.stack(tensors).to(self.backbone.device)
        self.backbone.set_train_mode(False)
        self.projection_head.eval()
        raw = self.backbone.extract_features(batch)
        genus, species = self.projection_head(raw)
        return genus.cpu().numpy(), species.cpu().numpy()

    def predict_single(self, image_path: str | Path) -> list[Prediction]:
        path = Path(image_path)
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        detections = self.detector.predict(
            str(path),
            confidence=self.detector_confidence,
            iou=self.detector_iou,
            image_size=self.detector_image_size,
            diatoms_only=True,
        )[0]
        if not detections:
            return []
        crops = [self._crop(image, detection) for detection in detections]
        gate_probabilities = (
            self.artifact_gate.predict(crops) if self.artifact_gate is not None
            else [None] * len(crops)
        )
        accepted_indices = [
            index for index, probability in enumerate(gate_probabilities)
            if probability is None or self.artifact_gate.accepts(probability)
        ]
        species_by_index: dict[int, np.ndarray] = {}
        genus_results_by_index: dict[int, list[Candidate]] = {}
        if accepted_indices:
            genus_embeddings, species_embeddings = self._embed_crops(
                [crops[index] for index in accepted_indices]
            )
            genus_results = self.genus_index.search(genus_embeddings, top_k=self.top_k)
            for local_index, crop_index in enumerate(accepted_indices):
                species_by_index[crop_index] = species_embeddings[local_index]
                genus_results_by_index[crop_index] = genus_results[local_index]
        predictions: list[Prediction] = []
        crop_dir: Path | None = None
        if self.save_crops and self.output_dir is not None:
            crop_dir = self.output_dir / "crops" / path.stem
            crop_dir.mkdir(parents=True, exist_ok=True)
        for index, (detection, crop) in enumerate(zip(detections, crops)):
            gate_probability = gate_probabilities[index]
            gate_accepted = gate_probability is None or index in accepted_indices
            if not gate_accepted:
                decision = ClassificationDecision(
                    status="unknown_genus", genus=None, species=None,
                    genus_similarity=-1.0, species_similarity=None,
                    genus_margin=0.0, species_margin=None,
                    genus_agreement=0.0, species_agreement=None,
                )
                genus_candidates = []
                species_candidates = []
                rejection_stage = "resnet_gate"
            else:
                genus_candidates = genus_results_by_index[index]
                genus_decision = self.decision_logic.decide_genus(genus_candidates)
                species_candidates = []
                species_decision = None
                if genus_decision.accepted and genus_decision.label is not None:
                    species_candidates = self.species_index.search(
                        species_by_index[index], top_k=self.top_k,
                        genus_filter=genus_decision.label,
                    )[0]
                    species_decision = self.decision_logic.decide_species(species_candidates)
                decision = self.decision_logic.combine(genus_decision, species_decision)
                rejection_stage = (
                    "dino_genus" if decision.status == "unknown_genus"
                    else "dino_species" if decision.status == "known_genus_unknown_species"
                    else None
                )
            crop_path: str | None = None
            if crop_dir is not None:
                saved_crop = crop_dir / f"crop_{index:04d}.png"
                crop.save(saved_crop)
                crop_path = str(saved_crop)
            predictions.append(
                Prediction(
                    bbox=detection.xyxy,
                    polygon=detection.polygon,
                    detection_confidence=detection.confidence,
                    decision=decision,
                    genus_candidates=tuple(genus_candidates),
                    species_candidates=tuple(species_candidates),
                    gate_probability=gate_probability,
                    gate_accepted=gate_accepted,
                    rejection_stage=rejection_stage,
                    crop_path=crop_path,
                )
            )
        return predictions

    def predict_batch(self, image_paths: Iterable[str | Path]) -> dict[str, list[Prediction]]:
        return {str(path): self.predict_single(path) for path in image_paths}
