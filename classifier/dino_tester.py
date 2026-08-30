"""Evaluate gallery-based genus/species retrieval on independent GT crops."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from core.base_tester import BaseTester
from core.clearml_logger import ExperimentLogger
from .crop_dataset import LabelCodec
from .dino_backbone import DINOBackbone
from .embeddings_vis import plot_embeddings
from .metrics import hierarchical_retrieval_predict, retrieval_metrics, retrieval_predict
from .projection_head import HierarchicalProjectionHead


class DINOTester(BaseTester):
    def __init__(
        self,
        *,
        backbone: DINOBackbone,
        projection_head: HierarchicalProjectionHead,
        gallery_loader: Any,
        test_loader: Any,
        label_codec: LabelCodec,
        output_dir: str | Path,
        logger: ExperimentLogger | None = None,
        k: int = 1,
        retrieval_strategies: Sequence[Mapping[str, Any]] | None = None,
        primary_strategy: str | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, logger=logger)
        self.backbone = backbone
        self.projection_head = projection_head.to(backbone.device)
        self.gallery_loader = gallery_loader
        self.test_loader = test_loader
        self.label_codec = label_codec
        self.k = int(k)
        self.retrieval_strategies = [dict(value) for value in (retrieval_strategies or [])]
        if not self.retrieval_strategies:
            self.retrieval_strategies = [{"name": f"nearest_k{self.k}", "method": "nearest", "k": self.k}]
        names = [str(value.get("name", "")).strip() for value in self.retrieval_strategies]
        if not all(names) or len(names) != len(set(names)):
            raise ValueError("Every retrieval strategy must have a unique non-empty name")
        self.primary_strategy = primary_strategy or names[0]
        if self.primary_strategy not in names:
            raise ValueError(f"Unknown primary retrieval strategy: {self.primary_strategy}")
        self.prediction_rows: list[dict[str, Any]] = []
        self.strategy_prediction_rows: dict[str, list[dict[str, Any]]] = {}
        self.test_genus_embeddings: np.ndarray | None = None
        self.test_species_embeddings: np.ndarray | None = None

    @torch.no_grad()
    def _embed(self, loader: Any) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str]]:
        self.backbone.set_train_mode(False)
        self.projection_head.eval()
        genus_values: list[np.ndarray] = []
        species_values: list[np.ndarray] = []
        genus_labels: list[int] = []
        species_labels: list[int] = []
        crop_ids: list[str] = []
        for batch in loader:
            images = batch["image"].to(self.backbone.device, non_blocking=True)
            raw = self.backbone.extract_features(images)
            genus_embedding, species_embedding = self.projection_head(raw)
            genus_values.append(genus_embedding.cpu().numpy())
            species_values.append(species_embedding.cpu().numpy())
            genus_labels.extend(torch.as_tensor(batch["genus_label"]).tolist())
            species_labels.extend(torch.as_tensor(batch["species_label"]).tolist())
            crop_ids.extend(str(value) for value in batch["crop_id"])
        return (
            np.concatenate(genus_values),
            np.concatenate(species_values),
            genus_labels,
            species_labels,
            crop_ids,
        )

    def run_test(self) -> dict[str, Any]:
        gallery_genus, gallery_species, gallery_genus_labels, gallery_species_labels, _ = self._embed(
            self.gallery_loader
        )
        test_genus, test_species, true_genus, true_species, crop_ids = self._embed(self.test_loader)
        self.test_genus_embeddings = test_genus
        self.test_species_embeddings = test_species
        metrics: dict[str, Any] = {"num_gallery": len(gallery_genus), "num_test": len(test_genus)}
        id_to_genus = self.label_codec.id_to_genus
        id_to_species = self.label_codec.id_to_species
        protocol_by_crop: dict[str, str] = {}
        dataset = getattr(self.test_loader, "dataset", None)
        if dataset is not None and "protocol_target" in getattr(dataset, "table", ()):
            protocol_by_crop = {
                str(row[dataset.id_column]): str(row["protocol_target"])
                for _, row in dataset.table.iterrows()
            }
        for strategy in self.retrieval_strategies:
            name = str(strategy["name"])
            predicted_genus = retrieval_predict(
                test_genus, gallery_genus, gallery_genus_labels, strategy
            )
            global_predicted_species = retrieval_predict(
                test_species, gallery_species, gallery_species_labels, strategy
            )
            predicted_species = hierarchical_retrieval_predict(
                test_species,
                gallery_species,
                predicted_genus,
                gallery_genus_labels,
                gallery_species_labels,
                strategy,
            )
            strategy_metrics = {
                **retrieval_metrics(true_genus, predicted_genus, "genus"),
                **retrieval_metrics(true_species, predicted_species, "species"),
                **retrieval_metrics(true_species, global_predicted_species, "species_global"),
            }
            metrics.update({f"{name}/{key}": value for key, value in strategy_metrics.items()})
            rows = [
                {
                    "crop_id": crop_id,
                    "true_genus": id_to_genus.get(int(genus_true), str(genus_true)),
                    "predicted_genus": id_to_genus.get(int(genus_pred), str(genus_pred)),
                    "true_species": id_to_species.get(int(species_true), str(species_true)),
                    "predicted_species": id_to_species.get(int(species_pred), str(species_pred)),
                    "protocol_target": protocol_by_crop.get(str(crop_id), "all"),
                }
                for crop_id, genus_true, genus_pred, species_true, species_pred in zip(
                    crop_ids, true_genus, predicted_genus, true_species, predicted_species
                )
            ]
            self.strategy_prediction_rows[name] = rows
            for target in sorted({row["protocol_target"] for row in rows} - {"all"}):
                target_rows = [row for row in rows if row["protocol_target"] == target]
                metrics[f"{name}/protocol/{target}/count"] = len(target_rows)
                metrics[f"{name}/protocol/{target}/accuracy_genus"] = sum(
                    row["true_genus"] == row["predicted_genus"] for row in target_rows
                ) / max(1, len(target_rows))
                metrics[f"{name}/protocol/{target}/accuracy_species"] = sum(
                    row["true_species"] == row["predicted_species"] for row in target_rows
                ) / max(1, len(target_rows))
        self.prediction_rows = self.strategy_prediction_rows[self.primary_strategy]
        return metrics

    def generate_report(self, results: dict[str, Any]) -> Path:
        metrics_path = self.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        predictions_path = self.output_dir / "predictions.csv"
        for strategy, rows in self.strategy_prediction_rows.items():
            strategy_path = self.output_dir / f"predictions_{strategy}.csv"
            if rows:
                with strategy_path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        if self.prediction_rows:
            with predictions_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.prediction_rows[0]))
                writer.writeheader()
                writer.writerows(self.prediction_rows)
            self._plot_confusion(
                [row["true_genus"] for row in self.prediction_rows],
                [row["predicted_genus"] for row in self.prediction_rows],
                self.output_dir / "confusion_genus.png",
                "Genus confusion matrix",
            )
            self._plot_confusion(
                [row["true_species"] for row in self.prediction_rows],
                [row["predicted_species"] for row in self.prediction_rows],
                self.output_dir / "confusion_species.png",
                "Species confusion matrix",
            )
            genus_plot = plot_embeddings(
                self.test_genus_embeddings,
                [row["true_genus"] for row in self.prediction_rows],
                output_path=self.output_dir / "embeddings_genus.png",
                title="Genus embeddings",
            )
            species_plot = plot_embeddings(
                self.test_species_embeddings,
                [row["true_species"] for row in self.prediction_rows],
                output_path=self.output_dir / "embeddings_species.png",
                title="Species embeddings",
            )
            if genus_plot is not None:
                self.logger.log_artifact("embeddings_genus", genus_plot)
            if species_plot is not None:
                self.logger.log_artifact("embeddings_species", species_plot)
        return metrics_path

    def _plot_confusion(
        self,
        truth: list[str],
        predictions: list[str],
        path: Path,
        title: str,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
        except ImportError:  # pragma: no cover - optional reporting dependency
            return
        labels = sorted(set(truth) | set(predictions))
        figure_size = max(6.0, min(18.0, 0.45 * len(labels)))
        figure, axis = plt.subplots(figsize=(figure_size, figure_size))
        ConfusionMatrixDisplay.from_predictions(
            truth,
            predictions,
            labels=labels,
            normalize="true",
            xticks_rotation=90,
            colorbar=False,
            ax=axis,
        )
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        self.logger.log_artifact(path.stem, path)
