"""A stable interface around Ultralytics YOLO/YOLO-OBB results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.base_model import BaseModel
from .obb_utils import polygon_to_xyxy


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    polygon: tuple[tuple[float, float], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YOLOModel(BaseModel):
    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        device: str = "cuda",
        diatom_class_id: int = 0,
    ) -> None:
        self.weights = str(weights) if weights is not None else None
        self.device = device
        self.diatom_class_id = int(diatom_class_id)
        self.model: Any | None = None

    def load(self, source: str | Path | None = None) -> "YOLOModel":
        weights = str(source) if source is not None else self.weights
        if not weights:
            raise ValueError("YOLO weights/model name is required")
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Install ultralytics to use YOLOModel") from exc
        self.weights = weights
        self.model = YOLO(weights)
        return self

    def to_device(self, device: str | Any) -> "YOLOModel":
        self.device = str(device)
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(self.device)
        return self

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("YOLO model has not been loaded")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output))
        return output

    def _ensure_loaded(self) -> Any:
        if self.model is None:
            self.load()
        return self.model

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    def _parse_result(self, result: Any, class_filter: set[int] | None) -> list[Detection]:
        parsed: list[Detection] = []
        if getattr(result, "obb", None) is not None:
            obb = result.obb
            polygons = self._to_numpy(obb.xyxyxyxy)
            confidences = self._to_numpy(obb.conf).reshape(-1)
            classes = self._to_numpy(obb.cls).astype(int).reshape(-1)
            for polygon, confidence, class_id in zip(polygons, confidences, classes):
                if class_filter is not None and int(class_id) not in class_filter:
                    continue
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                xyxy = polygon_to_xyxy(points)
                parsed.append(
                    Detection(
                        xyxy=tuple(float(value) for value in xyxy),
                        confidence=float(confidence),
                        class_id=int(class_id),
                        polygon=tuple(tuple(float(value) for value in point) for point in points),
                    )
                )
            return parsed

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return parsed
        coordinates = self._to_numpy(boxes.xyxy)
        confidences = self._to_numpy(boxes.conf).reshape(-1)
        classes = self._to_numpy(boxes.cls).astype(int).reshape(-1)
        for xyxy, confidence, class_id in zip(coordinates, confidences, classes):
            if class_filter is not None and int(class_id) not in class_filter:
                continue
            parsed.append(
                Detection(
                    xyxy=tuple(float(value) for value in xyxy),
                    confidence=float(confidence),
                    class_id=int(class_id),
                )
            )
        return parsed

    def predict(
        self,
        source: str | Path | np.ndarray | Iterable[str | Path | np.ndarray],
        *,
        confidence: float = 0.25,
        iou: float = 0.7,
        image_size: int | tuple[int, int] = 1024,
        diatoms_only: bool = True,
        **kwargs: Any,
    ) -> list[list[Detection]]:
        model = self._ensure_loaded()
        results = model.predict(
            source=source,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            device=self.device,
            verbose=False,
            **kwargs,
        )
        class_filter = {self.diatom_class_id} if diatoms_only else None
        return [self._parse_result(result, class_filter) for result in results]

