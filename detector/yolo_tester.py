"""Detector evaluation on a YOLO data.yaml test split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.base_tester import BaseTester
from core.clearml_logger import ExperimentLogger
from .yolo_model import YOLOModel


class YOLOTester(BaseTester):
    def __init__(
        self,
        *,
        model: YOLOModel,
        data_yaml: str | Path,
        output_dir: str | Path,
        logger: ExperimentLogger | None = None,
        split: str = "test",
        image_size: int = 1024,
        confidence: float = 0.001,
        iou: float = 0.7,
    ) -> None:
        super().__init__(output_dir=output_dir, logger=logger)
        self.model = model
        self.data_yaml = str(data_yaml)
        self.split = split
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou

    def run_test(self) -> dict[str, Any]:
        native = self.model._ensure_loaded()
        result = native.val(
            data=self.data_yaml,
            split=self.split,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            device=self.model.device,
            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            plots=True,
            verbose=False,
        )
        metrics = getattr(result, "results_dict", {}) or {}
        return {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

