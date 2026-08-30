"""Ultralytics-native trainer integrated with the project logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.base_trainer import BaseTrainer
from core.clearml_logger import ExperimentLogger
from .yolo_model import YOLOModel


class YOLOTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        model: YOLOModel,
        config: Mapping[str, Any],
        output_dir: str | Path,
        logger: ExperimentLogger | None = None,
    ) -> None:
        super().__init__(
            config=config,
            output_dir=output_dir,
            logger=logger,
            monitor="metrics/recall",
            mode="max",
        )
        self.model = model
        self.results: Any | None = None

    def fit(self, epochs: int | None = None) -> dict[str, float]:
        native = self.model._ensure_loaded()
        train_config = dict(self.config.get("training", self.config))
        if epochs is not None:
            train_config["epochs"] = int(epochs)
        train_config.setdefault("project", str(self.output_dir.parent))
        train_config.setdefault("name", self.output_dir.name)
        train_config.setdefault("device", self.model.device)
        self.logger.log_parameters(self.config)
        self.results = native.train(**train_config)
        metrics = self._extract_metrics(self.results)
        self.logger.log_metrics(metrics, step=int(train_config.get("epochs", 0)))
        best_path = Path(str(getattr(self.results, "save_dir", self.output_dir)))
        candidate = best_path / "weights" / "best.pt"
        if candidate.exists():
            self.logger.log_artifact("best_checkpoint", candidate)
        self.logger.close()
        return metrics

    @staticmethod
    def _extract_metrics(results: Any) -> dict[str, float]:
        metrics = getattr(results, "results_dict", {}) or {}
        return {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}

    # The native Ultralytics trainer owns its epoch lifecycle.
    def train_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError("YOLOTrainer uses Ultralytics' native fit loop")

    def validate(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError("YOLOTrainer uses Ultralytics' native validation loop")

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: Mapping[str, float]) -> Path:
        return self.model.save(path)
