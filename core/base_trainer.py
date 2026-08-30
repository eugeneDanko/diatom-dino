"""Reusable training lifecycle with checkpointing and experiment logging."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .clearml_logger import ExperimentLogger, NullLogger


class BaseTrainer(ABC):
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        output_dir: str | Path,
        logger: ExperimentLogger | None = None,
        monitor: str = "val/loss",
        mode: str = "min",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.config = dict(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or NullLogger()
        self.monitor = monitor
        self.monitor_components = {
            str(key): float(weight)
            for key, weight in dict(
                config.get("training", {}).get("monitor_components", {})
            ).items()
        }
        self.mode = mode
        self.best_value = math.inf if mode == "min" else -math.inf
        self.start_epoch = 0
        early_stopping = dict(config.get("training", {}).get("early_stopping", {}))
        self.early_stopping_enabled = bool(early_stopping.get("enabled", False))
        self.early_stopping_patience = max(1, int(early_stopping.get("patience", 5)))
        self.early_stopping_min_delta = max(0.0, float(early_stopping.get("min_delta", 0.0)))
        self.reporting = dict(config.get("reporting", {}))
        self._progress_rows: list[dict[str, Any]] = []

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, float]: ...

    @abstractmethod
    def validate(self, epoch: int) -> dict[str, float]: ...

    @abstractmethod
    def save_checkpoint(self, path: str | Path, epoch: int, metrics: Mapping[str, float]) -> Path: ...

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        return None

    def _is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value - self.early_stopping_min_delta
        return value > self.best_value + self.early_stopping_min_delta

    def fit(self, epochs: int) -> dict[str, float]:
        self.logger.log_parameters(self.config)
        final_metrics: dict[str, float] = {}
        epochs_without_improvement = 0
        for epoch in range(self.start_epoch, epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            final_metrics = {
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
            }
            if self.monitor_components:
                missing_components = [
                    key for key in self.monitor_components if key not in final_metrics
                ]
                if missing_components:
                    available = ", ".join(sorted(final_metrics))
                    raise KeyError(
                        f"Monitor components missing: {missing_components}; available: {available}"
                    )
                monitored = sum(
                    weight * float(final_metrics[key])
                    for key, weight in self.monitor_components.items()
                )
                final_metrics["checkpoint/score"] = monitored
            else:
                monitored = final_metrics.get(self.monitor)
            if monitored is None:
                available = ", ".join(sorted(final_metrics))
                raise KeyError(f"Monitor '{self.monitor}' missing; available: {available}")
            improved = self._is_better(float(monitored))
            if improved:
                self.best_value = float(monitored)
                epochs_without_improvement = 0
                checkpoint = self.save_checkpoint(self.output_dir / "best.pt", epoch, final_metrics)
                self.logger.log_artifact("best_checkpoint", checkpoint)
            else:
                epochs_without_improvement += 1
            self.logger.log_metrics(final_metrics, epoch)
            self.save_checkpoint(self.output_dir / "last.pt", epoch, final_metrics)
            if self.reporting.get("stage"):
                row: dict[str, Any] = {"epoch": epoch + 1}
                selected = (
                    list(self.reporting.get("target_metrics", []))
                    + list(self.reporting.get("supporting_metrics", []))
                )
                for key in selected:
                    if key in final_metrics:
                        row[key.removeprefix("val/")] = float(final_metrics[key])
                row["checkpoint_score"] = float(monitored)
                row["is_best"] = improved
                row["epochs_without_improvement"] = epochs_without_improvement
                self._progress_rows.append(row)
                self.logger.log_table(
                    "Stage progress",
                    pd.DataFrame(self._progress_rows),
                    epoch,
                    series=str(self.reporting.get("stage")),
                )
            self.on_epoch_end(epoch, final_metrics)
            if (
                self.early_stopping_enabled
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                self.logger.log_metrics(
                    {"early_stopping/epochs_without_improvement": epochs_without_improvement},
                    epoch,
                )
                break
        self.logger.close()
        return final_metrics
