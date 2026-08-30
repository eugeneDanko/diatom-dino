"""Small logging facade; training code does not depend directly on ClearML."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from .stage_reporting import stage_chart_coordinates


def metric_chart_coordinates(key: str) -> tuple[str, str]:
    """Map a flat metric key to a ClearML chart title and series."""

    parts = str(key).split("/")
    metric = parts[-1]
    scope = "/".join(parts[:-1]) or "metrics"
    families = (
        ("loss", "loss"),
        ("accuracy", "accuracy"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
    )
    for token, title in families:
        if token in metric:
            descriptor = metric.replace(token, "").strip("_") or "total"
            return title, f"{scope}/{descriptor}"
    if metric.startswith("cosine_"):
        return "cosine_similarity", f"{scope}/{metric.removeprefix('cosine_')}"
    if metric.startswith("num_") or metric.endswith("_count"):
        return "counts", f"{scope}/{metric.removeprefix('num_').removesuffix('_count')}"
    if metric.startswith("open_set_"):
        return "open_set", f"{scope}/{metric.removeprefix('open_set_')}"
    return "metrics", f"{scope}/{metric}"


class ExperimentLogger(ABC):
    enabled = False

    @abstractmethod
    def log_parameters(self, parameters: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None: ...

    @abstractmethod
    def log_artifact(self, name: str, path: str | Path) -> None: ...

    def log_figure(self, title: str, figure: Any, step: int, *, series: str = "validation") -> None:
        return None

    def log_table(self, title: str, table: Any, step: int, *, series: str = "validation") -> None:
        return None

    @abstractmethod
    def close(self) -> None: ...


class NullLogger(ExperimentLogger):
    """No-op implementation used for tests and disabled tracking."""

    def log_parameters(self, parameters: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        return None

    def log_artifact(self, name: str, path: str | Path) -> None:
        return None

    def close(self) -> None:
        return None


class ClearMLLogger(ExperimentLogger):
    enabled = True

    def __init__(
        self,
        *,
        project_name: str,
        task_name: str,
        tags: list[str] | None = None,
        output_uri: str | None = None,
        reuse_last_task_id: bool = False,
        reporting: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            from clearml import Task
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ClearML logging is enabled but clearml is not installed") from exc
        self.task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
            output_uri=output_uri,
            reuse_last_task_id=reuse_last_task_id,
        )
        self.logger = self.task.get_logger()
        self.reporting = dict(reporting or {})

    def log_parameters(self, parameters: Mapping[str, Any]) -> None:
        # JSON round-trip makes Path, tuples and other common config values stable.
        normalized = json.loads(json.dumps(parameters, default=str))
        self.task.connect(normalized)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            coordinates = (
                stage_chart_coordinates(key, self.reporting)
                if self.reporting.get("stage")
                else metric_chart_coordinates(key)
            )
            if coordinates is None:
                continue
            title, series = coordinates
            self.logger.report_scalar(title, series, float(value), iteration=step)

    def log_artifact(self, name: str, path: str | Path) -> None:
        self.task.upload_artifact(name=name, artifact_object=str(path))

    def log_figure(self, title: str, figure: Any, step: int, *, series: str = "validation") -> None:
        self.logger.report_matplotlib_figure(
            title=title,
            series=series,
            iteration=int(step),
            figure=figure,
            report_image=True,
        )

    def log_table(self, title: str, table: Any, step: int, *, series: str = "validation") -> None:
        self.logger.report_table(
            title=title,
            series=series,
            iteration=int(step),
            table_plot=table,
        )

    def close(self) -> None:
        self.task.close()


def create_experiment_logger(config: Mapping[str, Any], default_task_name: str) -> ExperimentLogger:
    section = dict(config.get("clearml", {}))
    if not section.get("enabled", False):
        return NullLogger()
    return ClearMLLogger(
        project_name=section.get("project_name", "DiatomDINO"),
        task_name=section.get("task_name", default_task_name),
        tags=list(section.get("tags", [])),
        output_uri=section.get("output_uri"),
        reuse_last_task_id=bool(section.get("reuse_last_task_id", False)),
        reporting=config.get("reporting", {}),
    )
