"""Common evaluation lifecycle."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from .clearml_logger import ExperimentLogger, NullLogger


class BaseTester(ABC):
    def __init__(
        self,
        *,
        output_dir: str | Path,
        logger: ExperimentLogger | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or NullLogger()

    @abstractmethod
    def run_test(self) -> dict[str, Any]: ...

    def generate_report(self, results: Mapping[str, Any]) -> Path:
        path = self.output_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(results, stream, ensure_ascii=False, indent=2, default=str)
        return path

    def test(self) -> dict[str, Any]:
        results = self.run_test()
        scalar_metrics = {
            key: float(value)
            for key, value in results.items()
            if isinstance(value, (int, float))
        }
        self.logger.log_metrics(scalar_metrics, step=0)
        report = self.generate_report(results)
        self.logger.log_artifact("test_report", report)
        self.logger.close()
        return results

