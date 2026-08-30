from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.base_trainer import BaseTrainer


class _Trainer(BaseTrainer):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(
            config={
                "training": {
                    "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0}
                }
            },
            output_dir=output_dir,
            monitor="val/score",
            mode="max",
        )
        self.validation_calls = 0

    def train_epoch(self, epoch: int) -> dict[str, float]:
        return {"loss": float(epoch)}

    def validate(self, epoch: int) -> dict[str, float]:
        self.validation_calls += 1
        return {"score": 1.0 - epoch * 0.1}

    def save_checkpoint(self, path: str | Path, epoch: int, metrics) -> Path:
        output = Path(path)
        output.write_text(str(epoch), encoding="utf-8")
        return output


class BaseTrainerEarlyStoppingTest(unittest.TestCase):
    def test_stops_after_configured_patience(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = _Trainer(Path(directory))
            trainer.fit(epochs=10)
            self.assertEqual(trainer.validation_calls, 3)
            self.assertEqual((Path(directory) / "best.pt").read_text(encoding="utf-8"), "0")
            self.assertEqual((Path(directory) / "last.pt").read_text(encoding="utf-8"), "2")

    def test_weighted_monitor_components_select_checkpoint(self) -> None:
        class Composite(_Trainer):
            def __init__(self, output_dir: Path) -> None:
                BaseTrainer.__init__(
                    self,
                    config={"training": {"monitor_components": {"val/a": 0.7, "val/b": 0.3}}},
                    output_dir=output_dir,
                    monitor="unused",
                    mode="max",
                )
                self.validation_calls = 0

            def validate(self, epoch: int) -> dict[str, float]:
                self.validation_calls += 1
                return {"a": 0.4 + epoch * 0.1, "b": 0.8 - epoch * 0.1}

        with tempfile.TemporaryDirectory() as directory:
            trainer = Composite(Path(directory))
            metrics = trainer.fit(epochs=2)
            self.assertAlmostEqual(metrics["checkpoint/score"], 0.7 * 0.5 + 0.3 * 0.7)
            self.assertEqual((Path(directory) / "best.pt").read_text(encoding="utf-8"), "1")


if __name__ == "__main__":
    unittest.main()
