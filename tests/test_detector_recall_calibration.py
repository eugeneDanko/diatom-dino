from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import yaml

from detector.recall_calibration import (
    build_ultralytics_calibration_yaml,
    select_recall_first,
    sweep_thresholds,
)
from detector.yolo_model import Detection


class RecallCalibrationTests(unittest.TestCase):
    def test_sweep_computes_metrics_at_each_actual_confidence(self) -> None:
        predictions = {
            "1.png": [
                Detection((0, 0, 10, 10), 0.90, 0),
                Detection((20, 20, 30, 30), 0.40, 0),
                Detection((40, 40, 50, 50), 0.20, 0),
            ]
        }
        ground_truth = {"1.png": [(0, 0, 10, 10), (20, 20, 30, 30)]}
        rows = sweep_thresholds(
            predictions, ground_truth, thresholds=[0.1, 0.5], match_iou=0.5
        )
        self.assertEqual(
            (rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]), (2, 1, 0)
        )
        self.assertAlmostEqual(rows[0]["metrics/recall(B)"], 1.0)
        self.assertEqual(
            (rows[1]["tp"], rows[1]["fp"], rows[1]["fn"]), (1, 0, 1)
        )
        self.assertAlmostEqual(rows[1]["metrics/precision(B)"], 1.0)

    def test_builds_ultralytics_yaml_with_calibration_mapped_to_val(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "datasetYOLO"
            (dataset / "images" / "calibration").mkdir(parents=True)
            source = dataset / "data.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "path": str(dataset),
                        "train": "images/train",
                        "val": "images/val",
                        "calibration": "images/calibration",
                        "test": "images/test",
                        "names": {0: "diatom"},
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "artifacts" / "data.calibration.yaml"
            result = build_ultralytics_calibration_yaml(source, destination)
            adapted = yaml.safe_load(result.read_text(encoding="utf-8"))
            self.assertEqual(Path(adapted["val"]), dataset / "images" / "calibration")
            self.assertNotIn("calibration", adapted)

    def test_uses_existing_calibration_txt_for_legacy_materialized_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "datasetYOLO"
            dataset.mkdir()
            (dataset / "calibration.txt").write_text("/data/images/1.png\n", encoding="utf-8")
            source = dataset / "data.yaml"
            source.write_text(yaml.safe_dump({"path": str(dataset), "val": "val.txt"}), encoding="utf-8")
            adapted = yaml.safe_load(
                build_ultralytics_calibration_yaml(source, dataset / "adapted.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(Path(adapted["val"]), dataset / "calibration.txt")

    def test_maximizes_precision_after_enforcing_recall(self) -> None:
        rows = [
            {"confidence": 0.1, "metrics/precision(B)": 0.60, "metrics/recall(B)": 0.96},
            {"confidence": 0.2, "metrics/precision(B)": 0.75, "metrics/recall(B)": 0.91},
            {"confidence": 0.3, "metrics/precision(B)": 0.86, "metrics/recall(B)": 0.84},
        ]
        selected = select_recall_first(rows, minimum_recall=0.90)
        self.assertEqual(selected["confidence"], 0.2)
        self.assertEqual(selected["selection_reason"], "maximum_precision_subject_to_minimum_recall")

    def test_falls_back_to_f2_when_recall_target_is_unreachable(self) -> None:
        rows = [
            {"confidence": 0.1, "metrics/precision(B)": 0.55, "metrics/recall(B)": 0.80},
            {"confidence": 0.2, "metrics/precision(B)": 0.80, "metrics/recall(B)": 0.70},
        ]
        selected = select_recall_first(rows, minimum_recall=0.95)
        self.assertEqual(selected["confidence"], 0.1)
        self.assertEqual(selected["selection_reason"], "maximum_f2_no_threshold_met_minimum_recall")


if __name__ == "__main__":
    unittest.main()
