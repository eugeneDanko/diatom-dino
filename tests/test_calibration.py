from __future__ import annotations

import unittest

from inference.calibration import CalibrationRecord, binary_score_metrics, calibrate_level
from inference.open_set_calibration import OpenSetScore, calibrate_hierarchy, decision_status


class CalibrationTest(unittest.TestCase):
    def test_binary_score_metrics_reports_perfect_separation(self) -> None:
        metrics = binary_score_metrics([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["fpr_at_95_tpr"], 0.0)

    def test_calibration_separates_known_from_unknown(self) -> None:
        records = [
            CalibrationRecord(0.95, 0.2, 1.0, True),
            CalibrationRecord(0.90, 0.1, 0.8, True),
            CalibrationRecord(0.55, 0.01, 0.4, False),
            CalibrationRecord(0.45, 0.02, 0.2, False),
        ]
        calibrated = calibrate_level(records)
        self.assertGreaterEqual(calibrated.balanced_accuracy, 0.99)

    def test_recall_first_calibration_preserves_positive_examples(self) -> None:
        records = [
            CalibrationRecord(0.95, 0.2, 1.0, True),
            CalibrationRecord(0.70, 0.1, 0.8, True),
            CalibrationRecord(0.65, 0.05, 0.6, False),
            CalibrationRecord(0.40, 0.01, 0.2, False),
        ]
        calibrated = calibrate_level(records, minimum_positive_recall=1.0)
        self.assertEqual(calibrated.positive_recall, 1.0)
        self.assertEqual(
            calibrated.selection_reason,
            "maximum_rejection_at_minimum_positive_recall",
        )

    def test_hierarchy_distinguishes_unknown_species_and_genus(self) -> None:
        def score(sample_id, target, genus_similarity, species_similarity):
            return OpenSetScore(
                sample_id, target,
                "Navicula" if target != "unknown_genus" else "Pinnularia",
                "Navicula_seen" if target == "known" else "Navicula_novel",
                "Navicula", genus_similarity, 0.2, 1.0,
                "Navicula_seen", species_similarity, 0.2, 1.0,
            )

        scores = [
            score("known-1", "known", 0.95, 0.95),
            score("known-2", "known", 0.90, 0.90),
            score("novel-species-1", "unknown_species", 0.92, 0.55),
            score("novel-species-2", "unknown_species", 0.88, 0.50),
            score("novel-genus-1", "unknown_genus", 0.40, 0.40),
            score("novel-genus-2", "unknown_genus", 0.35, 0.35),
        ]
        genus, species, _ = calibrate_hierarchy(
            scores,
            gallery_genera={"Navicula"},
            gallery_species={"Navicula_seen"},
            minimum_genus_recall=1.0,
            minimum_species_recall=1.0,
        )
        self.assertEqual(decision_status(scores[0], genus, species), "known_species")
        self.assertEqual(
            decision_status(scores[2], genus, species),
            "known_genus_unknown_species",
        )
        self.assertEqual(decision_status(scores[4], genus, species), "unknown_genus")


if __name__ == "__main__":
    unittest.main()
