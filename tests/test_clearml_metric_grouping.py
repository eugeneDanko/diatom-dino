from __future__ import annotations

import unittest

from core.clearml_logger import metric_chart_coordinates
from core.stage_reporting import reporting_config, stage_chart_coordinates


class ClearMLMetricGroupingTest(unittest.TestCase):
    def test_training_metrics_are_grouped_by_kind(self) -> None:
        self.assertEqual(metric_chart_coordinates("train/loss_species"), ("loss", "train/species"))
        self.assertEqual(
            metric_chart_coordinates("val/accuracy_species"), ("accuracy", "val/species")
        )
        self.assertEqual(
            metric_chart_coordinates("val/precision_macro_genus"),
            ("precision", "val/macro_genus"),
        )
        self.assertEqual(
            metric_chart_coordinates("val/recall_at_5_species"),
            ("recall", "val/at_5_species"),
        )
        self.assertEqual(metric_chart_coordinates("val/f1_macro_species"), ("f1", "val/macro_species"))

    def test_retrieval_strategy_becomes_series_scope(self) -> None:
        self.assertEqual(
            metric_chart_coordinates("weighted_k3/accuracy_species"),
            ("accuracy", "weighted_k3/species"),
        )

    def test_stage1_keeps_losses_separate_and_hides_species_metrics(self) -> None:
        config = reporting_config("stage1")
        self.assertEqual(
            stage_chart_coordinates("train/loss_species", config),
            ("Optimization/Losses", "train/loss_species"),
        )
        self.assertEqual(
            stage_chart_coordinates("val/f1_macro_genus", config),
            ("Target metrics/Genus", "val/f1_macro_genus"),
        )
        self.assertIsNone(stage_chart_coordinates("val/f1_macro_species", config))

    def test_stage2_logs_genus_as_guardrail(self) -> None:
        config = reporting_config("stage2")
        self.assertEqual(
            stage_chart_coordinates("val/f1_macro_species", config),
            ("Target metrics/Species", "val/f1_macro_species"),
        )
        self.assertEqual(
            stage_chart_coordinates("val/f1_macro_genus", config),
            ("Supporting metrics/Genus", "val/f1_macro_genus"),
        )


if __name__ == "__main__":
    unittest.main()
