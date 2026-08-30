from __future__ import annotations

import unittest

import numpy as np

from classifier.metrics import (
    hierarchical_knn_predict,
    hierarchical_retrieval_predict,
    prototype_weighted_predict,
    prototype_softmax_predict,
    retrieval_metrics,
    weighted_knn_predict,
)


class HierarchicalMetricsTest(unittest.TestCase):
    def test_species_search_is_restricted_to_predicted_genus(self) -> None:
        references = np.asarray(
            [
                [1.0, 0.0],   # genus 0, species 10
                [0.8, 0.2],   # genus 0, species 11
                [0.99, 0.01], # genus 1, species 20; globally closest
            ],
            dtype=np.float32,
        )
        prediction = hierarchical_knn_predict(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            references,
            predicted_query_genera=[0],
            reference_genera=[0, 0, 1],
            reference_species=[10, 11, 20],
            k=1,
        )
        self.assertEqual(prediction.tolist(), [10])

    def test_weighted_knn_can_outvote_single_nearest_reference(self) -> None:
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        references = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float32)
        prediction = weighted_knn_predict(
            query, references, [0, 1, 1], k=3, temperature=0.05
        )
        self.assertEqual(prediction.tolist(), [1])

    def test_prototype_weighted_predicts_compact_class(self) -> None:
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        references = np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
        )
        prediction = prototype_weighted_predict(query, references, [2, 2, 3, 3])
        self.assertEqual(prediction.tolist(), [2])

    def test_hierarchical_weighted_search_stays_inside_genus(self) -> None:
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        references = np.asarray([[0.8, 0.2], [0.7, 0.3], [1.0, 0.0]], dtype=np.float32)
        prediction = hierarchical_retrieval_predict(
            query,
            references,
            [0],
            [0, 0, 1],
            [10, 11, 20],
            {"method": "weighted_knn", "k": 3},
        )
        self.assertIn(int(prediction[0]), {10, 11})

    def test_retrieval_metrics_include_separate_macro_scores(self) -> None:
        metrics = retrieval_metrics([0, 0, 1], [0, 1, 1], "species")
        self.assertAlmostEqual(metrics["accuracy_species"], 2 / 3)
        self.assertIn("precision_macro_species", metrics)
        self.assertIn("recall_macro_species", metrics)
        self.assertIn("f1_macro_species", metrics)

    def test_prototype_softmax_uses_three_references_per_class(self) -> None:
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        references = np.asarray(
            [
                [1.0, 0.0], [0.99, 0.01], [0.98, 0.02],
                [0.0, 1.0], [0.1, 0.9], [0.2, 0.8],
            ],
            dtype=np.float32,
        )
        prediction = prototype_softmax_predict(
            query, references, [5, 5, 5, 6, 6, 6], k=3, temperature=0.03
        )
        self.assertEqual(prediction.tolist(), [5])

    def test_prototype_softmax_rejects_zero_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            prototype_softmax_predict(
                np.asarray([[1.0, 0.0]]),
                np.asarray([[1.0, 0.0]]),
                [0],
                temperature=0,
            )


if __name__ == "__main__":
    unittest.main()
