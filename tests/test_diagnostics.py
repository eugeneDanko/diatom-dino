from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from classifier.crop_dataset import LabelCodec
from classifier.diagnostics import run_retrieval_diagnostics
from core.clearml_logger import NullLogger


class RetrievalDiagnosticsTests(unittest.TestCase):
    def test_generates_metrics_figures_tables_and_neighbor_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: list[str] = []
            for index in range(4):
                path = root / f"{index}.png"
                Image.new("RGB", (16, 16), (index * 50, 20, 30)).save(path)
                paths.append(str(path))
            embeddings = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
            )
            labels = [0, 0, 1, 1]
            codec = LabelCodec(
                genus_to_id={"A": 0, "B": 1},
                species_to_id={"A_a": 0, "B_b": 1},
            )
            metrics = run_retrieval_diagnostics(
                epoch=0,
                query_genus_embeddings=embeddings,
                query_species_embeddings=embeddings,
                query_genus_labels=labels,
                query_species_labels=labels,
                query_ids=["q0", "q1", "q2", "q3"],
                query_paths=paths,
                reference_genus_embeddings=embeddings,
                reference_species_embeddings=embeddings,
                reference_genus_labels=labels,
                reference_species_labels=labels,
                reference_ids=["r0", "r1", "r2", "r3"],
                reference_paths=paths,
                genus_predictions=labels,
                species_predictions=labels,
                label_codec=codec,
                logger=NullLogger(),
                output_dir=root / "output",
                config={
                    "recall_at_k": [1, 2],
                    "embedding_method": "tsne",
                    "max_embedding_points": 4,
                    "nearest_neighbor_queries": 2,
                    "nearest_neighbor_examples": 2,
                    "nearest_neighbors_k": 2,
                },
            )
            epoch_dir = root / "output" / "diagnostics" / "epoch_0001"
            self.assertEqual(metrics["top1_accuracy_species_global"], 1.0)
            self.assertIn("open_set_proxy_similarity", metrics)
            self.assertTrue((epoch_dir / "confusion_genus.png").is_file())
            self.assertTrue((epoch_dir / "confusion_species.png").is_file())
            self.assertTrue((epoch_dir / "embedding_projection.png").is_file())
            self.assertTrue((epoch_dir / "nearest_neighbor_examples.png").is_file())
            self.assertTrue((epoch_dir / "nearest_neighbors.csv").is_file())
            self.assertTrue((epoch_dir / "per_class_species.csv").is_file())
            self.assertTrue((epoch_dir / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
