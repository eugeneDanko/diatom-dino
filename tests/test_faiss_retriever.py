from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.faiss_retriever import FAISSRetriever


class FAISSRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = FAISSRetriever(use_faiss=False)
        self.index.build_index(
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
            item_ids=["a", "b", "c"],
            genera=["G1", "G1", "G2"],
            species=["G1_a", "G1_b", "G2_c"],
        )

    def test_exact_search_and_filter(self) -> None:
        result = self.index.search(np.asarray([1.0, 0.0]), top_k=2)[0]
        self.assertEqual(result[0].item_id, "a")
        filtered = self.index.search(np.asarray([1.0, 0.0]), top_k=2, genus_filter="G2")[0]
        self.assertEqual([item.item_id for item in filtered], ["c"])

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.npz"
            self.index.save_index(path)
            loaded = FAISSRetriever(use_faiss=False).load_index(path)
            self.assertEqual(loaded.search(np.asarray([1.0, 0.0]), top_k=1)[0][0].item_id, "a")


if __name__ == "__main__":
    unittest.main()

