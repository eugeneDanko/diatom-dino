from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from data_pipeline.builder import build_dataset
from data_pipeline.splitter import build_splits
from inference.decision_logic import DecisionLogic, DecisionThresholds
from inference.faiss_retriever import Candidate


def _png(color: int = 128) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), (color, color, color)).save(stream, format="PNG")
    return stream.getvalue()


def _xml(filename: str) -> str:
    return f"""<annotation>
<filename>{filename}</filename><size><width>16</width><height>16</height></size>
<objects><object><name>Navicula cryptocephala</name>
<bbox><xmin>2</xmin><xmax>14</xmax><ymin>2</ymin><ymax>14</ymax></bbox>
</object></objects></annotation>"""


class PublicDataPipelineTests(unittest.TestCase):
    def _archives(self, root: Path) -> dict[str, Path]:
        archives: dict[str, Path] = {}
        gunduz = root / "gunduz.zip"
        with zipfile.ZipFile(gunduz, "w") as archive:
            for index in range(20):
                name = f"image_{index:03d}.png"
                archive.writestr(f"dataset/images/{name}", _png(30 + index))
                archive.writestr(f"dataset/xmls/image_{index:03d}.xml", _xml(name))
        archives["gunduz"] = gunduz

        members = {
            "ude": lambda i: f"root/UDE Diatoms in the Wild 2024-Subset_n25/Navicula cryptocephala/img {i}.png",
            "diatom1042": lambda i: f"Navicula cryptocephala {i}.png",
            "siyue_pu": lambda i: f"folder/Navicula cryptocephala {i}.png",
        }
        for source, member in members.items():
            path = root / f"{source}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for index in range(3):
                    archive.writestr(member(index), _png(80 + index))
            archives[source] = path
        return archives

    def test_build_and_split_exclude_nii_and_keep_gunduz_benchmark_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = build_dataset(root / "data", self._archives(root))
            self.assertFalse(result["nii_included"])
            dataset = root / "data" / "datasetDiatom"
            crops = pd.read_csv(dataset / "manifests/crops.csv")
            self.assertEqual(set(crops.source), {"gunduz", "ude", "diatom1042", "siyue_pu"})
            split = build_splits(root / "data")
            self.assertTrue(split["passed"])
            audit = json.loads((root / "data/splits/audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["classifier_gunduz_rows"], 0)
            self.assertEqual(audit["classifier_nii_rows"], 0)
            train = pd.read_csv(root / "data/splits/classifier/train.csv")
            self.assertEqual(set(train.source), {"ude", "diatom1042", "siyue_pu"})
            gallery = pd.read_csv(root / "data/splits/benchmark/gunduz/gallery.csv")
            query = pd.read_csv(root / "data/splits/benchmark/gunduz/query.csv")
            self.assertEqual(set(gallery.source) | set(query.source), {"gunduz"})
            self.assertFalse(set(gallery.image_id) & set(query.image_id))
            self.assertEqual(set(query.protocol_target), {"known"})
            self.assertTrue((root / "data/splits/detector/data.yaml").is_file())
            self.assertTrue((root / "data/splits/benchmark/gunduz/e2e_taxonomy.csv").is_file())

    def test_closed_set_decision_does_not_reject_low_similarity(self) -> None:
        logic = DecisionLogic(DecisionThresholds(), top_k=2, open_set=False)
        candidates = [
            Candidate("a", -0.20, "Navicula", "Navicula_cryptocephala"),
            Candidate("b", -0.21, "Cymbella", "Cymbella_excisa"),
        ]
        genus = logic.decide_genus(candidates)
        self.assertTrue(genus.accepted)
        self.assertEqual(genus.label, "Navicula")


if __name__ == "__main__":
    unittest.main()
