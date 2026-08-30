from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inference.report_generator import ReportGenerator


class ReportGeneratorTest(unittest.TestCase):
    def test_html_and_json_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = ReportGenerator(directory).generate(
                {
                    "image.png": [
                        {
                            "detection_confidence": 0.9,
                            "decision": {
                                "status": "known_species",
                                "genus": "Navicula",
                                "species": "Navicula_a",
                                "genus_similarity": 0.91,
                                "species_similarity": 0.88,
                            },
                            "genus_candidates": [],
                            "species_candidates": [],
                            "crop_path": None,
                        }
                    ]
                }
            )
            self.assertTrue(report.exists())
            self.assertTrue((Path(directory) / "report.json").exists())
            self.assertIn("Navicula_a", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

