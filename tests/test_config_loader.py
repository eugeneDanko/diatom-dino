from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.config_loader import ConfigError, load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_json_load_and_nested_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"model": {"device": "cpu"}, "epochs": 2}), encoding="utf-8")
            config = load_config(path, overrides=["model.device=cuda", "epochs=5"])
            self.assertEqual(config["model"]["device"], "cuda")
            self.assertEqual(config["epochs"], 5)

    def test_missing_required_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, required_sections=["model"])


if __name__ == "__main__":
    unittest.main()

