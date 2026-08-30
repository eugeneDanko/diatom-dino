from __future__ import annotations

import argparse
from pathlib import Path

from core.config_loader import load_config
from inference.report_generator import ReportGenerator
from scripts.common import build_pipeline


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiatomDINO inference")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.overrides)
    if args.output:
        config["paths"]["output_dir"] = args.output
    pipeline = build_pipeline(config)
    input_path = Path(args.input or config["paths"]["input"])
    image_paths = collect_inputs(input_path)
    predictions = pipeline.predict_batch(image_paths)
    serialized = {
        image_path: [prediction.to_dict() for prediction in values]
        for image_path, values in predictions.items()
    }
    ReportGenerator(config["paths"]["output_dir"]).generate(serialized, name="inference_report")


if __name__ == "__main__":
    main()

