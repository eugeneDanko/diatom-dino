from __future__ import annotations

import argparse
from pathlib import Path

from core.clearml_logger import create_experiment_logger
from core.config_loader import load_config
from inference.super_tester import SuperTester, load_ground_truth
from scripts.common import build_pipeline, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent end-to-end evaluation")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.overrides)
    set_seed(int(config.get("seed", 42)))
    output_dir = Path(str(config["paths"]["output_dir"]))
    if output_dir.exists():
        raise FileExistsError(f"Frozen E2E test was already evaluated: {output_dir}")
    logger = create_experiment_logger(config, "end-to-end-test")
    pipeline = build_pipeline(config)
    paths = config["paths"]
    evaluation = config.get("evaluation", {})
    ground_truth = load_ground_truth(
        table_path=paths["ground_truth"],
        images_dir=paths["input"],
        labels_dir=paths.get("labels_dir"),
        metadata_path=paths.get("metadata"),
        image_extension=evaluation.get("image_extension", ".png"),
    )
    tester = SuperTester(
        pipeline=pipeline,
        ground_truth=ground_truth,
        output_dir=paths["output_dir"],
        logger=logger,
        match_iou=float(evaluation.get("match_iou", 0.5)),
    )
    tester.test()


if __name__ == "__main__":
    main()
