from __future__ import annotations

import argparse

from core.clearml_logger import create_experiment_logger
from core.config_loader import load_config
from detector.yolo_model import YOLOModel
from detector.yolo_tester import YOLOTester


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the YOLO detector")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.overrides)
    model = YOLOModel(
        args.weights,
        device=str(config["model"].get("device", "0")),
        diatom_class_id=int(config["model"].get("diatom_class_id", 0)),
    ).load()
    evaluation = config.get("evaluation", {})
    tester = YOLOTester(
        model=model,
        data_yaml=config["paths"]["data_yaml"],
        output_dir=f"{config['paths']['output_dir']}/test",
        logger=create_experiment_logger(config, "test-detector"),
        split=evaluation.get("split", "test"),
        image_size=int(evaluation.get("imgsz", 1024)),
        confidence=float(evaluation.get("confidence", 0.001)),
        iou=float(evaluation.get("iou", 0.7)),
    )
    tester.test()


if __name__ == "__main__":
    main()

