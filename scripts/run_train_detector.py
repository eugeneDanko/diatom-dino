from __future__ import annotations

import argparse

from core.clearml_logger import create_experiment_logger
from core.config_loader import load_config
from detector.yolo_model import YOLOModel
from detector.yolo_trainer import YOLOTrainer
from scripts.common import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DiatomDINO YOLO detector")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--weights")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.overrides, required_sections=["model", "training"])
    set_seed(int(config.get("seed", 42)))
    logger = create_experiment_logger(config, "train-detector")
    model_config = config["model"]
    model = YOLOModel(
        args.weights or model_config["weights"],
        device=str(model_config.get("device", "0")),
        diatom_class_id=int(model_config.get("diatom_class_id", 0)),
    ).load()
    trainer = YOLOTrainer(
        model=model,
        config=config,
        output_dir=config["paths"]["output_dir"],
        logger=logger,
    )
    trainer.fit(args.epochs)


if __name__ == "__main__":
    main()
