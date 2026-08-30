from __future__ import annotations

import argparse

from classifier.dino_tester import DINOTester
from core.clearml_logger import create_experiment_logger
from core.config_loader import load_config
from scripts.common import build_classifier_components, build_eval_loader, build_label_codec


def main() -> None:
    parser = argparse.ArgumentParser(description="Test gallery-based DINO retrieval")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.overrides)
    checkpoint = args.checkpoint or config["paths"]["checkpoint"]
    backbone, head, codec = build_classifier_components(config, checkpoint_path=checkpoint)
    evaluation = config.get("evaluation", {})
    if bool(evaluation.get("allow_unseen_gallery_labels", False)):
        codec = build_label_codec(config)
    retrieval = dict(evaluation.get("retrieval", {}))
    gallery_loader, _ = build_eval_loader(config, evaluation.get("gallery_split", "gallery"), codec)
    test_loader, _ = build_eval_loader(config, evaluation.get("test_split", "test"), codec)
    tester = DINOTester(
        backbone=backbone,
        projection_head=head,
        gallery_loader=gallery_loader,
        test_loader=test_loader,
        label_codec=codec,
        output_dir=f"{config['paths']['output_dir']}/test",
        logger=create_experiment_logger(config, "test-classifier"),
        k=int(evaluation.get("k", 1)),
        retrieval_strategies=retrieval.get("strategies"),
        primary_strategy=retrieval.get("primary"),
    )
    tester.test()


if __name__ == "__main__":
    main()
