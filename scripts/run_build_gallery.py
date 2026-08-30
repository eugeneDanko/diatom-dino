from __future__ import annotations

import argparse

from core.config_loader import load_config
from inference.gallery_builder import build_gallery_indices
from scripts.common import build_classifier_components, build_eval_loader, build_label_codec


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persistent gallery indexes")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.overrides)
    checkpoint = args.checkpoint or config["paths"]["checkpoint"]
    output = args.output or config["paths"]["gallery_dir"]
    backbone, head, codec = build_classifier_components(config, checkpoint_path=checkpoint)
    if bool(config.get("evaluation", {}).get("allow_unseen_gallery_labels", False)):
        # Projection embeddings are label-agnostic. A benchmark gallery may
        # therefore use taxa absent from the optimization checkpoint codec.
        codec = build_label_codec(config)
    split = config.get("evaluation", {}).get("gallery_split", "gallery")
    gallery_loader, _ = build_eval_loader(config, split, codec)
    build_gallery_indices(
        backbone=backbone,
        projection_head=head,
        gallery_loader=gallery_loader,
        output_dir=output,
        checkpoint_path=checkpoint,
    )


if __name__ == "__main__":
    main()
