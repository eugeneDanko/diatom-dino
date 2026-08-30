from __future__ import annotations

import argparse

import torch

from classifier.dino_trainer import DINOTrainer, load_classifier_checkpoint
from core.clearml_logger import create_experiment_logger
from core.config_loader import load_config
from scripts.common import build_classifier_components, build_eval_loader, build_train_loaders, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINOv2 hierarchical projection heads")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--additional-epochs", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    if args.epochs is not None and args.additional_epochs is not None:
        parser.error("--epochs and --additional-epochs are mutually exclusive")
    if args.additional_epochs is not None and not args.resume:
        parser.error("--additional-epochs requires --resume")
    if args.additional_epochs is not None and args.additional_epochs < 1:
        parser.error("--additional-epochs must be positive")
    config = load_config(args.config, overrides=args.overrides, required_sections=["dataset", "model"])
    set_seed(int(config.get("seed", 42)))
    logger = create_experiment_logger(config, "train-classifier")
    backbone, head, codec = build_classifier_components(config)
    train_loader, val_loader, _, _ = build_train_loaders(config, codec)
    reference_split = config.get("evaluation", {}).get("validation_reference_split", "gallery")
    reference_loader, _ = build_eval_loader(config, reference_split, codec)
    trainer = DINOTrainer(
        backbone=backbone,
        projection_head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        reference_loader=reference_loader,
        config=config,
        output_dir=config["paths"]["output_dir"],
        label_codec=codec,
        logger=logger,
    )
    if args.resume:
        training_config = config.get("training", {})
        reset_optimizer = bool(training_config.get("reset_optimizer_on_resume", False))
        checkpoint, restored_codec = load_classifier_checkpoint(
            args.resume,
            backbone=backbone,
            projection_head=head,
            optimizer=None if reset_optimizer else trainer.optimizer,
        )
        codec_changed = restored_codec.to_dict() != codec.to_dict()
        allow_codec_remap = bool(
            config.get("training", {}).get("allow_codec_remap_on_resume", False)
        )
        if codec_changed and not allow_codec_remap:
            raise ValueError(
                "Resume checkpoint label codec differs from dataset.codec_table_path. "
                "Set training.allow_codec_remap_on_resume=true only for a "
                "label-agnostic metric-learning projection head."
            )
        trainer.label_codec = codec if codec_changed else restored_codec
        trainer.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        trainer.initialize_distillation_teacher()
        if bool(training_config.get("reset_learning_rate_on_resume", False)):
            learning_rates = [float(training_config.get("learning_rate", 3e-4))]
            if len(trainer.optimizer.param_groups) > 1:
                learning_rates.append(float(training_config.get("backbone_learning_rate", 1e-5)))
            for group, learning_rate in zip(trainer.optimizer.param_groups, learning_rates):
                group["lr"] = learning_rate
    if args.additional_epochs is not None:
        epochs = trainer.start_epoch + args.additional_epochs
    else:
        epochs = args.epochs or int(config.get("training", {}).get("epochs", 60))
    if config.get("training", {}).get("scheduler", "cosine") == "cosine":
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=max(1, epochs - trainer.start_epoch),
        )
    trainer.fit(epochs)


if __name__ == "__main__":
    main()
