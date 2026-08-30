"""Shared construction helpers for thin CLI entry points."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from classifier.crop_dataset import DiatomCropDataset, LabelCodec
    from classifier.dino_backbone import DINOBackbone
    from classifier.projection_head import HierarchicalProjectionHead
    from inference.decision_logic import DecisionThresholds
    from inference.super_pipeline import DiatomDINOPipeline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except (ImportError, OSError):
        pass


def build_label_codec(config: Mapping[str, Any]) -> LabelCodec:
    from classifier.crop_dataset import LabelCodec, read_table

    dataset = config["dataset"]
    table = read_table(dataset.get("codec_table_path", dataset["table_path"]))
    return LabelCodec.from_table(
        table,
        genus_column=dataset.get("genus_column", "genus"),
        species_column=dataset.get("species_column", "species"),
    )


def build_crop_dataset(
    config: Mapping[str, Any],
    split: str,
    *,
    training: bool,
    label_codec: LabelCodec,
) -> DiatomCropDataset:
    from classifier.crop_dataset import DiatomCropDataset
    from classifier.transforms import build_transform

    dataset = config["dataset"]
    return DiatomCropDataset(
        root=dataset["root"],
        table_path=dataset["table_path"],
        split=split,
        transform=build_transform(config.get("transforms", {}), training=training),
        label_codec=label_codec,
        id_column=dataset.get("id_column", "id_crop"),
        genus_column=dataset.get("genus_column", "genus"),
        species_column=dataset.get("species_column", "species"),
        split_column=dataset.get("split_column", "split"),
        path_column=dataset.get("path_column", "path"),
        filename_column=dataset.get("filename_column", "filename"),
        class_dir_template=dataset.get("class_dir_template", "{species_key}"),
        default_extension=dataset.get("default_extension", ".png"),
        power_groups=(
            dataset.get("train_power_groups")
            if training
            else dataset.get("evaluation_power_groups")
        ),
    )


def build_train_loaders(
    config: Mapping[str, Any], label_codec: LabelCodec
) -> tuple[Any, Any, DiatomCropDataset, DiatomCropDataset]:
    from classifier.batch_sampler import HierarchicalBatchSampler, ReplayHierarchicalBatchSampler
    from classifier.crop_dataset import build_dataloader

    train_dataset = build_crop_dataset(config, "train", training=True, label_codec=label_codec)
    val_dataset = build_crop_dataset(config, "val", training=False, label_codec=label_codec)
    sampler_config = config.get("sampler", {})
    train_species = set(train_dataset.species_labels)
    hard_groups: list[list[int]] = []
    strict_hard_groups = bool(sampler_config.get("strict_hard_negative_groups", False))
    for raw_group in sampler_config.get("hard_negative_groups", []):
        names = raw_group.get("species", []) if isinstance(raw_group, Mapping) else raw_group
        unknown = [str(name) for name in names if str(name) not in label_codec.species_to_id]
        if unknown:
            raise ValueError(f"Unknown hard-negative species in canonical codec: {unknown}")
        identifiers = [label_codec.species_to_id[str(name)] for name in names]
        missing_from_train = [name for name, identifier in zip(names, identifiers) if identifier not in train_species]
        if missing_from_train and strict_hard_groups:
            raise ValueError(f"Hard-negative species absent from current train profile: {missing_from_train}")
        available = [identifier for identifier in identifiers if identifier in train_species]
        if len(set(available)) >= 2:
            hard_groups.append(available)
    if not hard_groups and bool(sampler_config.get("auto_hard_negative_groups", True)):
        table = train_dataset.table
        if "is_hard_negative" in table.columns:
            marked = table[
                pd.to_numeric(table["is_hard_negative"], errors="coerce").fillna(0).eq(1)
            ]
            for genus, rows in marked.groupby(train_dataset.genus_column, sort=True):
                names = sorted(
                    {
                        label_codec.species_key(str(genus), str(species))
                        for species in rows[train_dataset.species_column]
                    }
                )
                available = [
                    label_codec.species_to_id[name]
                    for name in names
                    if name in label_codec.species_to_id
                    and label_codec.species_to_id[name] in train_species
                ]
                # Add seen target siblings of the same genus as well: the
                # desired conflict is between species, not only among rows
                # explicitly marked hard_negative.
                genus_id = label_codec.genus_to_id.get(str(genus))
                if genus_id is not None:
                    available.extend(
                        species
                        for species, item_genus in zip(
                            train_dataset.species_labels, train_dataset.genus_labels
                        )
                        if item_genus == genus_id
                    )
                unique = list(dict.fromkeys(available))
                if len(unique) >= 2:
                    hard_groups.append(unique)
    hard_probability = float(sampler_config.get("hard_negative_probability", 0.0))
    if hard_probability > 0 and not hard_groups:
        if bool(sampler_config.get("require_hard_negative_groups", False)):
            raise ValueError(
                "Hard-negative sampling was required, but the train manifest produced no valid same-genus groups"
            )
        hard_probability = 0.0
    sampler_arguments = dict(
        genera_per_batch=int(sampler_config.get("genera_per_batch", 4)),
        species_per_genus=int(sampler_config.get("species_per_genus", 2)),
        samples_per_species=int(sampler_config.get("samples_per_species", 2)),
        batches_per_epoch=sampler_config.get("batches_per_epoch"),
        seed=int(config.get("seed", 42)),
        class_sampling=str(sampler_config.get("class_sampling", "uniform")),
        sampling_power=float(sampler_config.get("sampling_power", 0.5)),
    )
    replay_fraction = float(sampler_config.get("replay_fraction", 0.0))
    if replay_fraction > 0:
        if "training_role" not in train_dataset.table.columns:
            raise ValueError("sampler.replay_fraction requires training_role in train manifest")
        sampler = ReplayHierarchicalBatchSampler(
            train_dataset.genus_labels, train_dataset.species_labels,
            train_dataset.table["training_role"].astype(str).tolist(),
            replay_fraction=replay_fraction, **sampler_arguments,
        )
    else:
        sampler = HierarchicalBatchSampler(
            train_dataset.genus_labels, train_dataset.species_labels,
            hard_negative_groups=hard_groups, hard_negative_probability=hard_probability,
            **sampler_arguments,
        )
    loader_config = config.get("loader", {})
    train_loader = build_dataloader(
        train_dataset,
        batch_size=sampler.batch_size,
        batch_sampler=sampler,
        num_workers=int(loader_config.get("num_workers", 4)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(loader_config.get("eval_batch_size", 64)),
        num_workers=int(loader_config.get("num_workers", 4)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def build_classifier_components(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> tuple[DINOBackbone, HierarchicalProjectionHead, LabelCodec]:
    from classifier.dino_backbone import DINOBackbone
    from classifier.dino_trainer import load_classifier_checkpoint
    from classifier.projection_head import HierarchicalProjectionHead

    model_config = config.get("model", {})
    backbone = DINOBackbone(
        model_name=model_config.get("name", "dinov2_vitb14_reg"),
        repository=model_config.get("repository", "facebookresearch/dinov2"),
        device=model_config.get("device", "cuda"),
        unfreeze_layers=int(model_config.get("unfreeze_layers", 0)),
        pretrained=bool(model_config.get("pretrained", True)),
    ).load()
    projection = config.get("projection_head", {})
    head = HierarchicalProjectionHead(
        int(backbone.embed_dim),
        hidden_dim=int(projection.get("hidden_dim", 512)),
        projection_dim=int(projection.get("projection_dim", 256)),
        dropout=float(projection.get("dropout", 0.1)),
    )
    codec = build_label_codec(config)
    if checkpoint_path:
        _, codec = load_classifier_checkpoint(
            checkpoint_path,
            backbone=backbone,
            projection_head=head,
        )
    return backbone, head, codec


def build_eval_loader(
    config: Mapping[str, Any], split: str, label_codec: LabelCodec
) -> tuple[Any, DiatomCropDataset]:
    from classifier.crop_dataset import build_dataloader

    dataset = build_crop_dataset(config, split, training=False, label_codec=label_codec)
    loader_config = config.get("loader", {})
    loader = build_dataloader(
        dataset,
        batch_size=int(loader_config.get("eval_batch_size", 64)),
        num_workers=int(loader_config.get("num_workers", 4)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
    )
    return loader, dataset


def load_thresholds(config: Mapping[str, Any]) -> DecisionThresholds:
    from inference.decision_logic import DecisionThresholds

    inference = config.get("inference", {})
    threshold_path = inference.get("thresholds_path")
    if threshold_path and Path(threshold_path).exists():
        value = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
        return DecisionThresholds.from_dict(value.get("thresholds", value))
    return DecisionThresholds.from_dict(dict(config.get("thresholds", {})))


def build_pipeline(config: Mapping[str, Any]) -> DiatomDINOPipeline:
    from classifier.transforms import build_transform
    from detector.yolo_model import YOLOModel
    from inference.decision_logic import DecisionLogic
    from inference.faiss_retriever import FAISSRetriever
    from inference.super_pipeline import DiatomDINOPipeline

    paths = config["paths"]
    detector_config = config.get("detector", {})
    detector = YOLOModel(
        paths["detector_weights"],
        device=str(detector_config.get("device", "cuda")),
        diatom_class_id=int(detector_config.get("diatom_class_id", 0)),
    ).load()
    backbone, head, _ = build_classifier_components(
        config,
        checkpoint_path=paths["classifier_checkpoint"],
    )
    genus_index = FAISSRetriever().load_index(paths["genus_index"])
    species_index = FAISSRetriever().load_index(paths["species_index"])
    inference_config = config.get("inference", {})
    detector_confidence = float(detector_config.get("confidence", 0.25))
    detector_threshold_path = paths.get("detector_thresholds")
    if detector_threshold_path:
        threshold_file = Path(str(detector_threshold_path))
        if not threshold_file.is_file():
            raise FileNotFoundError(f"Detector calibration threshold does not exist: {threshold_file}")
        detector_selection = json.loads(threshold_file.read_text(encoding="utf-8"))
        if "confidence" not in detector_selection:
            raise ValueError("Detector calibration JSON has no confidence")
        detector_confidence = float(detector_selection["confidence"])
    artifact_gate = None
    if paths.get("resnet_checkpoint") or paths.get("resnet_thresholds"):
        if not paths.get("resnet_checkpoint") or not paths.get("resnet_thresholds"):
            raise ValueError("Configure both paths.resnet_checkpoint and paths.resnet_thresholds")
        from inference.resnet_gate import ResNetArtifactGate

        gate_config = config.get("resnet_gate", {})
        model_device = str(config.get("model", {}).get("device", "cuda"))
        artifact_gate = ResNetArtifactGate(
            checkpoint=paths["resnet_checkpoint"],
            threshold_path=paths["resnet_thresholds"],
            device=str(gate_config.get("device", model_device)),
            transform_config=dict(gate_config.get("transforms", {"image_size": 224})),
        )
    return DiatomDINOPipeline(
        detector=detector,
        backbone=backbone,
        projection_head=head,
        genus_index=genus_index,
        species_index=species_index,
        decision_logic=DecisionLogic(
            load_thresholds(config),
            top_k=int(inference_config.get("top_k", 5)),
            open_set=bool(inference_config.get("open_set", True)),
        ),
        transform=build_transform(config.get("transforms", {}), training=False),
        artifact_gate=artifact_gate,
        detector_confidence=detector_confidence,
        detector_iou=float(detector_config.get("iou", 0.7)),
        detector_image_size=int(detector_config.get("image_size", 1024)),
        top_k=int(inference_config.get("top_k", 5)),
        crop_padding=float(inference_config.get("crop_padding", 0.05)),
        save_crops=bool(inference_config.get("save_crops", False)),
        output_dir=paths.get("output_dir"),
    )
