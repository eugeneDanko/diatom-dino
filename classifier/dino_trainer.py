"""Training loop for DINOv2 plus hierarchical projection heads."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch
import pandas as pd

from core.base_trainer import BaseTrainer
from core.clearml_logger import ExperimentLogger
from .contrastive_loss import HierarchicalContrastiveLoss
from .crop_dataset import LabelCodec
from .diagnostics import run_retrieval_diagnostics
from .dino_backbone import DINOBackbone
from .metrics import (
    exact_knn_predict,
    hierarchical_knn_predict,
    hierarchical_retrieval_predict,
    retrieval_metrics,
    retrieval_predict,
)
from .projection_head import HierarchicalProjectionHead


class DINOTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        backbone: DINOBackbone,
        projection_head: HierarchicalProjectionHead,
        train_loader: Any,
        val_loader: Any,
        reference_loader: Any | None = None,
        config: Mapping[str, Any],
        output_dir: str | Path,
        label_codec: LabelCodec,
        logger: ExperimentLogger | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> None:
        monitor = str(config.get("training", {}).get("monitor", "val/f1_macro_species"))
        super().__init__(
            config=config,
            output_dir=output_dir,
            logger=logger,
            monitor=monitor,
            mode="max",
        )
        self.backbone = backbone
        self.projection_head = projection_head.to(backbone.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.reference_loader = reference_loader
        self.label_codec = label_codec
        adaptation = dict(config.get("adaptation", {}))
        self.freeze_genus_head = bool(adaptation.get("freeze_genus_head", False))
        if self.freeze_genus_head:
            for parameter in self.projection_head.genus_head.parameters():
                parameter.requires_grad_(False)
        self.distillation_weight = float(adaptation.get("distillation_weight", 0.0))
        self.distillation_genus_weight = float(adaptation.get("distillation_genus_weight", 1.0))
        self.distillation_species_weight = float(adaptation.get("distillation_species_weight", 1.0))
        self.distillation_replay_only = bool(adaptation.get("distillation_replay_only", True))
        if self.distillation_weight < 0:
            raise ValueError("adaptation.distillation_weight must be non-negative")
        self.teacher_projection_head: HierarchicalProjectionHead | None = None
        loss_config = config.get("loss", {})
        self.criterion = HierarchicalContrastiveLoss(
            temperature=float(loss_config.get("temperature", 0.1)),
            genus_weight=float(loss_config.get("genus_weight", 1.0)),
            species_weight=float(loss_config.get("species_weight", 1.0)),
            hierarchy_weight=float(loss_config.get("hierarchy_weight", 0.0)),
            consistency_weight=float(loss_config.get("consistency_weight", 0.0)),
            species_margin=float(loss_config.get("species_margin", 0.1)),
            genus_margin=float(loss_config.get("genus_margin", 0.1)),
        )
        training = config.get("training", {})
        if optimizer is None:
            parameter_groups: list[dict[str, Any]] = [
                {
                    "params": [
                        parameter for parameter in self.projection_head.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": float(training.get("learning_rate", 3e-4)),
                }
            ]
            backbone_parameters = list(self.backbone.trainable_parameters())
            if backbone_parameters:
                parameter_groups.append(
                    {
                        "params": backbone_parameters,
                        "lr": float(training.get("backbone_learning_rate", 1e-5)),
                    }
                )
            optimizer = torch.optim.AdamW(
                parameter_groups,
                weight_decay=float(training.get("weight_decay", 1e-4)),
            )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = float(training.get("gradient_clip", 1.0))
        self.knn_k = int(training.get("validation_knn_k", 1))
        retrieval = dict(config.get("evaluation", {}).get("retrieval", {}))
        strategies = [dict(value) for value in retrieval.get("strategies", [])]
        primary = str(retrieval.get("primary", "")).strip()
        self.validation_retrieval: dict[str, Any] = {
            "name": f"nearest_k{self.knn_k}", "method": "nearest", "k": self.knn_k
        }
        if strategies:
            selected = next(
                (value for value in strategies if str(value.get("name", "")) == primary),
                strategies[0],
            )
            self.validation_retrieval = selected
        self.use_amp = bool(training.get("mixed_precision", True)) and backbone.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self._log_stage_summary()

    def initialize_distillation_teacher(self) -> None:
        """Freeze a Stage-2 snapshot after resume, without duplicating DINO."""
        if self.distillation_weight <= 0:
            return
        if any(True for _ in self.backbone.trainable_parameters()):
            raise ValueError(
                "Distillation currently requires a frozen backbone; set model.unfreeze_layers=0"
            )
        teacher = deepcopy(self.projection_head).to(self.backbone.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher_projection_head = teacher

    def _log_stage_summary(self) -> None:
        reporting = dict(self.config.get("reporting", {}))
        if not reporting.get("stage"):
            return
        train_table = self.train_loader.dataset.table
        val_table = self.val_loader.dataset.table
        species_column = self.config.get("dataset", {}).get("species_column", "species")
        genus_column = self.config.get("dataset", {}).get("genus_column", "genus")
        summary = pd.DataFrame(
            [
                {
                    "stage": reporting.get("stage"),
                    "objective": reporting.get("objective"),
                    "hypothesis": reporting.get("hypothesis"),
                    "train_crops": len(train_table),
                    "validation_crops": len(val_table),
                    "train_genera": train_table[genus_column].nunique(),
                    "validation_genera": val_table[genus_column].nunique(),
                    "train_species": train_table[species_column].nunique(),
                    "validation_species": val_table[species_column].nunique(),
                    "manifest": self.config.get("dataset", {}).get("table_path"),
                    "checkpoint_rule": str(self.monitor_components or self.monitor),
                }
            ]
        )
        self.logger.log_table("Stage summary", summary, 0, series=str(reporting.get("stage")))
        checks = pd.DataFrame(
            [
                {"check": check, "status": "pending post-checkpoint evaluation"}
                for check in reporting.get("hypothesis_checks", [])
            ]
        )
        if not checks.empty:
            self.logger.log_table("Hypothesis checks", checks, 0, series=str(reporting.get("stage")))

    def _forward_batch(
        self, batch: Mapping[str, Any], *, compute_distillation: bool = False
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        image_value = batch["image"]
        genus_labels = torch.as_tensor(batch["genus_label"], device=self.backbone.device, dtype=torch.long)
        species_labels = torch.as_tensor(batch["species_label"], device=self.backbone.device, dtype=torch.long)
        num_views = 1
        if isinstance(image_value, (list, tuple)):
            if not image_value:
                raise ValueError("Augmented batch contains no views")
            num_views = len(image_value)
            images = torch.cat(
                [view.to(self.backbone.device, non_blocking=True) for view in image_value], dim=0
            )
            genus_labels = genus_labels.repeat(len(image_value))
            species_labels = species_labels.repeat(len(image_value))
        else:
            images = image_value.to(self.backbone.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            raw_features = self.backbone.extract_features(images)
            genus_embeddings, species_embeddings = self.projection_head(raw_features)
            losses = self.criterion(
                genus_embeddings,
                species_embeddings,
                genus_labels,
                species_labels,
                num_views=num_views,
            )
            distillation = genus_embeddings.sum() * 0.0
            if self.distillation_weight > 0 and compute_distillation:
                if self.teacher_projection_head is None:
                    raise RuntimeError("Distillation teacher was not initialized from resume checkpoint")
                with torch.no_grad():
                    teacher_genus, teacher_species = self.teacher_projection_head(raw_features.detach())
                mask = torch.ones(len(genus_embeddings), dtype=torch.bool, device=self.backbone.device)
                if self.distillation_replay_only:
                    roles = batch.get("training_role")
                    if roles is None:
                        raise ValueError(
                            "Replay-only distillation requires training_role in the training manifest"
                        )
                    role_mask = torch.as_tensor(
                        [str(value) == "replay" for value in roles],
                        dtype=torch.bool,
                        device=self.backbone.device,
                    )
                    mask = role_mask.repeat(num_views)
                if mask.any():
                    genus_distance = 1.0 - (genus_embeddings[mask] * teacher_genus[mask]).sum(dim=-1)
                    species_distance = 1.0 - (species_embeddings[mask] * teacher_species[mask]).sum(dim=-1)
                    distillation = (
                        self.distillation_genus_weight * genus_distance.mean()
                        + self.distillation_species_weight * species_distance.mean()
                    )
                losses["distillation"] = distillation
                losses["total"] = losses["total"] + self.distillation_weight * distillation
            else:
                losses["distillation"] = distillation
        return losses, genus_embeddings, species_embeddings

    def train_epoch(self, epoch: int) -> dict[str, float]:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self.projection_head.train()
        self.backbone.set_train_mode(True)
        totals = {"loss": 0.0, "loss_genus": 0.0, "loss_species": 0.0, "loss_hierarchy": 0.0, "loss_consistency": 0.0, "loss_distillation": 0.0}
        batches = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            losses, _, _ = self._forward_batch(batch, compute_distillation=True)
            self.scaler.scale(losses["total"]).backward()
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                parameters = list(self.projection_head.parameters()) + list(self.backbone.trainable_parameters())
                torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            totals["loss"] += float(losses["total"].detach())
            totals["loss_genus"] += float(losses["genus"].detach())
            totals["loss_species"] += float(losses["species"].detach())
            totals["loss_hierarchy"] += float(losses["hierarchy"].detach())
            totals["loss_consistency"] += float(losses["consistency"].detach())
            totals["loss_distillation"] += float(losses["distillation"].detach())
            batches += 1
        if self.scheduler is not None:
            self.scheduler.step()
        return {key: value / max(1, batches) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.projection_head.eval()
        self.backbone.set_train_mode(False)
        genus_embeddings: list[np.ndarray] = []
        species_embeddings: list[np.ndarray] = []
        genus_labels: list[int] = []
        species_labels: list[int] = []
        query_ids: list[str] = []
        query_paths: list[str] = []
        power_groups: list[str] = []
        totals = {"loss": 0.0, "loss_genus": 0.0, "loss_species": 0.0, "loss_hierarchy": 0.0, "loss_consistency": 0.0, "loss_distillation": 0.0}
        batches = 0
        for batch in self.val_loader:
            losses, genus_batch, species_batch = self._forward_batch(batch)
            genus_embeddings.append(genus_batch.cpu().numpy())
            species_embeddings.append(species_batch.cpu().numpy())
            genus_labels.extend(torch.as_tensor(batch["genus_label"]).tolist())
            species_labels.extend(torch.as_tensor(batch["species_label"]).tolist())
            query_ids.extend(str(value) for value in batch["crop_id"])
            query_paths.extend(str(value) for value in batch["path"])
            if "power_group" in batch:
                power_groups.extend(str(value) for value in batch["power_group"])
            totals["loss"] += float(losses["total"])
            totals["loss_genus"] += float(losses["genus"])
            totals["loss_species"] += float(losses["species"])
            totals["loss_hierarchy"] += float(losses["hierarchy"])
            totals["loss_consistency"] += float(losses["consistency"])
            totals["loss_distillation"] += float(losses["distillation"])
            batches += 1
        genus_array = np.concatenate(genus_embeddings, axis=0)
        species_array = np.concatenate(species_embeddings, axis=0)
        if self.reference_loader is not None:
            (
                reference_genus,
                reference_species,
                reference_genus_labels,
                reference_species_labels,
                reference_ids,
                reference_paths,
            ) = self._embed_reference(self.reference_loader)
            genus_predictions = retrieval_predict(
                genus_array, reference_genus, reference_genus_labels, self.validation_retrieval
            )
            global_species_predictions = retrieval_predict(
                species_array, reference_species, reference_species_labels, self.validation_retrieval
            )
            species_predictions = hierarchical_retrieval_predict(
                species_array,
                reference_species,
                genus_predictions,
                reference_genus_labels,
                reference_species_labels,
                self.validation_retrieval,
            )
        else:
            genus_predictions = exact_knn_predict(
                genus_array, genus_array, genus_labels, k=self.knn_k, exclude_self=True
            )
            global_species_predictions = exact_knn_predict(
                species_array, species_array, species_labels, k=self.knn_k, exclude_self=True
            )
            species_predictions = hierarchical_knn_predict(
                species_array,
                species_array,
                genus_predictions,
                genus_labels,
                species_labels,
                k=self.knn_k,
                self_reference_indices=list(range(len(species_array))),
            )
        metrics = {key: value / max(1, batches) for key, value in totals.items()}
        metrics.update(retrieval_metrics(genus_labels, genus_predictions, "genus"))
        metrics.update(retrieval_metrics(species_labels, species_predictions, "species"))
        metrics.update(retrieval_metrics(species_labels, global_species_predictions, "species_global"))
        # Top-K is a target metric in the V3 protocol and is therefore computed
        # every epoch, independently of the expensive diagnostics interval.
        if self.reference_loader is not None:
            from .diagnostics import _rank_neighbors, _topk_metrics

            _, genus_order = _rank_neighbors(genus_array, reference_genus)
            species_similarities, species_order = _rank_neighbors(species_array, reference_species)
            ks = self.config.get("diagnostics", {}).get("recall_at_k", [1, 3, 5])
            metrics.update(_topk_metrics(genus_order, reference_genus_labels, genus_labels, ks, "genus"))
            metrics.update(_topk_metrics(species_order, reference_species_labels, species_labels, ks, "species"))
            reference_genus_array = np.asarray(reference_genus_labels)
            query_genus_array = np.asarray(genus_labels)
            allowed = reference_genus_array[None, :] == query_genus_array[:, None]
            oracle_order = np.argsort(-np.where(allowed, species_similarities, -np.inf), axis=1)
            metrics.update(
                _topk_metrics(
                    oracle_order, reference_species_labels, species_labels, ks,
                    "species_oracle_genus",
                )
            )
        if power_groups:
            if len(power_groups) != len(species_labels):
                raise ValueError("Validation power_group metadata does not match query count")
            power_group_array = np.asarray(power_groups)
            genus_truth = np.asarray(genus_labels)
            species_truth = np.asarray(species_labels)
            for power_group in sorted(set(power_groups)):
                selected = power_group_array == power_group
                metrics.update(
                    {
                        f"{key}_{power_group}": value
                        for key, value in retrieval_metrics(
                            genus_truth[selected], genus_predictions[selected], "genus"
                        ).items()
                    }
                )
                # few_shot has no statistically reliable query split in V3 and
                # must not silently enter the primary species checkpoint score.
                if power_group != "few_shot":
                    metrics.update(
                        {
                            f"{key}_{power_group}": value
                            for key, value in retrieval_metrics(
                                species_truth[selected], species_predictions[selected], "species"
                            ).items()
                        }
                    )
        diagnostics = dict(self.config.get("diagnostics", {}))
        interval = max(1, int(diagnostics.get("interval_epochs", 5)))
        if self.reference_loader is not None and bool(diagnostics.get("enabled", True)) and (
            epoch == 0 or (epoch + 1) % interval == 0
        ):
            metrics.update(
                run_retrieval_diagnostics(
                    epoch=epoch,
                    query_genus_embeddings=genus_array,
                    query_species_embeddings=species_array,
                    query_genus_labels=genus_labels,
                    query_species_labels=species_labels,
                    query_ids=query_ids,
                    query_paths=query_paths,
                    reference_genus_embeddings=reference_genus,
                    reference_species_embeddings=reference_species,
                    reference_genus_labels=reference_genus_labels,
                    reference_species_labels=reference_species_labels,
                    reference_ids=reference_ids,
                    reference_paths=reference_paths,
                    genus_predictions=genus_predictions,
                    species_predictions=species_predictions,
                    label_codec=self.label_codec,
                    logger=self.logger,
                    output_dir=self.output_dir,
                    config=diagnostics,
                )
            )
        return metrics

    @torch.no_grad()
    def _embed_reference(
        self, loader: Any
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str], list[str]]:
        genus_embeddings: list[np.ndarray] = []
        species_embeddings: list[np.ndarray] = []
        genus_labels: list[int] = []
        species_labels: list[int] = []
        item_ids: list[str] = []
        paths: list[str] = []
        for batch in loader:
            images = batch["image"].to(self.backbone.device, non_blocking=True)
            raw_features = self.backbone.extract_features(images)
            genus_batch, species_batch = self.projection_head(raw_features)
            genus_embeddings.append(genus_batch.cpu().numpy())
            species_embeddings.append(species_batch.cpu().numpy())
            genus_labels.extend(torch.as_tensor(batch["genus_label"]).tolist())
            species_labels.extend(torch.as_tensor(batch["species_label"]).tolist())
            item_ids.extend(str(value) for value in batch["crop_id"])
            paths.extend(str(value) for value in batch["path"])
        if not genus_embeddings:
            raise ValueError("Reference gallery loader is empty")
        return (
            np.concatenate(genus_embeddings, axis=0),
            np.concatenate(species_embeddings, axis=0),
            genus_labels,
            species_labels,
            item_ids,
            paths,
        )

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: Mapping[str, float]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "config": self.config,
            "model_name": self.backbone.model_name,
            "backbone_embed_dim": self.backbone.embed_dim,
            "unfreeze_layers": self.backbone.unfreeze_layers,
            # A frozen pretrained backbone is reproducible from model_name and
            # should not inflate every checkpoint by hundreds of megabytes.
            "backbone": (
                self.backbone.model.state_dict()
                if self.backbone.model is not None and self.backbone.unfreeze_layers > 0
                else None
            ),
            "projection_head": self.projection_head.state_dict(),
            "projection_head_config": {
                "embed_dim": self.projection_head.embed_dim,
                "projection_dim": self.projection_head.projection_dim,
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "label_codec": self.label_codec.to_dict(),
        }
        torch.save(checkpoint, output)
        self.label_codec.save(self.output_dir / "labels.json")
        return output


def load_classifier_checkpoint(
    checkpoint_path: str | Path,
    *,
    backbone: DINOBackbone,
    projection_head: HierarchicalProjectionHead,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, Any], LabelCodec]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if backbone.model is None:
        backbone.load(checkpoint.get("model_name"))
    if checkpoint.get("backbone") is not None:
        backbone.model.load_state_dict(checkpoint["backbone"])
    projection_head.load_state_dict(checkpoint["projection_head"])
    projection_head.to(backbone.device)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint, LabelCodec.from_dict(checkpoint["label_codec"])
