"""Supervised and hierarchical contrastive objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Numerically stable single-view supervised contrastive loss."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, dimension]")
        labels = labels.reshape(-1)
        if len(labels) != len(features):
            raise ValueError("features and labels have different batch sizes")
        features = F.normalize(features, dim=-1)
        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        identity = torch.eye(len(features), dtype=torch.bool, device=features.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
        valid_anchor = positive_mask.any(dim=1)
        if not valid_anchor.any():
            # Keep a differentiable zero so the caller can safely backpropagate.
            return features.sum() * 0.0
        logits_mask = ~identity
        exp_logits = torch.exp(logits) * logits_mask
        log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_positive_log_prob = (
            (positive_mask * log_probability).sum(dim=1)
            / positive_mask.sum(dim=1).clamp_min(1)
        )
        return -mean_positive_log_prob[valid_anchor].mean()


class HierarchicalMarginLoss(nn.Module):
    """Enforce same-species < same-genus < other-genus distances."""

    def __init__(self, species_margin: float = 0.1, genus_margin: float = 0.1) -> None:
        super().__init__()
        self.species_margin = float(species_margin)
        self.genus_margin = float(genus_margin)

    def forward(
        self,
        embeddings: torch.Tensor,
        genus_labels: torch.Tensor,
        species_labels: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        distances = 1.0 - embeddings @ embeddings.T
        identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
        same_species = species_labels[:, None].eq(species_labels[None, :]) & ~identity
        same_genus_other_species = (
            genus_labels[:, None].eq(genus_labels[None, :])
            & ~species_labels[:, None].eq(species_labels[None, :])
        )
        other_genus = ~genus_labels[:, None].eq(genus_labels[None, :])

        losses: list[torch.Tensor] = []
        for index in range(len(embeddings)):
            if same_species[index].any() and same_genus_other_species[index].any():
                farthest_same_species = distances[index][same_species[index]].max()
                closest_same_genus = distances[index][same_genus_other_species[index]].min()
                losses.append(F.relu(farthest_same_species - closest_same_genus + self.species_margin))
            if same_genus_other_species[index].any() and other_genus[index].any():
                farthest_same_genus = distances[index][same_genus_other_species[index]].max()
                closest_other_genus = distances[index][other_genus[index]].min()
                losses.append(F.relu(farthest_same_genus - closest_other_genus + self.genus_margin))
        return torch.stack(losses).mean() if losses else embeddings.sum() * 0.0


class HierarchicalContrastiveLoss(nn.Module):
    def __init__(
        self,
        *,
        temperature: float = 0.1,
        genus_weight: float = 1.0,
        species_weight: float = 1.0,
        hierarchy_weight: float = 0.0,
        consistency_weight: float = 0.0,
        species_margin: float = 0.1,
        genus_margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.genus_weight = float(genus_weight)
        self.species_weight = float(species_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.consistency_weight = float(consistency_weight)
        self.supcon = SupervisedContrastiveLoss(temperature)
        self.hierarchy = HierarchicalMarginLoss(species_margin, genus_margin)

    def forward(
        self,
        genus_embeddings: torch.Tensor,
        species_embeddings: torch.Tensor,
        genus_labels: torch.Tensor,
        species_labels: torch.Tensor,
        *,
        num_views: int = 1,
    ) -> dict[str, torch.Tensor]:
        genus_loss = self.supcon(genus_embeddings, genus_labels)
        species_loss = self.supcon(species_embeddings, species_labels)
        hierarchy_loss = self.hierarchy(species_embeddings, genus_labels, species_labels)
        consistency_loss = self._view_consistency(species_embeddings, num_views)
        total = (
            self.genus_weight * genus_loss
            + self.species_weight * species_loss
            + self.hierarchy_weight * hierarchy_loss
            + self.consistency_weight * consistency_loss
        )
        return {
            "total": total,
            "genus": genus_loss,
            "species": species_loss,
            "hierarchy": hierarchy_loss,
            "consistency": consistency_loss,
        }

    @staticmethod
    def _view_consistency(embeddings: torch.Tensor, num_views: int) -> torch.Tensor:
        """Mean cosine distance between augmented views of each physical crop.

        The trainer concatenates complete view batches in view-major order:
        ``[view_1(batch), view_2(batch), ...]``.
        """

        if num_views <= 1:
            return embeddings.sum() * 0.0
        if len(embeddings) % num_views:
            raise ValueError("Embedding batch size must be divisible by num_views")
        normalized = F.normalize(embeddings, dim=-1)
        grouped = normalized.reshape(num_views, len(embeddings) // num_views, -1)
        centroid = F.normalize(grouped.mean(dim=0), dim=-1)
        return (1.0 - (grouped * centroid.unsqueeze(0)).sum(dim=-1)).mean()
