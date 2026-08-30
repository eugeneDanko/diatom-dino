"""Shared projector with separate genus and species embedding spaces."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HierarchicalProjectionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.projection_dim = int(projection_dim)
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
        )
        self.genus_head = nn.Linear(projection_dim, projection_dim)
        self.species_head = nn.Linear(projection_dim, projection_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(features)
        genus_embedding = F.normalize(self.genus_head(shared), dim=-1)
        species_embedding = F.normalize(self.species_head(shared), dim=-1)
        return genus_embedding, species_embedding

