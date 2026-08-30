"""Build persistent genus/species indexes from the manually curated gallery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from classifier.dino_backbone import DINOBackbone
from classifier.projection_head import HierarchicalProjectionHead
from .faiss_retriever import FAISSRetriever


@torch.no_grad()
def build_gallery_indices(
    *,
    backbone: DINOBackbone,
    projection_head: HierarchicalProjectionHead,
    gallery_loader: Any,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
) -> tuple[FAISSRetriever, FAISSRetriever]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backbone.set_train_mode(False)
    projection_head.eval()
    genus_embeddings: list[np.ndarray] = []
    species_embeddings: list[np.ndarray] = []
    item_ids: list[str] = []
    genera: list[str] = []
    species: list[str] = []
    for batch in gallery_loader:
        images = batch["image"].to(backbone.device, non_blocking=True)
        raw = backbone.extract_features(images)
        genus_batch, species_batch = projection_head(raw)
        genus_embeddings.append(genus_batch.cpu().numpy())
        species_embeddings.append(species_batch.cpu().numpy())
        item_ids.extend(str(value) for value in batch["crop_id"])
        genera.extend(str(value) for value in batch["genus"])
        species.extend(str(value) for value in batch["species"])
    genus_index = FAISSRetriever()
    species_index = FAISSRetriever()
    genus_index.build_index(
        np.concatenate(genus_embeddings), item_ids=item_ids, genera=genera, species=species
    )
    species_index.build_index(
        np.concatenate(species_embeddings), item_ids=item_ids, genera=genera, species=species
    )
    genus_path = genus_index.save_index(output / "genus_index.npz")
    species_path = species_index.save_index(output / "species_index.npz")
    (output / "gallery_metadata.json").write_text(
        json.dumps(
            {
                "num_items": len(item_ids),
                "num_genera": len(set(genera)),
                "num_species": len(set(species)),
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "genus_index": genus_path.name,
                "species_index": species_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return genus_index, species_index

