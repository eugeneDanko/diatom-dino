"""Diagnostic 2D embedding plots with UMAP → PCA fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Sequence[str],
    *,
    output_path: str | Path,
    title: str,
    random_state: int = 42,
) -> Path | None:
    if len(embeddings) < 2:
        return None
    method = "UMAP"
    try:
        from umap import UMAP

        reducer = UMAP(n_components=2, metric="cosine", random_state=random_state)
        coordinates = reducer.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA

        method = "PCA"
        coordinates = PCA(n_components=2, random_state=random_state).fit_transform(embeddings)
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_array = np.asarray(labels)
    unique_labels = sorted(set(str(label) for label in labels))
    figure, axis = plt.subplots(figsize=(10, 8))
    color_map = plt.get_cmap("tab20")
    for index, label in enumerate(unique_labels):
        mask = labels_array.astype(str) == label
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18,
            alpha=0.75,
            label=label,
            color=color_map(index % 20),
        )
    axis.set_title(f"{title} ({method})")
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    if len(unique_labels) <= 30:
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output

