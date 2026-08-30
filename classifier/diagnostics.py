"""Rich validation diagnostics for gallery-based DINO retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from core.clearml_logger import ExperimentLogger
from inference.calibration import CalibrationRecord, calibrate_level
from .crop_dataset import LabelCodec
from .metrics import l2_normalize


def _rank_neighbors(queries: np.ndarray, references: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    similarities = l2_normalize(queries) @ l2_normalize(references).T
    order = np.argsort(-similarities, axis=1)
    return similarities, order


def _topk_metrics(
    order: np.ndarray,
    reference_labels: Sequence[int],
    true_labels: Sequence[int],
    ks: Sequence[int],
    prefix: str,
) -> dict[str, float]:
    reference = np.asarray(reference_labels)
    truth = np.asarray(true_labels)
    metrics: dict[str, float] = {}
    for k in sorted({max(1, int(value)) for value in ks}):
        labels = reference[order[:, : min(k, order.shape[1])]]
        hits = (labels == truth[:, None]).any(axis=1)
        metrics[f"top{k}_accuracy_{prefix}"] = float(hits.mean())
        class_recalls = [float(hits[truth == label].mean()) for label in np.unique(truth)]
        metrics[f"recall_at_{k}_{prefix}"] = float(np.mean(class_recalls))
    return metrics


def _confusion(
    truth: Sequence[int], prediction: Sequence[int], names: Mapping[int, str]
) -> tuple[np.ndarray, list[str]]:
    labels = sorted(set(int(value) for value in truth) | set(int(value) for value in prediction))
    positions = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for expected, predicted in zip(truth, prediction):
        matrix[positions[int(expected)], positions[int(predicted)]] += 1
    return matrix, [names.get(label, f"unknown:{label}") for label in labels]


def _plot_confusion(matrix: np.ndarray, labels: Sequence[str], title: str):
    size = min(18, max(7, 0.45 * len(labels)))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(title=title, xlabel="Predicted", ylabel="True")
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def _per_class_table(
    truth: Sequence[int], prediction: Sequence[int], names: Mapping[int, str]
) -> pd.DataFrame:
    truth_array = np.asarray(truth)
    prediction_array = np.asarray(prediction)
    rows: list[dict[str, Any]] = []
    for label in sorted(set(int(value) for value in truth_array)):
        true_positive = int(np.sum((truth_array == label) & (prediction_array == label)))
        false_positive = int(np.sum((truth_array != label) & (prediction_array == label)))
        false_negative = int(np.sum((truth_array == label) & (prediction_array != label)))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "class_id": label,
                "class_name": names.get(label, str(label)),
                "support": int(np.sum(truth_array == label)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def _similarity_samples(
    similarities: np.ndarray,
    true_labels: Sequence[int],
    reference_labels: Sequence[int],
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(true_labels)[:, None]
    reference = np.asarray(reference_labels)[None, :]
    positive = similarities[truth == reference]
    negative = similarities[truth != reference]
    generator = np.random.default_rng(seed)
    if len(positive) > maximum:
        positive = generator.choice(positive, maximum, replace=False)
    if len(negative) > maximum:
        negative = generator.choice(negative, maximum, replace=False)
    return positive, negative


def _plot_similarity(positive: np.ndarray, negative: np.ndarray, title: str):
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(negative, bins=40, alpha=0.55, density=True, label="negative")
    axis.hist(positive, bins=40, alpha=0.55, density=True, label="positive")
    axis.set(title=title, xlabel="Cosine similarity", ylabel="Density")
    axis.legend()
    figure.tight_layout()
    return figure


def _embedding_projection(values: np.ndarray, seed: int, method: str) -> tuple[np.ndarray, str]:
    normalized = l2_normalize(values)
    if method in {"auto", "umap"}:
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(normalized), "UMAP"
        except ImportError:
            if method == "umap":
                raise
    from sklearn.manifold import TSNE

    perplexity = max(2, min(30, len(normalized) - 1))
    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(normalized)
    return projection, "t-SNE"


def _plot_projection(
    points: np.ndarray, labels: Sequence[int], names: Mapping[int, str], title: str
):
    figure, axis = plt.subplots(figsize=(10, 8))
    labels_array = np.asarray(labels)
    unique = sorted(set(int(value) for value in labels_array))
    for label in unique:
        mask = labels_array == label
        axis.scatter(points[mask, 0], points[mask, 1], s=16, alpha=0.75, label=names.get(label, str(label)))
    axis.set_title(title)
    if len(unique) <= 20:
        axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    return figure


def _plot_neighbor_examples(
    query_paths: Sequence[str],
    reference_paths: Sequence[str],
    order: np.ndarray,
    similarities: np.ndarray,
    true_labels: Sequence[int],
    reference_labels: Sequence[int],
    names: Mapping[int, str],
    *,
    query_count: int,
    top_k: int,
):
    rows = min(query_count, len(query_paths))
    columns = 1 + min(top_k, order.shape[1])
    figure, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 2.8 * rows), squeeze=False)
    for row in range(rows):
        with Image.open(query_paths[row]) as image:
            axes[row, 0].imshow(image.convert("RGB"))
        axes[row, 0].set_title(f"Query\n{names[int(true_labels[row])]}", fontsize=8)
        axes[row, 0].axis("off")
        for column, neighbor in enumerate(order[row, : columns - 1], 1):
            with Image.open(reference_paths[int(neighbor)]) as image:
                axes[row, column].imshow(image.convert("RGB"))
            axes[row, column].set_title(
                f"#{column} {names[int(reference_labels[int(neighbor)])]}\n"
                f"cos={similarities[row, int(neighbor)]:.3f}",
                fontsize=8,
            )
            axes[row, column].axis("off")
    figure.suptitle("Gallery nearest-neighbor examples")
    figure.tight_layout()
    return figure


def _proxy_calibration(
    similarities: np.ndarray,
    order: np.ndarray,
    true_labels: Sequence[int],
    reference_labels: Sequence[int],
    top_k: int,
) -> dict[str, float]:
    reference = np.asarray(reference_labels)
    truth = np.asarray(true_labels)
    records: list[CalibrationRecord] = []
    for row, ranked in enumerate(order):
        neighbors = ranked[: min(max(2, top_k), len(ranked))]
        scores = similarities[row, neighbors]
        labels = reference[neighbors]
        top_label = int(labels[0])
        agreement = float(np.mean(labels == top_label))
        margin = float(scores[0] - scores[1]) if len(scores) > 1 else float(scores[0])
        records.append(
            CalibrationRecord(float(scores[0]), margin, agreement, top_label == int(truth[row]))
        )
    calibrated = calibrate_level(records)
    return {
        "open_set_proxy_similarity": calibrated.similarity,
        "open_set_proxy_margin": calibrated.margin,
        "open_set_proxy_agreement": calibrated.agreement,
        "open_set_proxy_balanced_accuracy": calibrated.balanced_accuracy,
    }


def run_retrieval_diagnostics(
    *,
    epoch: int,
    query_genus_embeddings: np.ndarray,
    query_species_embeddings: np.ndarray,
    query_genus_labels: Sequence[int],
    query_species_labels: Sequence[int],
    query_ids: Sequence[str],
    query_paths: Sequence[str],
    reference_genus_embeddings: np.ndarray,
    reference_species_embeddings: np.ndarray,
    reference_genus_labels: Sequence[int],
    reference_species_labels: Sequence[int],
    reference_ids: Sequence[str],
    reference_paths: Sequence[str],
    genus_predictions: Sequence[int],
    species_predictions: Sequence[int],
    label_codec: LabelCodec,
    logger: ExperimentLogger,
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Save and report diagnostics for one validation epoch."""

    settings = dict(config)
    reporting_stage = str(settings.get("reporting_stage", ""))
    genus_only = reporting_stage == "stage1"
    ks = settings.get("recall_at_k", [1, 3, 5])
    maximum_pairs = int(settings.get("max_similarity_pairs", 50_000))
    maximum_embeddings = int(settings.get("max_embedding_points", 1_000))
    maximum_neighbors = int(settings.get("nearest_neighbor_queries", 25))
    neighbor_examples = int(settings.get("nearest_neighbor_examples", 5))
    top_k = int(settings.get("nearest_neighbors_k", 5))
    seed = int(settings.get("seed", 42)) + int(epoch)
    epoch_dir = Path(output_dir) / "diagnostics" / f"epoch_{epoch + 1:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    genus_similarities, genus_order = _rank_neighbors(query_genus_embeddings, reference_genus_embeddings)
    species_similarities, species_order = _rank_neighbors(
        query_species_embeddings, reference_species_embeddings
    )
    metrics = {}
    metrics.update(_topk_metrics(genus_order, reference_genus_labels, query_genus_labels, ks, "genus"))
    metrics.update(
        _topk_metrics(species_order, reference_species_labels, query_species_labels, ks, "species_global")
    )
    metrics.update(
        _proxy_calibration(
            species_similarities,
            species_order,
            query_species_labels,
            reference_species_labels,
            top_k,
        )
    )

    genus_matrix, genus_names = _confusion(
        query_genus_labels, genus_predictions, label_codec.id_to_genus
    )
    species_matrix, species_names = _confusion(
        query_species_labels, species_predictions, label_codec.id_to_species
    )
    figures = {
        "confusion_genus": _plot_confusion(genus_matrix, genus_names, "Genus confusion matrix"),
    }
    if not genus_only:
        figures["confusion_species"] = _plot_confusion(
            species_matrix, species_names, "Species confusion matrix"
        )
    positive, negative = _similarity_samples(
        species_similarities, query_species_labels, reference_species_labels, maximum_pairs, seed
    )
    metrics.update(
        {
            "cosine_positive_mean": float(np.mean(positive)),
            "cosine_positive_std": float(np.std(positive)),
            "cosine_negative_mean": float(np.mean(negative)),
            "cosine_negative_std": float(np.std(negative)),
            "cosine_separation": float(np.mean(positive) - np.mean(negative)),
        }
    )
    if not genus_only:
        figures["cosine_similarity_species"] = _plot_similarity(
            positive, negative, "Species cosine similarity: positive vs negative"
        )
    if neighbor_examples > 0 and not genus_only:
        figures["nearest_neighbor_examples"] = _plot_neighbor_examples(
            query_paths,
            reference_paths,
            species_order,
            species_similarities,
            query_species_labels,
            reference_species_labels,
            label_codec.id_to_species,
            query_count=neighbor_examples,
            top_k=top_k,
        )

    generator = np.random.default_rng(seed)
    sample_count = min(maximum_embeddings, len(query_species_embeddings))
    sample = generator.choice(len(query_species_embeddings), sample_count, replace=False)
    if sample_count >= 3 and not genus_only:
        points, method = _embedding_projection(
            query_species_embeddings[sample], seed, str(settings.get("embedding_method", "auto"))
        )
        figures["embedding_projection"] = _plot_projection(
            points,
            np.asarray(query_species_labels)[sample],
            label_codec.id_to_species,
            f"{method} validation species embeddings",
        )

    genus_table = _per_class_table(query_genus_labels, genus_predictions, label_codec.id_to_genus)
    species_table = _per_class_table(
        query_species_labels, species_predictions, label_codec.id_to_species
    )
    genus_table.to_csv(epoch_dir / "per_class_genus.csv", index=False)
    species_table.to_csv(epoch_dir / "per_class_species.csv", index=False)
    logger.log_table("per_class_genus", genus_table, epoch)
    if not genus_only:
        logger.log_table("per_class_species", species_table, epoch)

    nearest_rows: list[dict[str, Any]] = []
    reference_species = np.asarray(reference_species_labels)
    for query_index in range(min(maximum_neighbors, len(query_ids))):
        for rank, neighbor in enumerate(species_order[query_index, :top_k], 1):
            nearest_rows.append(
                {
                    "query_id": str(query_ids[query_index]),
                    "true_species": label_codec.id_to_species[int(query_species_labels[query_index])],
                    "rank": rank,
                    "reference_id": str(reference_ids[int(neighbor)]),
                    "neighbor_species": label_codec.id_to_species[int(reference_species[int(neighbor)])],
                    "cosine_similarity": float(species_similarities[query_index, int(neighbor)]),
                }
            )
    nearest_table = pd.DataFrame(nearest_rows)
    nearest_table.to_csv(epoch_dir / "nearest_neighbors.csv", index=False)
    if not genus_only:
        logger.log_table("nearest_neighbors_gallery_query", nearest_table, epoch)

    for name, figure in figures.items():
        path = epoch_dir / f"{name}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        logger.log_figure(name, figure, epoch)
        plt.close(figure)
    (epoch_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
