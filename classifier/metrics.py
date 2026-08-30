"""Small dependency-light retrieval and classification metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def exact_knn_predict(
    queries: np.ndarray,
    references: np.ndarray,
    reference_labels: Sequence[int],
    *,
    k: int = 1,
    exclude_self: bool = False,
) -> np.ndarray:
    queries = l2_normalize(queries)
    references = l2_normalize(references)
    similarities = queries @ references.T
    if exclude_self:
        if len(queries) != len(references):
            raise ValueError("exclude_self requires equally sized query/reference arrays")
        np.fill_diagonal(similarities, -np.inf)
    effective_k = min(max(1, int(k)), references.shape[0] - int(exclude_self))
    if effective_k <= 0:
        return np.full(len(queries), -1, dtype=np.int64)
    indices = np.argpartition(-similarities, effective_k - 1, axis=1)[:, :effective_k]
    labels = np.asarray(reference_labels)
    predictions: list[int] = []
    for row, neighbor_indices in enumerate(indices):
        ordered = neighbor_indices[np.argsort(-similarities[row, neighbor_indices])]
        counts = Counter(int(labels[index]) for index in ordered)
        max_count = max(counts.values())
        winners = {label for label, count in counts.items() if count == max_count}
        predictions.append(next(int(labels[index]) for index in ordered if int(labels[index]) in winners))
    return np.asarray(predictions, dtype=np.int64)


def weighted_knn_predict(
    queries: np.ndarray,
    references: np.ndarray,
    reference_labels: Sequence[int],
    *,
    k: int = 3,
    temperature: float = 0.05,
) -> np.ndarray:
    """Class-balanced soft voting over the nearest references."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    queries = l2_normalize(queries)
    references = l2_normalize(references)
    labels = np.asarray(reference_labels)
    similarities = queries @ references.T
    effective_k = min(max(1, int(k)), len(references))
    predictions: list[int] = []
    for row in similarities:
        neighbors = np.argpartition(-row, effective_k - 1)[:effective_k]
        neighbors = neighbors[np.argsort(-row[neighbors])]
        local_scores = row[neighbors]
        weights = np.exp((local_scores - local_scores.max()) / temperature)
        votes: dict[int, float] = {}
        best_similarity: dict[int, float] = {}
        for neighbor, weight in zip(neighbors, weights):
            label = int(labels[neighbor])
            votes[label] = votes.get(label, 0.0) + float(weight)
            best_similarity[label] = max(best_similarity.get(label, -np.inf), float(row[neighbor]))
        predictions.append(max(votes, key=lambda label: (votes[label], best_similarity[label], -label)))
    return np.asarray(predictions, dtype=np.int64)


def prototype_weighted_predict(
    queries: np.ndarray,
    references: np.ndarray,
    reference_labels: Sequence[int],
    *,
    prototype_weight: float = 0.5,
    neighbors_weight: float = 0.5,
    neighbors_per_class: int = 3,
) -> np.ndarray:
    """Combine a normalized class prototype with its strongest references."""

    if prototype_weight < 0 or neighbors_weight < 0:
        raise ValueError("prototype and neighbor weights must be non-negative")
    if prototype_weight + neighbors_weight <= 0:
        raise ValueError("at least one prototype/neighbor weight must be positive")
    queries = l2_normalize(queries)
    references = l2_normalize(references)
    labels = np.asarray(reference_labels)
    unique_labels = sorted(int(value) for value in np.unique(labels))
    class_indices = {label: np.flatnonzero(labels == label) for label in unique_labels}
    prototypes = np.stack(
        [l2_normalize(references[indices].mean(axis=0, keepdims=True))[0] for indices in class_indices.values()]
    )
    predictions: list[int] = []
    for query in queries:
        prototype_scores = prototypes @ query
        class_scores: list[float] = []
        for class_index, label in enumerate(unique_labels):
            similarities = references[class_indices[label]] @ query
            count = min(max(1, int(neighbors_per_class)), len(similarities))
            strongest = np.partition(similarities, len(similarities) - count)[-count:]
            class_scores.append(
                prototype_weight * float(prototype_scores[class_index])
                + neighbors_weight * float(strongest.mean())
            )
        predictions.append(unique_labels[int(np.argmax(class_scores))])
    return np.asarray(predictions, dtype=np.int64)


def prototype_softmax_predict(
    queries: np.ndarray,
    references: np.ndarray,
    reference_labels: Sequence[int],
    *,
    k: int = 3,
    temperature: float = 0.03,
    prototype_weight: float = 0.4,
    neighbors_weight: float = 0.6,
) -> np.ndarray:
    """Score each class by its prototype and a softmax-weighted top-k score.

    Softmax is computed independently inside every class, so classes with more
    gallery references do not receive extra voting mass. A low temperature is
    deliberate for cosine similarities that often differ only by 0.01-0.03.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if prototype_weight < 0 or neighbors_weight < 0:
        raise ValueError("prototype and neighbor weights must be non-negative")
    total_weight = prototype_weight + neighbors_weight
    if total_weight <= 0:
        raise ValueError("at least one prototype/neighbor weight must be positive")
    queries = l2_normalize(queries)
    references = l2_normalize(references)
    labels = np.asarray(reference_labels)
    unique_labels = sorted(int(value) for value in np.unique(labels))
    class_indices = {label: np.flatnonzero(labels == label) for label in unique_labels}
    prototypes = np.stack(
        [l2_normalize(references[indices].mean(axis=0, keepdims=True))[0] for indices in class_indices.values()]
    )
    predictions: list[int] = []
    for query in queries:
        prototype_scores = prototypes @ query
        class_scores: list[float] = []
        for class_index, label in enumerate(unique_labels):
            similarities = references[class_indices[label]] @ query
            count = min(max(1, int(k)), len(similarities))
            top = np.partition(similarities, len(similarities) - count)[-count:]
            weights = np.exp((top - top.max()) / temperature)
            softmax_score = float(np.sum(weights * top) / np.sum(weights))
            class_scores.append(
                (
                    prototype_weight * float(prototype_scores[class_index])
                    + neighbors_weight * softmax_score
                )
                / total_weight
            )
        predictions.append(unique_labels[int(np.argmax(class_scores))])
    return np.asarray(predictions, dtype=np.int64)


def retrieval_predict(
    queries: np.ndarray,
    references: np.ndarray,
    reference_labels: Sequence[int],
    strategy: Mapping[str, Any] | None = None,
) -> np.ndarray:
    settings = dict(strategy or {})
    method = str(settings.get("method", "nearest")).lower()
    if method == "nearest":
        return exact_knn_predict(queries, references, reference_labels, k=int(settings.get("k", 1)))
    if method == "weighted_knn":
        return weighted_knn_predict(
            queries,
            references,
            reference_labels,
            k=int(settings.get("k", 3)),
            temperature=float(settings.get("temperature", 0.05)),
        )
    if method == "prototype_weighted":
        return prototype_weighted_predict(
            queries,
            references,
            reference_labels,
            prototype_weight=float(settings.get("prototype_weight", 0.5)),
            neighbors_weight=float(settings.get("neighbors_weight", 0.5)),
            neighbors_per_class=int(settings.get("neighbors_per_class", 3)),
        )
    if method == "prototype_softmax":
        return prototype_softmax_predict(
            queries,
            references,
            reference_labels,
            k=int(settings.get("k", 3)),
            temperature=float(settings.get("temperature", 0.03)),
            prototype_weight=float(settings.get("prototype_weight", 0.4)),
            neighbors_weight=float(settings.get("neighbors_weight", 0.6)),
        )
    raise ValueError(f"Unknown retrieval method: {method}")


def hierarchical_knn_predict(
    queries: np.ndarray,
    references: np.ndarray,
    predicted_query_genera: Sequence[int],
    reference_genera: Sequence[int],
    reference_species: Sequence[int],
    *,
    k: int = 1,
    self_reference_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Predict species only among references of the predicted genus."""

    queries = l2_normalize(queries)
    references = l2_normalize(references)
    reference_genera_array = np.asarray(reference_genera)
    reference_species_array = np.asarray(reference_species)
    predictions: list[int] = []
    for query_index, (query, predicted_genus) in enumerate(zip(queries, predicted_query_genera)):
        allowed = np.flatnonzero(reference_genera_array == predicted_genus)
        if self_reference_indices is not None:
            own_index = int(self_reference_indices[query_index])
            allowed = allowed[allowed != own_index]
        if len(allowed) == 0:
            predictions.append(-1)
            continue
        similarities = references[allowed] @ query
        effective_k = min(max(1, int(k)), len(allowed))
        local = np.argpartition(-similarities, effective_k - 1)[:effective_k]
        local = local[np.argsort(-similarities[local])]
        labels = reference_species_array[allowed[local]]
        counts = Counter(int(label) for label in labels)
        maximum = max(counts.values())
        winners = {label for label, count in counts.items() if count == maximum}
        predictions.append(next(int(label) for label in labels if int(label) in winners))
    return np.asarray(predictions, dtype=np.int64)


def hierarchical_retrieval_predict(
    queries: np.ndarray,
    references: np.ndarray,
    predicted_query_genera: Sequence[int],
    reference_genera: Sequence[int],
    reference_species: Sequence[int],
    strategy: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply the configured strategy within each predicted genus."""

    reference_genera_array = np.asarray(reference_genera)
    reference_species_array = np.asarray(reference_species)
    predictions: list[int] = []
    for query, predicted_genus in zip(queries, predicted_query_genera):
        allowed = np.flatnonzero(reference_genera_array == predicted_genus)
        if not len(allowed):
            predictions.append(-1)
            continue
        prediction = retrieval_predict(
            np.asarray(query)[None, :],
            np.asarray(references)[allowed],
            reference_species_array[allowed],
            strategy,
        )
        predictions.append(int(prediction[0]))
    return np.asarray(predictions, dtype=np.int64)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    return float(np.mean(truth == prediction)) if len(truth) else 0.0


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    scores: list[float] = []
    for label in np.unique(truth):
        true_positive = np.sum((truth == label) & (prediction == label))
        false_positive = np.sum((truth != label) & (prediction == label))
        false_negative = np.sum((truth == label) & (prediction != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(float(2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def macro_precision_recall(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[float, float]:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    for label in np.unique(truth):
        true_positive = np.sum((truth == label) & (prediction == label))
        false_positive = np.sum((truth != label) & (prediction == label))
        false_negative = np.sum((truth == label) & (prediction != label))
        precision_scores.append(float(true_positive / max(1, true_positive + false_positive)))
        recall_scores.append(float(true_positive / max(1, true_positive + false_negative)))
    return (
        float(np.mean(precision_scores)) if precision_scores else 0.0,
        float(np.mean(recall_scores)) if recall_scores else 0.0,
    )


def retrieval_metrics(y_true: Sequence[int], y_pred: Sequence[int], prefix: str) -> dict[str, float]:
    precision, recall = macro_precision_recall(y_true, y_pred)
    return {
        f"accuracy_{prefix}": accuracy(y_true, y_pred),
        f"precision_macro_{prefix}": precision,
        f"recall_macro_{prefix}": recall,
        f"f1_macro_{prefix}": macro_f1(y_true, y_pred),
    }
