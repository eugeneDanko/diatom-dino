"""Grid-search calibration for similarity, margin and agreement thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def binary_score_metrics(scores: Sequence[float], positives: Sequence[bool]) -> dict[str, float]:
    """Return threshold-free diagnostics for a score where larger means known."""
    values = np.asarray(scores, dtype=np.float64)
    expected = np.asarray(positives, dtype=bool)
    if values.size == 0 or not expected.any() or expected.all():
        return {"auroc": float("nan"), "average_precision": float("nan"), "fpr_at_95_tpr": float("nan")}

    order = np.argsort(-values, kind="mergesort")
    labels = expected[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    tpr = tp / int(expected.sum())
    fpr = fp / int((~expected).sum())
    # Include the reject-all origin and accept-all endpoint.
    curve_tpr = np.r_[0.0, tpr, 1.0]
    curve_fpr = np.r_[0.0, fpr, 1.0]
    auroc = float(np.sum(np.diff(curve_fpr) * (curve_tpr[:-1] + curve_tpr[1:]) / 2.0))
    precision = tp / np.arange(1, len(labels) + 1)
    average_precision = float(precision[labels].mean())
    eligible = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[eligible[0]]) if eligible.size else 1.0
    return {"auroc": auroc, "average_precision": average_precision, "fpr_at_95_tpr": fpr95}


@dataclass(frozen=True)
class CalibrationRecord:
    similarity: float
    margin: float
    agreement: float
    should_accept: bool


@dataclass(frozen=True)
class CalibratedLevel:
    similarity: float
    margin: float
    agreement: float
    balanced_accuracy: float
    positive_recall: float = 0.0
    negative_rejection: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    selection_reason: str = "maximum_balanced_accuracy"


def _balanced_accuracy(expected: np.ndarray, predicted: np.ndarray) -> float:
    positive = expected
    negative = ~expected
    true_positive_rate = float(predicted[positive].mean()) if positive.any() else 1.0
    true_negative_rate = float((~predicted[negative]).mean()) if negative.any() else 1.0
    return (true_positive_rate + true_negative_rate) / 2.0


def _safe_threshold(value: float) -> float:
    """Keep an observed float32 boundary inclusive after JSON/Python conversion."""
    return float(np.nextafter(np.float32(value), np.float32(-np.inf)))


def calibrate_level(
    records: Sequence[CalibrationRecord],
    *,
    similarity_grid: Iterable[float] | None = None,
    margin_grid: Iterable[float] | None = None,
    agreement_grid: Iterable[float] | None = None,
    minimum_positive_recall: float | None = None,
    enabled_signals: Sequence[str] = ("similarity", "margin", "agreement"),
) -> CalibratedLevel:
    if not records:
        raise ValueError("Calibration records are empty")
    similarity = np.asarray([record.similarity for record in records], dtype=np.float32)
    margin = np.asarray([record.margin for record in records], dtype=np.float32)
    agreement = np.asarray([record.agreement for record in records], dtype=np.float32)
    expected = np.asarray([record.should_accept for record in records], dtype=bool)
    if minimum_positive_recall is not None and (not expected.any() or expected.all()):
        raise ValueError("Calibration requires both accepted and rejected examples")
    if minimum_positive_recall is not None and not 0.0 <= minimum_positive_recall <= 1.0:
        raise ValueError("minimum_positive_recall must be between 0 and 1")
    enabled = set(enabled_signals)
    invalid = enabled - {"similarity", "margin", "agreement"}
    if invalid or not enabled:
        raise ValueError(f"Invalid enabled_signals: {sorted(invalid) or sorted(enabled)}")
    similarity_grid = list(
        (similarity_grid or np.quantile(similarity, np.linspace(0.05, 0.95, 19)))
        if "similarity" in enabled else [-1.000001]
    )
    margin_grid = list(
        (margin_grid or np.quantile(margin, np.linspace(0.0, 0.9, 10)))
        if "margin" in enabled else [-0.000001]
    )
    agreement_grid = list(
        (agreement_grid or sorted(set(agreement.tolist())))
        if "agreement" in enabled else [0.0]
    )
    candidates: list[CalibratedLevel] = []
    for similarity_threshold in similarity_grid:
        for margin_threshold in margin_grid:
            for agreement_threshold in agreement_grid:
                predicted = (
                    (similarity >= similarity_threshold)
                    & (margin >= margin_threshold)
                    & (agreement >= agreement_threshold)
                )
                positive_recall = float(predicted[expected].mean()) if expected.any() else 1.0
                negative_rejection = float((~predicted[~expected]).mean()) if (~expected).any() else 1.0
                candidate = CalibratedLevel(
                    _safe_threshold(similarity_threshold),
                    _safe_threshold(margin_threshold),
                    _safe_threshold(agreement_threshold),
                    _balanced_accuracy(expected, predicted),
                    positive_recall,
                    negative_rejection,
                    int(expected.sum()),
                    int((~expected).sum()),
                )
                candidates.append(candidate)
    if minimum_positive_recall is not None:
        feasible = [item for item in candidates if item.positive_recall >= minimum_positive_recall]
        if feasible:
            best = max(
                feasible,
                key=lambda item: (
                    item.negative_rejection,
                    item.balanced_accuracy,
                    item.positive_recall,
                    item.similarity,
                    item.margin,
                    item.agreement,
                ),
            )
            return CalibratedLevel(
                **{
                    **best.__dict__,
                    "selection_reason": "maximum_rejection_at_minimum_positive_recall",
                }
            )
    best = max(
        candidates,
        key=lambda item: (
            item.balanced_accuracy,
            item.positive_recall,
            item.negative_rejection,
        ),
    )
    reason = (
        "maximum_balanced_accuracy_no_threshold_met_minimum_positive_recall"
        if minimum_positive_recall is not None
        else "maximum_balanced_accuracy"
    )
    return CalibratedLevel(**{**best.__dict__, "selection_reason": reason})
