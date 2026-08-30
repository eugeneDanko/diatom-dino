"""Stage-specific ClearML reporting rules for the DINO V3 curriculum."""

from __future__ import annotations

from typing import Any, Mapping


STAGE_REPORTS: dict[str, dict[str, Any]] = {
    "stage1": {
        "objective": "Separate genera on held-out open validation data",
        "hypothesis": "A: fine-tuning improves genus geometry",
        "target_metrics": [
            "val/accuracy_genus", "val/precision_macro_genus",
            "val/recall_macro_genus", "val/f1_macro_genus",
            "val/top1_accuracy_genus", "val/top3_accuracy_genus",
            "val/top5_accuracy_genus", "val/recall_at_1_genus",
            "val/recall_at_3_genus", "val/recall_at_5_genus",
        ],
        "supporting_metrics": [],
        "hypothesis_checks": [
            "Compare Stage 0 and Stage 1 on the same open validation/test",
            "Bootstrap 95% CI for the change in genus Macro F1",
            "Confirm improvement for target genera and per-class recall",
        ],
    },
    "stage2": {
        "objective": "Separate seen species while preserving genus structure",
        "hypothesis": "A: DINO embeddings support hierarchical species retrieval",
        "target_metrics": [
            "val/accuracy_species", "val/precision_macro_species",
            "val/recall_macro_species", "val/f1_macro_species",
            "val/top1_accuracy_species", "val/top3_accuracy_species",
            "val/top5_accuracy_species", "val/recall_at_1_species",
            "val/recall_at_3_species", "val/recall_at_5_species",
        ],
        "supporting_metrics": [
            "val/accuracy_genus", "val/precision_macro_genus",
            "val/recall_macro_genus", "val/f1_macro_genus",
        ],
        "hypothesis_checks": [
            "Compare Stage 0/1/2 on frozen open test",
            "Report large/medium/small and hard-negative cohorts separately",
            "Verify that genus Macro F1 is not degraded",
        ],
    },
    "stage3": {
        "objective": "Adapt to NII/few-shot data without catastrophic forgetting",
        "hypothesis": "B/C: transfer to NII and few-shot target species",
        "target_metrics": [
            "val/accuracy_species", "val/precision_macro_species",
            "val/recall_macro_species", "val/f1_macro_species",
            "val/top1_accuracy_species", "val/top3_accuracy_species",
            "val/top5_accuracy_species", "val/recall_at_1_species",
            "val/recall_at_3_species", "val/recall_at_5_species",
            "val/accuracy_genus", "val/precision_macro_genus",
            "val/recall_macro_genus", "val/f1_macro_genus",
        ],
        "supporting_metrics": [],
        "hypothesis_checks": [
            "B: compare Stage 2/3 on frozen NII final test with bootstrap CI",
            "B: measure Stage 3 degradation on the same frozen open test",
            "C0: evaluate unseen-species genus geometry and open-set AUROC/AUPRC",
            "C1/C2: run repeated 1/3/5-shot support/query episodes",
        ],
    },
}


def stage_report(stage: str) -> dict[str, Any]:
    try:
        return dict(STAGE_REPORTS[str(stage)])
    except KeyError as exc:
        raise ValueError(f"Unknown DINO V3 reporting stage: {stage}") from exc


def reporting_config(stage: str) -> dict[str, Any]:
    report = stage_report(stage)
    return {
        "stage": stage,
        "objective": report["objective"],
        "hypothesis": report["hypothesis"],
        "target_metrics": list(report["target_metrics"]),
        "supporting_metrics": list(report["supporting_metrics"]),
        "hypothesis_checks": list(report["hypothesis_checks"]),
        "log_losses": True,
        "log_unlisted_scalars": False,
    }


def is_loss_metric(key: str) -> bool:
    return "loss" in str(key).split("/")[-1]


def should_log_scalar(key: str, config: Mapping[str, Any] | None) -> bool:
    settings = dict(config or {})
    if not settings.get("stage"):
        return True
    if is_loss_metric(key):
        return bool(settings.get("log_losses", True))
    allowed = set(settings.get("target_metrics", [])) | set(settings.get("supporting_metrics", []))
    if key in allowed or key.startswith("checkpoint/") or key.startswith("early_stopping/"):
        return True
    return bool(settings.get("log_unlisted_scalars", False))


def stage_chart_coordinates(key: str, config: Mapping[str, Any] | None) -> tuple[str, str] | None:
    settings = dict(config or {})
    if not should_log_scalar(key, settings):
        return None
    if is_loss_metric(key):
        return "Optimization/Losses", key
    if key.startswith("checkpoint/") or key.startswith("early_stopping/"):
        return "Stage progress/Checkpoint", key
    level = "Genus" if "genus" in key else "Species" if "species" in key else "Other"
    if key in set(settings.get("target_metrics", [])):
        return f"Target metrics/{level}", key
    return f"Supporting metrics/{level}", key
