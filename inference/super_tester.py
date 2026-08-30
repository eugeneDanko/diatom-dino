"""End-to-end evaluation including missed detections and hierarchical labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from classifier.metrics import accuracy, macro_f1
from core.base_tester import BaseTester
from core.clearml_logger import ExperimentLogger
from .report_generator import ReportGenerator
from .super_pipeline import DiatomDINOPipeline, Prediction


@dataclass(frozen=True)
class GroundTruthObject:
    object_id: str
    bbox: tuple[float, float, float, float]
    genus: str
    species: str
    source_cohort: str = "unknown"


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union > 0 else 0.0


def _yolo_line_to_xyxy(values: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    coordinates = values[1:]
    if len(coordinates) == 4:
        center_x, center_y, box_width, box_height = coordinates
        return (
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        )
    if len(coordinates) == 8:
        points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
        return (
            float(points[:, 0].min() * width),
            float(points[:, 1].min() * height),
            float(points[:, 0].max() * width),
            float(points[:, 1].max() * height),
        )
    raise ValueError(f"Unsupported YOLO label with {len(coordinates)} coordinates")


def load_ground_truth(
    *,
    table_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    metadata_path: str | Path | None = None,
    image_extension: str = ".png",
) -> dict[Path, list[GroundTruthObject]]:
    table_path = Path(table_path)
    table = pd.read_parquet(table_path) if table_path.suffix.lower() == ".parquet" else pd.read_csv(table_path)
    table = table.rename(columns={"id_image": "image_id", "id_object": "object_id"})
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata = (
            pd.read_parquet(metadata_path)
            if metadata_path.suffix.lower() == ".parquet"
            else pd.read_csv(metadata_path)
        ).rename(columns={"id_image": "image_id"})
        filename_column = next(
            (column for column in ("filename", "image_path") if column in metadata.columns), None
        )
        if filename_column is None:
            raise ValueError("Ground-truth metadata needs filename or image_path")
        metadata_columns = ["image_id", filename_column]
        for optional in ("source_cohort", "name_dataset"):
            if optional in metadata.columns and optional not in table.columns:
                metadata_columns.append(optional)
        table = table.merge(
            metadata[metadata_columns].rename(columns={filename_column: "image_path"}),
            on="image_id", how="left", validate="many_to_one",
        )
    required = {"image_id", "object_id", "genus", "species"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Ground-truth table is missing columns: {sorted(missing)}")
    result: dict[Path, list[GroundTruthObject]] = {}
    for image_id, rows in table.groupby("image_id", sort=False):
        rows = rows.copy()
        numeric_object_ids = pd.to_numeric(rows["object_id"], errors="coerce")
        if numeric_object_ids.notna().all():
            rows = rows.assign(_object_order=numeric_object_ids).sort_values("_object_order")
        if "image_path" in rows.columns and rows["image_path"].notna().any():
            raw_path = Path(str(rows["image_path"].dropna().iloc[0]))
            image_path = raw_path if raw_path.is_absolute() else Path(images_dir) / raw_path
        else:
            raw_name = str(image_id)
            filename = raw_name if Path(raw_name).suffix else f"{raw_name}{image_extension}"
            image_path = Path(images_dir) / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Ground-truth image does not exist: {image_path}")

        boxes: list[tuple[float, float, float, float]] = []
        if {"x1", "y1", "x2", "y2"}.issubset(rows.columns):
            boxes = [tuple(map(float, values)) for values in rows[["x1", "y1", "x2", "y2"]].to_numpy()]
        else:
            if labels_dir is None:
                raise ValueError("labels_dir is required when the table has no x1/y1/x2/y2 columns")
            label_path = Path(labels_dir) / f"{Path(str(image_id)).stem}.txt"
            with Image.open(image_path) as image:
                width, height = image.size
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != len(rows):
                raise ValueError(
                    f"Object count differs for {image_id}: table={len(rows)}, labels={len(lines)}"
                )
            boxes = [
                _yolo_line_to_xyxy([float(value) for value in line.split()], width, height)
                for line in lines
            ]
        result[image_path] = [
            GroundTruthObject(
                object_id=str(row["object_id"]),
                bbox=box,
                genus=str(row["genus"]),
                species=(
                    str(row["species"])
                    if str(row["species"]).startswith(f"{row['genus']}_")
                    else f"{row['genus']}_{row['species']}"
                ),
                source_cohort=str(row.get("source_cohort", row.get("name_dataset", "unknown"))).strip().casefold(),
            )
            for (_, row), box in zip(rows.iterrows(), boxes)
        ]
    return result


class SuperTester(BaseTester):
    def __init__(
        self,
        *,
        pipeline: DiatomDINOPipeline,
        ground_truth: dict[Path, list[GroundTruthObject]],
        output_dir: str | Path,
        logger: ExperimentLogger | None = None,
        match_iou: float = 0.5,
    ) -> None:
        super().__init__(output_dir=output_dir, logger=logger)
        self.pipeline = pipeline
        self.ground_truth = ground_truth
        self.match_iou = float(match_iou)
        self.serialized_predictions: dict[str, list[dict[str, Any]]] = {}
        self.object_outcomes: list[dict[str, Any]] = []

    def _match(
        self,
        predictions: list[Prediction],
        truth: list[GroundTruthObject],
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for prediction_index, prediction in enumerate(predictions):
            for truth_index, target in enumerate(truth):
                overlap = box_iou(prediction.bbox, target.bbox)
                if overlap >= self.match_iou:
                    candidates.append((overlap, prediction_index, truth_index))
        matches: list[tuple[int, int]] = []
        used_predictions: set[int] = set()
        used_truth: set[int] = set()
        for _, prediction_index, truth_index in sorted(candidates, reverse=True):
            if prediction_index in used_predictions or truth_index in used_truth:
                continue
            used_predictions.add(prediction_index)
            used_truth.add(truth_index)
            matches.append((prediction_index, truth_index))
        return matches

    @staticmethod
    def _calculate_metrics(
        records: list[tuple[list[Prediction], list[GroundTruthObject], list[tuple[int, int]]]],
    ) -> dict[str, Any]:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        conditional_true_genus: list[str] = []
        conditional_pred_genus: list[str] = []
        conditional_true_species: list[str] = []
        conditional_pred_species: list[str] = []
        end_true_genus: list[str] = []
        end_pred_genus: list[str] = []
        end_true_species: list[str] = []
        end_pred_species: list[str] = []
        correct_genus = 0
        correct_species = 0

        for predictions, truth, matches in records:
            matched_truth = {truth_index for _, truth_index in matches}
            true_positive += len(matches)
            false_positive += len(predictions) - len(matches)
            false_negative += len(truth) - len(matches)
            for prediction_index, truth_index in matches:
                prediction = predictions[prediction_index].decision
                target = truth[truth_index]
                predicted_genus = prediction.genus or "__unknown__"
                predicted_species = prediction.species or "__unknown__"
                conditional_true_genus.append(target.genus)
                conditional_pred_genus.append(predicted_genus)
                conditional_true_species.append(target.species)
                conditional_pred_species.append(predicted_species)
                end_true_genus.append(target.genus)
                end_pred_genus.append(predicted_genus)
                end_true_species.append(target.species)
                end_pred_species.append(predicted_species)
                correct_genus += int(predicted_genus == target.genus)
                correct_species += int(predicted_species == target.species)
            for truth_index, target in enumerate(truth):
                if truth_index in matched_truth:
                    continue
                end_true_genus.append(target.genus)
                end_pred_genus.append("__missed__")
                end_true_species.append(target.species)
                end_pred_species.append("__missed__")

        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        total_ground_truth = true_positive + false_negative
        penalized_denominator = total_ground_truth + false_positive
        return {
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_tp": true_positive,
            "detection_fp": false_positive,
            "detection_fn": false_negative,
            "conditional_accuracy_genus": accuracy(conditional_true_genus, conditional_pred_genus),
            "conditional_f1_macro_genus": macro_f1(conditional_true_genus, conditional_pred_genus),
            "conditional_accuracy_species": accuracy(conditional_true_species, conditional_pred_species),
            "conditional_f1_macro_species": macro_f1(conditional_true_species, conditional_pred_species),
            "end_to_end_accuracy_genus": accuracy(end_true_genus, end_pred_genus),
            "end_to_end_f1_macro_genus": macro_f1(end_true_genus, end_pred_genus),
            "end_to_end_accuracy_species": accuracy(end_true_species, end_pred_species),
            "end_to_end_f1_macro_species": macro_f1(end_true_species, end_pred_species),
            "genus_success_rate": correct_genus / max(1, total_ground_truth),
            "species_success_rate": correct_species / max(1, total_ground_truth),
            "genus_system_score": correct_genus / max(1, penalized_denominator),
            "species_system_score": correct_species / max(1, penalized_denominator),
            "false_positives_per_image": false_positive / max(1, len(records)),
            "num_images": len(records),
            "num_ground_truth_objects": total_ground_truth,
        }

    def run_test(self) -> dict[str, Any]:
        records: list[
            tuple[list[Prediction], list[GroundTruthObject], list[tuple[int, int]]]
        ] = []
        cohort_records: dict[
            str, list[tuple[list[Prediction], list[GroundTruthObject], list[tuple[int, int]]]]
        ] = {}
        for image_path, truth in self.ground_truth.items():
            predictions = self.pipeline.predict_single(image_path)
            self.serialized_predictions[str(image_path)] = [item.to_dict() for item in predictions]
            matches = self._match(predictions, truth)
            record = (predictions, truth, matches)
            records.append(record)
            for cohort in {target.source_cohort for target in truth if target.source_cohort}:
                cohort_records.setdefault(cohort, []).append(record)
            matched_predictions = {prediction_index for prediction_index, _ in matches}
            matched_truth = {truth_index for _, truth_index in matches}
            for prediction_index, truth_index in matches:
                prediction = predictions[prediction_index]
                target = truth[truth_index]
                self.object_outcomes.append({
                    "image_path": str(image_path), "object_id": target.object_id,
                    "source_cohort": target.source_cohort, "outcome": "matched",
                    "true_genus": target.genus, "true_species": target.species,
                    "gate_probability": prediction.gate_probability,
                    "gate_accepted": prediction.gate_accepted,
                    "rejection_stage": prediction.rejection_stage,
                    "decision_status": prediction.decision.status,
                    "predicted_genus": prediction.decision.genus or "Unknown",
                    "predicted_species": prediction.decision.species or "Unknown",
                    "genus_correct": float(prediction.decision.genus == target.genus),
                    "species_correct": float(prediction.decision.species == target.species),
                })
            for truth_index, target in enumerate(truth):
                if truth_index not in matched_truth:
                    self.object_outcomes.append({
                        "image_path": str(image_path), "object_id": target.object_id,
                        "source_cohort": target.source_cohort, "outcome": "missed_by_yolo",
                        "true_genus": target.genus, "true_species": target.species,
                        "gate_probability": None, "gate_accepted": False,
                        "rejection_stage": "yolo", "decision_status": "missed",
                        "predicted_genus": "Missed", "predicted_species": "Missed",
                        "genus_correct": 0.0, "species_correct": 0.0,
                    })
            image_cohort = truth[0].source_cohort if truth else "unknown"
            for prediction_index, prediction in enumerate(predictions):
                if prediction_index not in matched_predictions:
                    self.object_outcomes.append({
                        "image_path": str(image_path), "object_id": "",
                        "source_cohort": image_cohort, "outcome": "false_positive_yolo",
                        "true_genus": "", "true_species": "",
                        "gate_probability": prediction.gate_probability,
                        "gate_accepted": prediction.gate_accepted,
                        "rejection_stage": prediction.rejection_stage,
                        "decision_status": prediction.decision.status,
                        "predicted_genus": prediction.decision.genus or "Unknown",
                        "predicted_species": prediction.decision.species or "Unknown",
                        "genus_correct": 0.0, "species_correct": 0.0,
                    })
        overall = self._calculate_metrics(records)
        result = {f"overall/{key}": value for key, value in overall.items()}
        for cohort, cohort_items in sorted(cohort_records.items()):
            result.update({
                f"{cohort}/{key}": value
                for key, value in self._calculate_metrics(cohort_items).items()
            })
        return result

    def generate_report(self, results: dict[str, Any]) -> Path:
        metrics_path = self.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        metric_rows = [
            {"cohort": cohort, "metric": metric, "value": value}
            for key, value in results.items()
            for cohort, metric in [key.split("/", 1)]
        ]
        metrics_table = self.output_dir / "metrics_by_cohort.csv"
        pd.DataFrame(metric_rows).to_csv(metrics_table, index=False)
        self.logger.log_artifact("metrics_by_cohort", metrics_table)
        outcomes = pd.DataFrame(self.object_outcomes)
        outcomes_path = self.output_dir / "object_outcomes.csv"
        outcomes.to_csv(outcomes_path, index=False)
        truth_outcomes = outcomes[outcomes["outcome"].ne("false_positive_yolo")].copy()
        truth_outcomes["detected"] = truth_outcomes["outcome"].eq("matched").astype(float)
        truth_outcomes["passed_resnet"] = (
            truth_outcomes["outcome"].eq("matched") & truth_outcomes["gate_accepted"].eq(True)
        ).astype(float)
        truth_outcomes["genus_accepted"] = truth_outcomes["decision_status"].isin(
            ["known_species", "known_genus_unknown_species"]
        ).astype(float)
        truth_outcomes["species_accepted"] = truth_outcomes["decision_status"].eq(
            "known_species"
        ).astype(float)
        funnel = truth_outcomes.groupby("source_cohort")[[
            "detected", "passed_resnet", "genus_accepted", "species_accepted",
            "genus_correct", "species_correct",
        ]].agg(["count", "mean"])
        funnel.columns = [f"{metric}_{stat}" for metric, stat in funnel.columns]
        funnel = funnel.reset_index()
        funnel_path = self.output_dir / "failure_funnel_by_cohort.csv"
        funnel.to_csv(funnel_path, index=False)
        per_class = truth_outcomes.groupby(
            ["source_cohort", "true_genus", "true_species"], dropna=False
        )[["detected", "passed_resnet", "genus_correct", "species_correct"]].agg(
            ["count", "mean"]
        )
        per_class.columns = [f"{metric}_{stat}" for metric, stat in per_class.columns]
        per_class = per_class.reset_index()
        per_class_path = self.output_dir / "per_class_e2e_diagnostics.csv"
        per_class.to_csv(per_class_path, index=False)
        self.logger.log_table("E2E failure funnel", funnel, 0, series="test")
        self.logger.log_table("E2E per-class diagnostics", per_class, 0, series="test")
        for name, path in (
            ("object_outcomes", outcomes_path), ("failure_funnel_by_cohort", funnel_path),
            ("per_class_e2e_diagnostics", per_class_path),
        ):
            self.logger.log_artifact(name, path)
        return ReportGenerator(self.output_dir).generate(
            self.serialized_predictions,
            metrics=results,
            name="end_to_end_report",
        )
