"""Recall-constrained confidence calibration for a YOLO detector."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .yolo_model import Detection, YOLOModel


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _calibration_value(data: Mapping[str, Any], source_path: Path) -> str:
    if "calibration" in data:
        return str(data["calibration"])
    fallback = source_path.parent / "calibration.txt"
    if fallback.is_file():
        return "calibration.txt"
    raise KeyError(f"YOLO data YAML has no calibration split: {source_path}")


def build_ultralytics_calibration_yaml(
    source: str | Path,
    destination: str | Path,
) -> Path:
    """Map the custom calibration split to Ultralytics' supported ``val`` key."""

    source_path = Path(source)
    data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YOLO data YAML: {source_path}")
    calibration_value = _calibration_value(data, source_path)

    dataset_root = Path(str(data.get("path", source_path.parent)))
    if not dataset_root.is_absolute():
        dataset_root = (source_path.parent / dataset_root).resolve()
    calibration_path = Path(calibration_value)
    if not calibration_path.is_absolute():
        calibration_path = dataset_root / calibration_path
    if not calibration_path.is_dir() and not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration source does not exist: {calibration_path}")

    adapted = dict(data)
    adapted["path"] = str(dataset_root)
    adapted["val"] = str(calibration_path)
    adapted.pop("calibration", None)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        yaml.safe_dump(adapted, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return destination_path


def _metric(row: Mapping[str, Any], suffix: str) -> float:
    matches = [float(value) for key, value in row.items() if str(key).lower().endswith(suffix)]
    if not matches:
        raise KeyError(f"YOLO metrics do not contain a {suffix!r} value: {sorted(row)}")
    return matches[0]


def select_recall_first(rows: Iterable[Mapping[str, Any]], *, minimum_recall: float) -> dict[str, Any]:
    """Prefer maximum precision subject to recall, with F2 as a safe fallback."""

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("Calibration produced no threshold results")
    for row in candidates:
        precision = _metric(row, "precision(b)")
        recall = _metric(row, "recall(b)")
        row["f2"] = 5 * precision * recall / max(4 * precision + recall, 1e-12)
    feasible = [row for row in candidates if _metric(row, "recall(b)") >= minimum_recall]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (_metric(row, "precision(b)"), _metric(row, "recall(b)"), float(row["confidence"])),
        )
        reason = "maximum_precision_subject_to_minimum_recall"
    else:
        selected = max(candidates, key=lambda row: (float(row["f2"]), _metric(row, "recall(b)")))
        reason = "maximum_f2_no_threshold_met_minimum_recall"
    selected["selection_reason"] = reason
    selected["minimum_recall"] = float(minimum_recall)
    return selected


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    return intersection / max(left_area + right_area - intersection, 1e-12)


def _match_image(
    detections: Sequence[Detection],
    ground_truth: Sequence[Sequence[float]],
    *,
    confidence: float,
    match_iou: float,
) -> tuple[int, int, int]:
    kept = sorted(
        (item for item in detections if item.confidence >= confidence),
        key=lambda item: item.confidence,
        reverse=True,
    )
    unmatched = set(range(len(ground_truth)))
    true_positives = 0
    for detection in kept:
        candidates = [
            (_iou(detection.xyxy, ground_truth[index]), index) for index in unmatched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= match_iou:
            true_positives += 1
            unmatched.remove(best_index)
    false_positives = len(kept) - true_positives
    false_negatives = len(unmatched)
    return true_positives, false_positives, false_negatives


def sweep_thresholds(
    predictions: Mapping[str, Sequence[Detection]],
    ground_truth: Mapping[str, Sequence[Sequence[float]]],
    *,
    thresholds: Iterable[float],
    match_iou: float,
) -> list[dict[str, Any]]:
    """Compute threshold-specific TP/FP/FN without Ultralytics summary metrics."""

    if not 0 < match_iou <= 1:
        raise ValueError("match_iou must be in (0, 1]")
    image_keys = set(predictions) | set(ground_truth)
    rows: list[dict[str, Any]] = []
    for threshold in sorted({float(value) for value in thresholds}):
        if not 0 <= threshold <= 1:
            raise ValueError("confidence thresholds must be in [0, 1]")
        tp = fp = fn = 0
        for image_key in image_keys:
            image_tp, image_fp, image_fn = _match_image(
                predictions.get(image_key, ()),
                ground_truth.get(image_key, ()),
                confidence=threshold,
                match_iou=match_iou,
            )
            tp += image_tp
            fp += image_fp
            fn += image_fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(
            {
                "confidence": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "metrics/precision(B)": precision,
                "metrics/recall(B)": recall,
            }
        )
    return rows


def _label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_positions = [index for index, value in enumerate(parts) if value == "images"]
    if not image_positions:
        raise ValueError(f"Cannot infer YOLO label path outside an images/ tree: {image_path}")
    parts[image_positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _calibration_files(data_yaml: str | Path) -> list[tuple[Path, Path, str]]:
    source_path = Path(data_yaml)
    data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YOLO data YAML: {source_path}")
    calibration_value = _calibration_value(data, source_path)
    dataset_root = Path(str(data.get("path", source_path.parent)))
    if not dataset_root.is_absolute():
        dataset_root = (source_path.parent / dataset_root).resolve()
    configured = Path(calibration_value)
    source = configured if configured.is_absolute() else dataset_root / configured
    if source.is_dir():
        image_paths = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    elif source.is_file():
        image_paths = []
        for value in source.read_text(encoding="utf-8").splitlines():
            if not value.strip():
                continue
            image = Path(value.strip())
            if not image.is_absolute():
                image = dataset_root / image
            image_paths.append(image)
    else:
        raise FileNotFoundError(f"Calibration source does not exist: {source}")
    if not image_paths:
        raise ValueError(f"Calibration split has no images: {source}")
    result: list[tuple[Path, Path, str]] = []
    for image in image_paths:
        if not image.is_file():
            raise FileNotFoundError(f"Missing calibration image: {image}")
        label = _label_for_image(image)
        if not label.is_file():
            raise FileNotFoundError(f"Missing calibration label: {label}")
        result.append((image, label, image.as_posix()))
    return result


def _ground_truth_boxes(label_path: Path, width: int, height: int, class_id: int) -> list[tuple[float, ...]]:
    boxes: list[tuple[float, ...]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        label_class, x_center, y_center, box_width, box_height = map(float, values)
        if int(label_class) != class_id:
            continue
        boxes.append(
            (
                (x_center - box_width / 2) * width,
                (y_center - box_height / 2) * height,
                (x_center + box_width / 2) * width,
                (y_center + box_height / 2) * height,
            )
        )
    return boxes


def calibrate_confidence(
    *,
    model: YOLOModel,
    data_yaml: str | Path,
    thresholds: Iterable[float],
    minimum_recall: float,
    output_dir: str | Path,
    image_size: int = 1024,
    nms_iou: float = 0.7,
    match_iou: float = 0.5,
    inference_batch: int = 1,
    half: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    threshold_values = sorted({float(value) for value in thresholds})
    if not threshold_values:
        raise ValueError("confidence_grid must not be empty")
    if inference_batch < 1:
        raise ValueError("inference_batch must be positive")
    calibration_files = _calibration_files(data_yaml)
    image_paths = [item[0] for item in calibration_files]

    predictions: dict[str, list[Detection]] = {}
    ground_truth: dict[str, list[tuple[float, ...]]] = {}
    native = model._ensure_loaded()
    seen: set[str] = set()
    for image_path, label_path, image_key in calibration_files:
        # One source per call also works for txt-backed immutable views and
        # prevents older Ultralytics releases from creating one giant batch.
        results = native.predict(
            source=str(image_path), conf=min(threshold_values), iou=float(nms_iou),
            imgsz=int(image_size), batch=int(inference_batch), half=bool(half),
            device=model.device, verbose=False, stream=True,
        )
        result = next(iter(results))
        height, width = (int(value) for value in result.orig_shape)
        predictions[image_key] = model._parse_result(result, {model.diatom_class_id})
        ground_truth[image_key] = _ground_truth_boxes(
            label_path, width, height, model.diatom_class_id
        )
        seen.add(image_key)
    expected = {item[2] for item in calibration_files}
    if seen != expected:
        raise RuntimeError(f"Inference result mismatch: expected={len(expected)}, received={len(seen)}")

    rows = sweep_thresholds(
        predictions,
        ground_truth,
        thresholds=threshold_values,
        match_iou=match_iou,
    )
    selected = select_recall_first(rows, minimum_recall=minimum_recall)
    selected["match_iou"] = float(match_iou)
    selected["nms_iou"] = float(nms_iou)
    selected["calibration_images"] = len(image_paths)
    selected["calibration_objects"] = sum(len(value) for value in ground_truth.values())
    selected["inference_batch"] = int(inference_batch)
    selected["half"] = bool(half)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "selected_threshold.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selected
