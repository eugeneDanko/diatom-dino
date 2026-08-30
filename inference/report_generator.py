"""Dependency-free HTML/JSON inference report."""

from __future__ import annotations

import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


class ReportGenerator:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        predictions: Mapping[str, list[Mapping[str, Any]]],
        *,
        metrics: Mapping[str, Any] | None = None,
        name: str = "report",
    ) -> Path:
        json_path = self.output_dir / f"{name}.json"
        json_path.write_text(
            json.dumps({"metrics": metrics or {}, "predictions": predictions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        statuses = Counter()
        rows: list[str] = []
        for image_path, image_predictions in predictions.items():
            for index, prediction in enumerate(image_predictions):
                decision = prediction.get("decision", {})
                status = str(decision.get("status", "unknown"))
                statuses[status] += 1
                genus = decision.get("genus") or "Unknown"
                species = decision.get("species") or "Unknown"
                species_similarity = decision.get("species_similarity")
                species_similarity_text = (
                    "" if species_similarity is None else f"{float(species_similarity):.3f}"
                )
                crop_path = prediction.get("crop_path")
                crop_html = ""
                if crop_path:
                    try:
                        relative_crop = os.path.relpath(str(crop_path), self.output_dir).replace("\\", "/")
                    except ValueError:
                        relative_crop = Path(str(crop_path)).as_uri()
                    crop_html = (
                        f'<img src="{html.escape(relative_crop)}" alt="crop" '
                        'style="width:96px;height:96px;object-fit:contain">'
                    )
                genus_candidates = prediction.get("genus_candidates", [])
                species_candidates = prediction.get("species_candidates", [])
                candidate_lines = [
                    f"G: {item.get('genus')} ({float(item.get('similarity', 0.0)):.3f})"
                    for item in genus_candidates
                ] + [
                    f"S: {item.get('species')} ({float(item.get('similarity', 0.0)):.3f})"
                    for item in species_candidates
                ]
                candidate_html = "<br>".join(html.escape(line) for line in candidate_lines)
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(Path(image_path).name)}</td><td>{crop_html}</td>"
                    f"<td>{index}</td><td>{html.escape(status)}</td>"
                    f"<td>{html.escape(str(genus))}</td><td>{html.escape(str(species))}</td>"
                    f"<td>{float(prediction.get('detection_confidence', 0.0)):.3f}</td>"
                    f"<td>{float(decision.get('genus_similarity', 0.0)):.3f}</td>"
                    f"<td>{species_similarity_text}</td>"
                    f"<td>{candidate_html}</td>"
                    "</tr>"
                )
        metric_items = "".join(
            f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
            for key, value in (metrics or {}).items()
        )
        status_items = "".join(
            f"<li><strong>{html.escape(key)}</strong>: {value}</li>" for key, value in statuses.items()
        )
        document = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>DiatomDINO report</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#eef3f8}}tr:nth-child(even){{background:#fafafa}}</style>
</head><body><h1>DiatomDINO report</h1><h2>Metrics</h2><ul>{metric_items}</ul>
<h2>Prediction status</h2><ul>{status_items}</ul><h2>Objects</h2>
<table><thead><tr><th>Image</th><th>Crop</th><th>#</th><th>Status</th><th>Genus</th><th>Species</th><th>Detector</th><th>Genus sim.</th><th>Species sim.</th><th>Top-k candidates</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
        html_path = self.output_dir / f"{name}.html"
        html_path.write_text(document, encoding="utf-8")
        return html_path
