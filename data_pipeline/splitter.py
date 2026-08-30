"""Create simple detector/classifier splits and a Gunduz transfer benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _score(seed: int, value: str) -> int:
    return int(hashlib.sha256(f"{seed}|{value}".encode()).hexdigest(), 16)


def _assign_fractional(
    values: list[str], fractions: Mapping[str, float], seed: int
) -> dict[str, str]:
    if not values:
        return {}
    total = sum(float(value) for value in fractions.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to one, got {total}")
    ordered = sorted(set(values), key=lambda value: (_score(seed, value), value))
    names = list(fractions)
    raw = [len(ordered) * float(fractions[name]) for name in names]
    counts = [int(value) for value in raw]
    for index in sorted(range(len(names)), key=lambda i: raw[i] - counts[i], reverse=True)[: len(ordered)-sum(counts)]:
        counts[index] += 1
    if len(ordered) >= len(names):
        for index, count in enumerate(counts):
            if count == 0:
                donor = max(range(len(counts)), key=counts.__getitem__)
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    result: dict[str, str] = {}
    offset = 0
    for name, count in zip(names, counts):
        for value in ordered[offset : offset + count]:
            result[value] = name
        offset += count
    return result


def _classifier_split(crops: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (_, _), group in crops.groupby(["genus", "species"], sort=True):
        group = group.sort_values("id_crop").copy()
        # Exact duplicates from different public archives are one split unit.
        ordered = sorted(group["content_sha256"].unique(), key=lambda value: (_score(seed, str(value)), str(value)))
        val_count = 0 if len(ordered) == 1 else max(1, min(len(ordered) - 1, round(len(ordered) * val_fraction)))
        validation = set(ordered[:val_count])
        group["split"] = group["content_sha256"].map(lambda value: "val" if value in validation else "train")
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else crops.assign(split="train")


def _gunduz_benchmark(
    crops: pd.DataFrame,
    test_image_ids: set[str],
    gallery_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = crops[crops["source"].eq("gunduz") & crops["image_id"].isin(test_image_ids)].copy()
    role_by_image = _assign_fractional(
        pool["image_id"].unique().tolist(),
        {"gallery": gallery_fraction, "test": 1.0 - gallery_fraction},
        seed,
    )
    pool["split"] = pool["image_id"].map(role_by_image)
    eligible: set[tuple[str, str]] = set()
    exclusions: list[dict[str, Any]] = []
    for (genus, species), group in pool.groupby(["genus", "species"], sort=True):
        roles = set(group["split"])
        if roles == {"gallery", "test"}:
            eligible.add((genus, species))
        else:
            exclusions.append({
                "genus": genus, "species": species,
                "reason": "class_absent_from_gallery_or_query",
                "images": group["image_id"].nunique(), "crops": len(group),
            })
    mask = pool.apply(lambda row: (row.genus, row.species) in eligible, axis=1)
    gallery = pool[mask & pool.split.eq("gallery")].copy()
    query = pool[mask & pool.split.eq("test")].copy()
    return gallery, query, pd.DataFrame(exclusions)


def build_splits(
    data_root: str | Path,
    *,
    seed: int = 42,
    detector_fractions: Mapping[str, float] | None = None,
    classifier_val_fraction: float = 0.15,
    benchmark_gallery_fraction: float = 0.30,
) -> dict[str, Any]:
    root = Path(data_root)
    dataset = root / "datasetDiatom"
    output = root / "splits"
    if output.exists():
        raise FileExistsError(f"Split output already exists: {output}")
    temporary = root / ".splits.building"
    if temporary.exists():
        raise FileExistsError(f"Inspect stale split directory: {temporary}")
    images = pd.read_csv(dataset / "manifests/images.csv", dtype=str, keep_default_na=False)
    objects = pd.read_csv(dataset / "manifests/objects.csv", dtype=str, keep_default_na=False)
    crops = pd.read_csv(dataset / "manifests/crops.csv", dtype=str, keep_default_na=False)
    if set(images.source) != {"gunduz"} or "nii" in set(crops.source.str.lower()):
        raise ValueError("Public split protocol requires Gunduz detector data and no NII")
    fractions = detector_fractions or {"train": 0.70, "val": 0.15, "test": 0.15}
    assignment = _assign_fractional(images.image_id.tolist(), fractions, seed)
    images["split"] = images.image_id.map(assignment)
    if images.split.isna().any():
        raise ValueError("Detector split assignment is incomplete")
    external = crops[crops.source.isin(["ude", "diatom1042", "siyue_pu"])].copy()
    classifier = _classifier_split(external, classifier_val_fraction, seed)
    duplicate_splits = classifier.groupby("content_sha256")["split"].nunique()
    if (duplicate_splits > 1).any():
        raise ValueError("Exact classifier duplicate crosses train/validation")
    train_taxa = classifier[classifier.split.eq("train")][["genus", "species"]].drop_duplicates()
    train_pairs = set(map(tuple, train_taxa.itertuples(index=False, name=None)))
    train_genera = set(train_taxa.genus)
    gallery, query, exclusions = _gunduz_benchmark(
        crops, set(images.loc[images.split.eq("test"), "image_id"]), benchmark_gallery_fraction, seed
    )
    if gallery.empty or query.empty:
        raise ValueError("Gunduz benchmark has no gallery/query classes; inspect split settings")
    query["protocol_target"] = query.apply(
        lambda row: "known" if (row.genus, row.species) in train_pairs
        else "unseen_species" if row.genus in train_genera else "unseen_genus",
        axis=1,
    )
    gallery["protocol_target"] = "support"
    if set(gallery.image_id) & set(query.image_id):
        raise ValueError("Gunduz gallery/query image leakage")
    if set(classifier.source) & {"gunduz", "nii"}:
        raise ValueError("Classifier optimization data contains a forbidden source")
    temporary.mkdir(parents=True)
    try:
        detector_dir = temporary / "detector"
        classifier_dir = temporary / "classifier"
        benchmark_dir = temporary / "benchmark" / "gunduz"
        detector_dir.mkdir(parents=True)
        classifier_dir.mkdir(parents=True)
        benchmark_dir.mkdir(parents=True)
        absolute_dataset = dataset.resolve()
        for split in fractions:
            subset = images[images.split.eq(split)].copy()
            subset.to_csv(detector_dir / f"{split}.csv", index=False, encoding="utf-8")
            paths = [str((absolute_dataset / value).resolve()) for value in subset.path]
            (detector_dir / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        # This deliberately small YAML needs no serializer and quotes Windows
        # drive paths safely for Ultralytics.
        def quoted(value: Path) -> str:
            return json.dumps(str(value.resolve()).replace("\\", "/"))

        (detector_dir / "data.yaml").write_text(
            "\n".join([
                f"path: {quoted(absolute_dataset)}",
                f"train: {quoted(output / 'detector' / 'train.txt')}",
                f"val: {quoted(output / 'detector' / 'val.txt')}",
                f"test: {quoted(output / 'detector' / 'test.txt')}",
                "names:", "  0: diatom", "",
            ]),
            encoding="utf-8",
        )
        classifier.to_csv(classifier_dir / "all.csv", index=False, encoding="utf-8")
        classifier[classifier.split.eq("train")].to_csv(classifier_dir / "train.csv", index=False, encoding="utf-8")
        classifier[classifier.split.eq("val")].to_csv(classifier_dir / "val.csv", index=False, encoding="utf-8")
        classifier[["genus", "species"]].drop_duplicates().sort_values(["genus", "species"]).to_csv(
            classifier_dir / "canonical_codec.csv", index=False, encoding="utf-8"
        )
        gallery.to_csv(benchmark_dir / "gallery.csv", index=False, encoding="utf-8")
        query.to_csv(benchmark_dir / "query.csv", index=False, encoding="utf-8")
        pd.concat([gallery, query], ignore_index=True).to_csv(
            benchmark_dir / "all.csv", index=False, encoding="utf-8"
        )
        pd.concat([gallery, query], ignore_index=True)[["genus", "species"]].drop_duplicates().sort_values(
            ["genus", "species"]
        ).to_csv(benchmark_dir / "canonical_codec.csv", index=False, encoding="utf-8")
        for target in ("known", "unseen_species", "unseen_genus"):
            query[query.protocol_target.eq(target)].to_csv(benchmark_dir / f"query_{target}.csv", index=False, encoding="utf-8")
        exclusions.to_csv(benchmark_dir / "excluded_classes.csv", index=False, encoding="utf-8")
        # Support gallery images are never E2E queries.
        test_images = images[images.image_id.isin(set(query.image_id))].copy()
        e2e_metadata = test_images.rename(columns={"image_id": "id_image"})
        e2e_metadata["image_path"] = e2e_metadata["path"].map(
            lambda value: str((absolute_dataset / value).resolve())
        )
        e2e_metadata["source_cohort"] = "gunduz"
        e2e_metadata[["id_image", "image_path", "source_cohort"]].to_csv(
            benchmark_dir / "e2e_metadata.csv", index=False, encoding="utf-8"
        )
        e2e_objects = objects[objects.image_id.isin(set(test_images.image_id))].rename(
            columns={"image_id": "id_image", "object_id": "id_object"}
        )
        e2e_objects[["id_image", "id_object", "genus", "species", "x1", "y1", "x2", "y2"]].to_csv(
            benchmark_dir / "e2e_taxonomy.csv", index=False, encoding="utf-8"
        )
        distribution = (
            pd.concat([classifier.assign(role="classifier"), gallery.assign(role="benchmark_gallery"), query.assign(role="benchmark_query")])
            .groupby(["role", "source", "genus", "species", "split"], dropna=False)
            .size().rename("count").reset_index()
        )
        distribution.to_csv(temporary / "distribution.csv", index=False, encoding="utf-8")
        audit = {
            "passed": True, "seed": seed, "nii_included": False,
            "detector_source": "gunduz", "classifier_sources": ["ude", "diatom1042", "siyue_pu"],
            "detector_counts": images.split.value_counts().sort_index().to_dict(),
            "classifier_counts": classifier.split.value_counts().sort_index().to_dict(),
            "benchmark_query_targets": query.protocol_target.value_counts().sort_index().to_dict(),
            "gallery_query_image_overlap": 0,
            "classifier_gunduz_rows": 0,
            "classifier_nii_rows": 0,
        }
        (temporary / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output)
        return {**audit, "path": str(output)}
    except BaseException:
        raise
