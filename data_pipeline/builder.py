"""Build the public datasetDiatom layout from immutable source archives."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pandas as pd
from PIL import Image

from .sources import IMAGE_EXTENSIONS, PARSERS, content_id, image_members, normalized_taxon


_AUGMENTED_STEM = re.compile(
    r"(?:[ _-](?:horizontal|vertical|flip(?:ped)?|rot(?:ate)?\d*|rotation\d*|"
    r"aug(?:mented)?\d*|copy\d*))+$",
    re.IGNORECASE,
)


def _required_text(parent: ET.Element, path: str, context: str) -> str:
    value = parent.findtext(path)
    if value is None or not value.strip():
        raise ValueError(f"Missing XML field {path!r}: {context}")
    return value.strip()


def _number(parent: ET.Element, paths: tuple[str, ...], context: str) -> float:
    for path in paths:
        value = parent.findtext(path)
        if value is not None and value.strip():
            try:
                return float(value)
            except ValueError as error:
                raise ValueError(f"Invalid XML number {path}={value!r}: {context}") from error
    raise ValueError(f"Missing XML number {paths}: {context}")


def _save_crop(image: Image.Image, box: tuple[float, float, float, float], target: Path) -> None:
    crop = image.crop(tuple(int(round(v)) for v in box)).convert("L").convert("RGB")
    target.parent.mkdir(parents=True, exist_ok=True)
    crop.save(target, format="PNG", optimize=False)


def _save_image(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target, format="PNG", optimize=False)


def _parse_gunduz(
    archive_path: Path,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    skipped_empty = 0
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            info for info in image_members(archive)
            if "images" in {part.lower() for part in PurePosixPath(info.filename).parts[:-1]}
        ]
        by_stem: dict[str, zipfile.ZipInfo] = {}
        duplicates: set[str] = set()
        for info in candidates:
            stem = PurePosixPath(info.filename).stem
            if stem in by_stem:
                duplicates.add(stem)
            by_stem[stem] = info
        if duplicates:
            raise ValueError(f"Duplicate Gunduz image stems: {sorted(duplicates)[:20]}")
        xml_infos = sorted(
            (i for i in archive.infolist() if PurePosixPath(i.filename).suffix.lower() == ".xml"),
            key=lambda i: i.filename,
        )
        if not xml_infos or not by_stem:
            raise ValueError("Gunduz archive must contain images/ and XML annotations")
        used_images: set[str] = set()
        for xml_info in xml_infos:
            context = xml_info.filename
            root = ET.fromstring(archive.read(xml_info))
            filename = _required_text(root, "filename", context)
            stem = PurePosixPath(filename).stem
            image_info = by_stem.get(stem)
            if image_info is None:
                raise FileNotFoundError(f"Gunduz image referenced by {context} is missing: {filename}")
            used_images.add(stem)
            elements = root.findall("objects/object") or root.findall("object")
            if not elements or any(not (e.findtext("name") or "").strip() for e in elements):
                skipped_empty += 1
                continue
            width = _number(root, ("size/width",), context)
            height = _number(root, ("size/height",), context)
            raw_image = archive.read(image_info)
            with Image.open(io.BytesIO(raw_image)) as opened:
                image = opened.convert("RGB")
            if image.width != int(round(width)) or image.height != int(round(height)):
                raise ValueError(f"XML/image dimensions differ for {context}")
            image_id = content_id("gunduz-image", image_info.filename)
            image_path = Path("detector/images") / f"{image_id}.png"
            label_path = Path("detector/labels") / f"{image_id}.txt"
            _save_image(image, output / image_path)
            yolo_lines: list[str] = []
            parsed_objects: list[tuple[str, str, tuple[float, float, float, float]]] = []
            for object_index, element in enumerate(elements, 1):
                raw_taxon = _required_text(element, "name", context).split(maxsplit=1)
                if len(raw_taxon) != 2:
                    raise ValueError(f"Cannot split Gunduz taxon in {context}: {raw_taxon}")
                genus, species = normalized_taxon(*raw_taxon)
                xmin = _number(element, ("bbox/xmin", "bndbox/xmin"), context)
                xmax = _number(element, ("bbox/xmax", "bndbox/xmax"), context)
                ymin = _number(element, ("bbox/ymin", "bndbox/ymin"), context)
                ymax = _number(element, ("bbox/ymax", "bndbox/ymax"), context)
                if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                    raise ValueError(f"BBox outside image in {context}: {(xmin, ymin, xmax, ymax)}")
                yolo_lines.append(
                    f"0 {((xmin+xmax)/2)/width:.10f} {((ymin+ymax)/2)/height:.10f} "
                    f"{(xmax-xmin)/width:.10f} {(ymax-ymin)/height:.10f}"
                )
                parsed_objects.append((genus, species, (xmin, ymin, xmax, ymax)))
            (output / label_path).parent.mkdir(parents=True, exist_ok=True)
            (output / label_path).write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            image_rows.append({
                "image_id": image_id, "source": "gunduz",
                "source_record_id": image_info.filename, "path": image_path.as_posix(),
                "label_path": label_path.as_posix(), "num_objects": len(parsed_objects),
                "width": image.width, "height": image.height,
                "content_sha256": hashlib.sha256(raw_image).hexdigest(),
            })
            for object_index, (genus, species, box) in enumerate(parsed_objects, 1):
                object_id = f"{image_id}:{object_index}"
                crop_id = content_id("gunduz-crop", object_id)
                crop_path = Path("classifier/crops/gunduz") / f"{crop_id}.png"
                _save_crop(image, box, output / crop_path)
                x1, y1, x2, y2 = box
                object_rows.append({
                    "object_id": object_id, "image_id": image_id, "source": "gunduz",
                    "genus": genus, "species": species, "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                })
                crop_rows.append({
                    "id_crop": crop_id, "image_id": image_id, "object_id": object_id,
                    "source": "gunduz", "genus": genus, "species": species,
                    "path": crop_path.as_posix(), "kind": "bbox_crop",
                    "content_sha256": hashlib.sha256((output / crop_path).read_bytes()).hexdigest(),
                })
    return image_rows, object_rows, crop_rows, {
        "source": "gunduz", "archive": str(archive_path), "images": len(image_rows),
        "objects": len(object_rows), "crops": len(crop_rows), "skipped_empty_xml": skipped_empty,
    }


def _parse_external(
    source: str,
    archive_path: Path,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parser = PARSERS[source]
    rows: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    seen_originals: set[str] = set()
    image_members_seen = unparsed = duplicate = corrupt = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in sorted(image_members(archive), key=lambda i: i.filename):
            taxon = parser(info.filename)
            if taxon is None:
                continue
            image_members_seen += 1
            if not taxon:
                unparsed += 1
                continue
            genus, species = taxon
            stem = _AUGMENTED_STEM.sub("", PurePosixPath(info.filename).stem).casefold()
            original_key = f"{source}|{genus}|{species}|{stem}"
            if original_key in seen_originals:
                duplicate += 1
                continue
            raw = archive.read(info)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen_content:
                duplicate += 1
                continue
            try:
                with Image.open(io.BytesIO(raw)) as opened:
                    image = opened.convert("L").convert("RGB")
                    image.load()
            except Exception:
                corrupt += 1
                continue
            seen_originals.add(original_key)
            seen_content.add(digest)
            crop_id = content_id(source, info.filename)
            crop_path = Path("classifier/crops") / source / f"{crop_id}.png"
            (output / crop_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output / crop_path, format="PNG", optimize=False)
            rows.append({
                "id_crop": crop_id, "image_id": "", "object_id": "", "source": source,
                "genus": genus, "species": species, "path": crop_path.as_posix(),
                "kind": "source_crop", "content_sha256": digest,
            })
    return rows, {
        "source": source, "archive": str(archive_path), "images_seen": image_members_seen,
        "crops": len(rows), "unparsed": unparsed, "duplicates_skipped": duplicate,
        "corrupt_skipped": corrupt,
    }


def _validate(output: Path, images: pd.DataFrame, objects: pd.DataFrame, crops: pd.DataFrame) -> None:
    if images.empty or objects.empty or crops.empty:
        raise ValueError("Built dataset contains an empty required table")
    if set(images["source"]) != {"gunduz"}:
        raise ValueError("Detector images must contain Gunduz only")
    if "nii" in set(crops["source"].str.lower()):
        raise ValueError("NII data is forbidden in the public training dataset")
    if images["image_id"].duplicated().any() or objects["object_id"].duplicated().any():
        raise ValueError("Duplicate image/object identifier")
    if crops["id_crop"].duplicated().any():
        raise ValueError("Duplicate crop identifier")
    conflicting_hashes = crops.groupby("content_sha256")[["genus", "species"]].nunique()
    if ((conflicting_hashes["genus"] > 1) | (conflicting_hashes["species"] > 1)).any():
        raise ValueError("Identical crop content has conflicting taxonomy")
    if set(objects["image_id"]) - set(images["image_id"]):
        raise ValueError("Object references an unknown Gunduz image")
    counts = objects.groupby("image_id").size().to_dict()
    for row in images.itertuples(index=False):
        if counts.get(row.image_id, 0) != int(row.num_objects):
            raise ValueError(f"Object count mismatch for {row.image_id}")
        for relative in (row.path, row.label_path):
            if not (output / relative).is_file():
                raise FileNotFoundError(output / relative)
    missing_crops = [path for path in crops["path"] if not (output / path).is_file()]
    if missing_crops:
        raise FileNotFoundError(f"Missing materialized crops: {missing_crops[:5]}")


def build_dataset(
    data_root: str | Path,
    archives: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Create ``data/datasetDiatom`` atomically; never append or overwrite."""

    root = Path(data_root)
    target = root / "datasetDiatom"
    temporary = root / ".datasetDiatom.building"
    if target.exists():
        raise FileExistsError(f"Dataset already exists: {target}")
    if temporary.exists():
        raise FileExistsError(f"Inspect stale build directory: {temporary}")
    required = {"gunduz", "ude", "diatom1042", "siyue_pu"}
    if set(archives) != required:
        raise ValueError(f"Exactly these public sources are required: {sorted(required)}")
    for source, path in archives.items():
        if not zipfile.is_zipfile(Path(path)):
            raise FileNotFoundError(f"{source} archive is missing or invalid: {path}")
    temporary.mkdir(parents=True)
    try:
        image_rows, object_rows, crop_rows, audit = _parse_gunduz(
            Path(archives["gunduz"]), temporary
        )
        audits = [audit]
        for source in ("ude", "diatom1042", "siyue_pu"):
            rows, source_audit = _parse_external(source, Path(archives[source]), temporary)
            crop_rows.extend(rows)
            audits.append(source_audit)
        images = pd.DataFrame(image_rows)
        objects = pd.DataFrame(object_rows)
        crops = pd.DataFrame(crop_rows)
        _validate(temporary, images, objects, crops)
        manifests = temporary / "manifests"
        manifests.mkdir()
        images.to_csv(manifests / "images.csv", index=False, encoding="utf-8")
        objects.to_csv(manifests / "objects.csv", index=False, encoding="utf-8")
        crops.to_csv(manifests / "crops.csv", index=False, encoding="utf-8")
        pd.DataFrame(audits).to_csv(manifests / "source_audit.csv", index=False, encoding="utf-8")
        summary = {
            "schema_version": 1, "sources": sorted(required), "nii_included": False,
            "detector_images": len(images), "detector_objects": len(objects),
            "classifier_crops": len(crops),
            "classifier_train_sources": ["ude", "diatom1042", "siyue_pu"],
            "benchmark_source": "gunduz",
        }
        (temporary / "build_manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
        return {**summary, "path": str(target)}
    except BaseException:
        # Preserve the partial build for explicit inspection; never expose it as final.
        raise
