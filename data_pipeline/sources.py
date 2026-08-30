"""Source definitions, downloads, and label parsers for public datasets."""

from __future__ import annotations

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_SOURCES = {
    "gunduz": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/huseyingunduz/diatom-dataset",
        "archive": "gunduz-diatom-dataset.zip",
    },
    "ude": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/michaelkloster/ude-diatoms-in-the-wild-2024",
        "archive": "ude-diatoms-in-the-wild-2024.zip",
    },
    "diatom1042": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/suziwsz/diatom1042",
        "archive": "diatom1042.zip",
    },
    "siyue_pu": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/siyuepu/diatom-datasets",
        "archive": "diatom-datasets-siyue-pu.zip",
    },
}

UDE_SUBSET = "UDE Diatoms in the Wild 2024-Subset_n25"
_TWO_LATIN_WORDS = re.compile(r"^([A-Za-z]+)[ _]+([A-Za-z]+)(?=[^A-Za-z]|$)")


def normalized_taxon(genus: str, species: str) -> tuple[str, str]:
    genus = re.sub(r"\s+", " ", genus).strip()
    species = re.sub(r"\s+", " ", species).strip()
    if not genus or not species:
        raise ValueError("Taxonomy requires non-empty genus and species")
    return genus[:1].upper() + genus[1:].lower(), species.lower()


def _two_words(value: str) -> tuple[str, str] | None:
    match = _TWO_LATIN_WORDS.match(value)
    return normalized_taxon(*match.groups()) if match else None


def parse_ude(member: str) -> tuple[str, str] | None:
    path = PurePosixPath(member)
    if path.suffix.lower() not in IMAGE_EXTENSIONS or UDE_SUBSET not in path.parts:
        return None
    index = path.parts.index(UDE_SUBSET)
    directories = [p for p in path.parts[index + 1 : -1] if p.lower() not in {
        "data", "dataset", "image", "images", "train", "training", "val",
        "validation", "test",
    }]
    if not directories:
        return None
    direct = _two_words(directories[-1].replace("_", " "))
    if direct:
        return direct
    if len(directories) >= 2:
        genus = re.findall(r"[A-Za-z]+", directories[-2])
        species = re.findall(r"[A-Za-z]+", directories[-1])
        if len(genus) == len(species) == 1:
            return normalized_taxon(genus[0], species[0])
    return None


def parse_filename_taxon(member: str) -> tuple[str, str] | None:
    path = PurePosixPath(member)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return _two_words(path.stem.replace("_", " "))


PARSERS: dict[str, Callable[[str], tuple[str, str] | None]] = {
    "ude": parse_ude,
    "diatom1042": parse_filename_taxon,
    "siyue_pu": parse_filename_taxon,
}


def download_archive(url: str, destination: Path) -> None:
    """Download one immutable ZIP atomically and validate the response."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if not zipfile.is_zipfile(destination):
            raise ValueError(f"Existing source is not a ZIP archive: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Inspect or remove stale partial download: {partial}")
    request = urllib.request.Request(url, headers={"User-Agent": "DiatomDINO/1.0"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        if not zipfile.is_zipfile(partial):
            raise ValueError(f"Downloaded response is not ZIP: {url}")
        partial.replace(destination)
    except BaseException:
        # A partial file is intentionally retained for inspection.
        raise


def content_id(source: str, member: str, *, length: int = 24) -> str:
    digest = hashlib.sha256(f"{source}|{member}".encode("utf-8")).hexdigest()
    return f"{source}_{digest[:length]}"


def image_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    return (
        info for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() in IMAGE_EXTENSIONS
    )
