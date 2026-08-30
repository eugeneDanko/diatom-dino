"""Download public sources, build data/datasetDiatom, and create splits."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from core.config_loader import load_config
from data_pipeline.builder import build_dataset
from data_pipeline.sources import DEFAULT_SOURCES, download_archive
from data_pipeline.splitter import build_splits


def _settings(config: dict) -> tuple[Path, dict[str, Path], dict]:
    root = Path(str(config.get("data_root", "data")))
    raw = root / str(config.get("raw_archives_dir", "raw/archives"))
    configured = config.get("sources", {})
    archives: dict[str, Path] = {}
    source_settings: dict = {}
    for name, defaults in DEFAULT_SOURCES.items():
        value = {**defaults, **dict(configured.get(name, {}))}
        archives[name] = raw / str(value["archive"])
        source_settings[name] = value
    return root, archives, source_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "build", "split", "all"])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.overrides)
    root, archives, source_settings = _settings(config)
    if args.command in {"download", "all"}:
        for name, path in archives.items():
            if args.dry_run:
                print(f"{name}: {source_settings[name]['url']} -> {path}")
            else:
                print(f"Preparing {name}: {path}")
                download_archive(str(source_settings[name]["url"]), path)
    if args.command in {"build", "all"}:
        if args.dry_run:
            for name, path in archives.items():
                print(f"{name}: exists={path.is_file()}, valid_zip={zipfile.is_zipfile(path) if path.is_file() else False}")
            print(f"Would create: {root/'datasetDiatom'}")
        else:
            print(json.dumps(build_dataset(root, archives), indent=2, ensure_ascii=False))
    if args.command in {"split", "all"}:
        split_config = config.get("splits", {})
        if args.dry_run:
            print(f"Would create: {root/'splits'} with {json.dumps(split_config, ensure_ascii=False)}")
        else:
            print(json.dumps(build_splits(
                root,
                seed=int(config.get("seed", 42)),
                detector_fractions=split_config.get("detector"),
                classifier_val_fraction=float(split_config.get("classifier_validation", 0.15)),
                benchmark_gallery_fraction=float(split_config.get("benchmark_gallery", 0.30)),
            ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
