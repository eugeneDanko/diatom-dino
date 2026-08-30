"""CSV/Parquet-driven crop dataset; no directory crawling is required."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class LabelCodec:
    genus_to_id: dict[str, int]
    species_to_id: dict[str, int]

    @classmethod
    def from_table(
        cls,
        table: pd.DataFrame,
        *,
        genus_column: str = "genus",
        species_column: str = "species",
    ) -> "LabelCodec":
        genera = sorted(str(value) for value in table[genus_column].dropna().unique())
        species_keys = sorted(
            {
                cls.species_key(str(row[genus_column]), str(row[species_column]))
                for _, row in table.dropna(subset=[genus_column, species_column]).iterrows()
            }
        )
        return cls(
            genus_to_id={name: index for index, name in enumerate(genera)},
            species_to_id={name: index for index, name in enumerate(species_keys)},
        )

    @staticmethod
    def species_key(genus: str, species: str) -> str:
        species = str(species).strip()
        genus = str(genus).strip()
        prefix = f"{genus}_"
        return species if species.startswith(prefix) else f"{genus}_{species}"

    @property
    def id_to_genus(self) -> dict[int, str]:
        return {value: key for key, value in self.genus_to_id.items()}

    @property
    def id_to_species(self) -> dict[int, str]:
        return {value: key for key, value in self.species_to_id.items()}

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "genus_to_id": self.genus_to_id,
            "species_to_id": self.species_to_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LabelCodec":
        return cls(
            genus_to_id={str(key): int(item) for key, item in value["genus_to_id"].items()},
            species_to_id={str(key): int(item) for key, item in value["species_to_id"].items()},
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


class DiatomCropDataset:
    """PyTorch-compatible dataset driven entirely by a metadata table.

    A row may provide ``path``/``relative_path`` directly. Otherwise the path is
    deterministically constructed as ``root/split/class_dir/filename``.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        table_path: str | Path,
        split: str | Sequence[str],
        transform: Callable[[Image.Image], Any] | None = None,
        label_codec: LabelCodec | None = None,
        id_column: str = "id_crop",
        genus_column: str = "genus",
        species_column: str = "species",
        split_column: str = "split",
        path_column: str = "path",
        filename_column: str = "filename",
        class_dir_template: str = "{species_key}",
        default_extension: str = ".png",
        power_groups: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.id_column = id_column
        self.genus_column = genus_column
        self.species_column = species_column
        self.split_column = split_column
        self.path_column = path_column
        self.filename_column = filename_column
        self.class_dir_template = class_dir_template
        self.default_extension = default_extension

        table = read_table(table_path)
        required = {id_column, genus_column, species_column, split_column}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Crop table is missing columns: {sorted(missing)}")
        splits = {split} if isinstance(split, str) else set(split)
        table = table[table[split_column].astype(str).isin(splits)].copy()
        if power_groups is not None:
            if "power_group" not in table.columns:
                raise ValueError("Crop table is missing column required by power_groups: power_group")
            allowed_power_groups = {str(value) for value in power_groups}
            table = table[table["power_group"].astype(str).isin(allowed_power_groups)].copy()
        table = table.dropna(subset=[genus_column, species_column]).reset_index(drop=True)
        if table.empty:
            raise ValueError(f"No rows found for split(s): {sorted(splits)}")

        self.table = table
        self.label_codec = label_codec or LabelCodec.from_table(table)
        self.genus_labels: list[int] = []
        self.species_labels: list[int] = []
        for _, row in table.iterrows():
            genus = str(row[genus_column])
            species = str(row[species_column])
            species_key = LabelCodec.species_key(genus, species)
            if genus not in self.label_codec.genus_to_id:
                raise ValueError(f"Unknown genus not present in label codec: {genus}")
            if species_key not in self.label_codec.species_to_id:
                raise ValueError(f"Unknown species not present in label codec: {species_key}")
            self.genus_labels.append(self.label_codec.genus_to_id[genus])
            self.species_labels.append(self.label_codec.species_to_id[species_key])

    def __len__(self) -> int:
        return len(self.table)

    def _resolve_path(self, row: pd.Series) -> Path:
        if self.path_column in row and pd.notna(row[self.path_column]):
            candidate = Path(str(row[self.path_column]))
            return candidate if candidate.is_absolute() else self.root / candidate
        split = str(row[self.split_column])
        genus = str(row[self.genus_column])
        species = str(row[self.species_column])
        species_key = LabelCodec.species_key(genus, species)
        if self.filename_column in row and pd.notna(row[self.filename_column]):
            filename = str(row[self.filename_column])
        else:
            raw_id = str(row[self.id_column])
            filename = raw_id if Path(raw_id).suffix else f"{raw_id}{self.default_extension}"
        class_dir = self.class_dir_template.format(
            genus=genus,
            species=species,
            species_key=species_key,
        )
        return self.root / split / class_dir / filename

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        path = self._resolve_path(row)
        if not path.exists():
            raise FileNotFoundError(f"Crop file listed in metadata does not exist: {path}")
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "genus_label": self.genus_labels[index],
            "species_label": self.species_labels[index],
            "crop_id": str(row[self.id_column]),
            "path": str(path),
            "genus": str(row[self.genus_column]),
            "species": LabelCodec.species_key(str(row[self.genus_column]), str(row[self.species_column])),
            **({"power_group": str(row["power_group"])} if "power_group" in row else {}),
            **({"training_role": str(row["training_role"])} if "training_role" in row else {}),
        }


def build_dataloader(
    dataset: DiatomCropDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    batch_sampler: Any | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Any:
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to build a DataLoader") from exc
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(num_workers > 0),
    }
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs.update(batch_size=int(batch_size), shuffle=bool(shuffle))
    return DataLoader(**kwargs)
