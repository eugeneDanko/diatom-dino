"""Exact cosine retrieval with optional FAISS acceleration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Candidate:
    item_id: str
    similarity: float
    genus: str
    species: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


class FAISSRetriever:
    """Store immutable gallery metadata next to its embedding matrix.

    FAISS is used when installed. The retained NumPy matrix provides an exact
    fallback and enables genus-filtered species search without a second library.
    """

    def __init__(self, dimension: int | None = None, *, use_faiss: bool = True) -> None:
        self.dimension = dimension
        self.use_faiss = bool(use_faiss)
        self.embeddings: np.ndarray | None = None
        self.items: list[dict[str, str]] = []
        self.index: Any | None = None

    def build_index(
        self,
        embeddings: np.ndarray,
        *,
        item_ids: Sequence[str],
        genera: Sequence[str],
        species: Sequence[str],
    ) -> None:
        vectors = _normalize(embeddings)
        count, dimension = vectors.shape
        if not (len(item_ids) == len(genera) == len(species) == count):
            raise ValueError("Embeddings and metadata lengths differ")
        if count == 0:
            raise ValueError("Cannot build an empty gallery index")
        if len({str(item_id) for item_id in item_ids}) != len(item_ids):
            raise ValueError("Gallery item_ids must be unique")
        if self.dimension is not None and self.dimension != dimension:
            raise ValueError(f"Expected dimension {self.dimension}, received {dimension}")
        self.dimension = dimension
        self.embeddings = np.ascontiguousarray(vectors, dtype=np.float32)
        self.items = [
            {"item_id": str(item_id), "genus": str(genus), "species": str(species_name)}
            for item_id, genus, species_name in zip(item_ids, genera, species)
        ]
        self.index = None
        if self.use_faiss:
            try:
                import faiss
            except ImportError:
                return
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

    def _check_ready(self) -> None:
        if self.embeddings is None or not self.items:
            raise RuntimeError("Gallery index has not been built or loaded")

    def search(
        self,
        query_embeddings: np.ndarray,
        *,
        top_k: int = 5,
        genus_filter: str | None = None,
    ) -> list[list[Candidate]]:
        self._check_ready()
        queries = _normalize(query_embeddings)
        top_k = max(1, int(top_k))
        if genus_filter is None and self.index is not None:
            scores, indices = self.index.search(queries, min(top_k, len(self.items)))
            return [
                [self._candidate(int(index), float(score)) for index, score in zip(row_indices, row_scores)]
                for row_indices, row_scores in zip(indices, scores)
            ]

        allowed = np.arange(len(self.items))
        if genus_filter is not None:
            allowed = np.asarray(
                [index for index, item in enumerate(self.items) if item["genus"] == genus_filter],
                dtype=np.int64,
            )
        if len(allowed) == 0:
            return [[] for _ in range(len(queries))]
        similarities = queries @ self.embeddings[allowed].T
        effective_k = min(top_k, len(allowed))
        local_indices = np.argpartition(-similarities, effective_k - 1, axis=1)[:, :effective_k]
        results: list[list[Candidate]] = []
        for row_index, local_row in enumerate(local_indices):
            local_row = local_row[np.argsort(-similarities[row_index, local_row])]
            results.append(
                [
                    self._candidate(int(allowed[local]), float(similarities[row_index, local]))
                    for local in local_row
                ]
            )
        return results

    def _candidate(self, index: int, score: float) -> Candidate:
        item = self.items[index]
        return Candidate(
            item_id=item["item_id"],
            similarity=score,
            genus=item["genus"],
            species=item["species"],
        )

    def save_index(self, path: str | Path) -> Path:
        self._check_ready()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, embeddings=self.embeddings)
        metadata_path = output.with_suffix(output.suffix + ".json")
        metadata_path.write_text(
            json.dumps(
                {"dimension": self.dimension, "items": self.items},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output

    def load_index(self, path: str | Path) -> "FAISSRetriever":
        source = Path(path)
        with np.load(source) as archive:
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        metadata = json.loads(source.with_suffix(source.suffix + ".json").read_text(encoding="utf-8"))
        items = metadata["items"]
        self.build_index(
            embeddings,
            item_ids=[item["item_id"] for item in items],
            genera=[item["genus"] for item in items],
            species=[item["species"] for item in items],
        )
        return self
