"""Explicit, unit-testable genus-then-species open-set decisions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .faiss_retriever import Candidate


@dataclass(frozen=True)
class DecisionThresholds:
    genus_similarity: float = 0.70
    genus_margin: float = 0.03
    genus_agreement: float = 0.50
    species_similarity: float = 0.75
    species_margin: float = 0.03
    species_agreement: float = 0.50

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionThresholds":
        fields = cls.__dataclass_fields__
        return cls(**{key: float(item) for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class LevelDecision:
    accepted: bool
    label: str | None
    similarity: float
    margin: float
    agreement: float


@dataclass(frozen=True)
class ClassificationDecision:
    status: str
    genus: str | None
    species: str | None
    genus_similarity: float
    species_similarity: float | None
    genus_margin: float
    species_margin: float | None
    genus_agreement: float
    species_agreement: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionLogic:
    def __init__(
        self,
        thresholds: DecisionThresholds,
        *,
        top_k: int = 5,
        open_set: bool = True,
    ) -> None:
        self.thresholds = thresholds
        self.top_k = int(top_k)
        self.open_set = bool(open_set)

    @staticmethod
    def _decide_level(
        candidates: Sequence[Candidate],
        *,
        label_attribute: str,
        similarity_threshold: float,
        margin_threshold: float,
        agreement_threshold: float,
        top_k: int,
    ) -> LevelDecision:
        candidates = list(candidates[:top_k])
        if not candidates:
            # Cosine similarity cannot be below -1, which also keeps JSON and
            # calibration arrays finite.
            return LevelDecision(False, None, -1.0, 0.0, 0.0)
        # Class score is the best exemplar score. This avoids gallery-size bias.
        best_by_label: dict[str, float] = {}
        for candidate in candidates:
            label = str(getattr(candidate, label_attribute))
            best_by_label[label] = max(best_by_label.get(label, float("-inf")), candidate.similarity)
        ordered_labels = sorted(best_by_label, key=best_by_label.get, reverse=True)
        label = ordered_labels[0]
        top_score = best_by_label[label]
        has_competing_label = len(ordered_labels) > 1
        second_score = best_by_label[ordered_labels[1]] if has_competing_label else top_score
        # No competing label means that the margin is unobserved, not nearly 2.0.
        # Keep the value finite and neutral while allowing similarity/agreement
        # to decide this otherwise unanimous neighbourhood.
        margin = top_score - second_score if has_competing_label else 0.0
        counts = Counter(str(getattr(candidate, label_attribute)) for candidate in candidates)
        agreement = counts[label] / len(candidates)
        accepted = (
            top_score >= similarity_threshold
            and (not has_competing_label or margin >= margin_threshold)
            and agreement >= agreement_threshold
        )
        return LevelDecision(accepted, label, top_score, margin, agreement)

    def decide_genus(self, candidates: Sequence[Candidate]) -> LevelDecision:
        if not self.open_set:
            return self._decide_level(
                candidates,
                label_attribute="genus",
                similarity_threshold=-1.0,
                margin_threshold=-2.0,
                agreement_threshold=0.0,
                top_k=self.top_k,
            )
        return self._decide_level(
            candidates,
            label_attribute="genus",
            similarity_threshold=self.thresholds.genus_similarity,
            margin_threshold=self.thresholds.genus_margin,
            agreement_threshold=self.thresholds.genus_agreement,
            top_k=self.top_k,
        )

    def decide_species(self, candidates: Sequence[Candidate]) -> LevelDecision:
        if not self.open_set:
            return self._decide_level(
                candidates,
                label_attribute="species",
                similarity_threshold=-1.0,
                margin_threshold=-2.0,
                agreement_threshold=0.0,
                top_k=self.top_k,
            )
        return self._decide_level(
            candidates,
            label_attribute="species",
            similarity_threshold=self.thresholds.species_similarity,
            margin_threshold=self.thresholds.species_margin,
            agreement_threshold=self.thresholds.species_agreement,
            top_k=self.top_k,
        )

    def combine(self, genus: LevelDecision, species: LevelDecision | None) -> ClassificationDecision:
        if not genus.accepted:
            return ClassificationDecision(
                status="unknown_genus",
                genus=None,
                species=None,
                genus_similarity=genus.similarity,
                species_similarity=None,
                genus_margin=genus.margin,
                species_margin=None,
                genus_agreement=genus.agreement,
                species_agreement=None,
            )
        if species is None or not species.accepted:
            return ClassificationDecision(
                status="known_genus_unknown_species",
                genus=genus.label,
                species=None,
                genus_similarity=genus.similarity,
                species_similarity=None if species is None else species.similarity,
                genus_margin=genus.margin,
                species_margin=None if species is None else species.margin,
                genus_agreement=genus.agreement,
                species_agreement=None if species is None else species.agreement,
            )
        return ClassificationDecision(
            status="known_species",
            genus=genus.label,
            species=species.label,
            genus_similarity=genus.similarity,
            species_similarity=species.similarity,
            genus_margin=genus.margin,
            species_margin=species.margin,
            genus_agreement=genus.agreement,
            species_agreement=species.agreement,
        )
