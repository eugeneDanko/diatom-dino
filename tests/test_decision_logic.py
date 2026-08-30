from __future__ import annotations

import unittest

from inference.decision_logic import DecisionLogic, DecisionThresholds
from inference.faiss_retriever import Candidate


def candidate(score: float, genus: str, species: str) -> Candidate:
    return Candidate(item_id=f"{genus}-{species}-{score}", similarity=score, genus=genus, species=species)


class DecisionLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logic = DecisionLogic(
            DecisionThresholds(
                genus_similarity=0.7,
                genus_margin=0.05,
                genus_agreement=0.5,
                species_similarity=0.75,
                species_margin=0.03,
                species_agreement=0.5,
            ),
            top_k=3,
        )

    def test_known_species(self) -> None:
        genus = self.logic.decide_genus(
            [candidate(0.90, "Navicula", "N_a"), candidate(0.86, "Navicula", "N_b"), candidate(0.70, "G", "G_a")]
        )
        species = self.logic.decide_species(
            [candidate(0.91, "Navicula", "Navicula_a"), candidate(0.88, "Navicula", "Navicula_a"), candidate(0.80, "Navicula", "Navicula_b")]
        )
        decision = self.logic.combine(genus, species)
        self.assertEqual(decision.status, "known_species")
        self.assertEqual(decision.genus, "Navicula")
        self.assertEqual(decision.species, "Navicula_a")

    def test_low_similarity_returns_unknown_genus(self) -> None:
        genus = self.logic.decide_genus([candidate(0.5, "Navicula", "Navicula_a")])
        decision = self.logic.combine(genus, None)
        self.assertEqual(decision.status, "unknown_genus")

    def test_ambiguous_species_returns_genus_only(self) -> None:
        genus = self.logic.decide_genus([candidate(0.9, "N", "N_a"), candidate(0.88, "N", "N_b")])
        species = self.logic.decide_species([candidate(0.9, "N", "N_a"), candidate(0.89, "N", "N_b")])
        decision = self.logic.combine(genus, species)
        self.assertEqual(decision.status, "known_genus_unknown_species")

    def test_single_unique_label_has_neutral_margin_without_forced_rejection(self) -> None:
        decision = self.logic.decide_genus([
            candidate(0.9, "N", "N_a"), candidate(0.85, "N", "N_b")
        ])
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.margin, 0.0)


if __name__ == "__main__":
    unittest.main()
