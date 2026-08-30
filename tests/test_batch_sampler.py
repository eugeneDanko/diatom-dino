from __future__ import annotations

import unittest
from collections import Counter

from classifier.batch_sampler import HierarchicalBatchSampler


class HierarchicalBatchSamplerTest(unittest.TestCase):
    def test_batch_contains_positive_species_pairs(self) -> None:
        genera = [0, 0, 0, 0, 1, 1, 1, 1]
        species = [0, 0, 1, 1, 2, 2, 3, 3]
        sampler = HierarchicalBatchSampler(
            genera,
            species,
            genera_per_batch=2,
            species_per_genus=2,
            samples_per_species=2,
            batches_per_epoch=1,
            seed=1,
        )
        batch = next(iter(sampler))
        counts = Counter(species[index] for index in batch)
        self.assertEqual(len(batch), 8)
        self.assertTrue(all(count >= 2 for count in counts.values()))

    def test_rare_species_is_sampled_with_replacement(self) -> None:
        sampler = HierarchicalBatchSampler(
            [0],
            [0],
            genera_per_batch=1,
            species_per_genus=1,
            samples_per_species=2,
            batches_per_epoch=1,
        )
        self.assertEqual(next(iter(sampler)), [0, 0])

    def test_power_sampling_favors_more_powerful_species(self) -> None:
        species = [0] * 100 + [1]
        sampler = HierarchicalBatchSampler(
            [0] * len(species),
            species,
            genera_per_batch=1,
            species_per_genus=1,
            samples_per_species=2,
            batches_per_epoch=500,
            seed=7,
            class_sampling="power",
            sampling_power=1.0,
        )
        selected = Counter(species[batch[0]] for batch in sampler)
        self.assertGreater(selected[0], selected[1] * 20)

    def test_power_sampling_is_deterministic_per_epoch(self) -> None:
        kwargs = dict(
            genera_per_batch=1,
            species_per_genus=1,
            samples_per_species=2,
            batches_per_epoch=5,
            seed=9,
            class_sampling="power",
        )
        first = HierarchicalBatchSampler([0] * 6, [0, 0, 0, 0, 1, 1], **kwargs)
        second = HierarchicalBatchSampler([0] * 6, [0, 0, 0, 0, 1, 1], **kwargs)
        self.assertEqual(list(first), list(second))

    def test_hard_negative_group_is_forced_into_batch(self) -> None:
        genera = [0, 0, 0, 0, 0, 0, 1, 1]
        species = [0, 0, 1, 1, 4, 4, 2, 2]
        sampler = HierarchicalBatchSampler(
            genera,
            species,
            genera_per_batch=1,
            species_per_genus=2,
            samples_per_species=2,
            batches_per_epoch=5,
            hard_negative_groups=[[0, 1]],
            hard_negative_probability=1.0,
            seed=3,
        )
        for batch in sampler:
            selected = {species[index] for index in batch}
            self.assertEqual(selected, {0, 1})

    def test_hard_negative_group_rejects_different_genera(self) -> None:
        with self.assertRaisesRegex(ValueError, "one genus"):
            HierarchicalBatchSampler(
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                genera_per_batch=1,
                species_per_genus=2,
                samples_per_species=2,
                hard_negative_groups=[[0, 1]],
                hard_negative_probability=1.0,
            )


if __name__ == "__main__":
    unittest.main()
