"""Hierarchical PK sampler ensuring useful positive pairs for SupCon."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence


class HierarchicalBatchSampler:
    """Sample genera, species per genus, and images per species.

    Sampling with replacement is deliberate for rare classes; every selected
    species still contributes at least two views/objects to the loss.
    """

    def __init__(
        self,
        genus_labels: Sequence[int],
        species_labels: Sequence[int],
        *,
        genera_per_batch: int,
        species_per_genus: int,
        samples_per_species: int,
        batches_per_epoch: int | None = None,
        seed: int = 42,
        class_sampling: str = "uniform",
        sampling_power: float = 0.5,
        hard_negative_groups: Sequence[Sequence[int]] | None = None,
        hard_negative_probability: float = 0.0,
    ) -> None:
        if len(genus_labels) != len(species_labels):
            raise ValueError("genus_labels and species_labels must have equal length")
        if min(genera_per_batch, species_per_genus, samples_per_species) < 1:
            raise ValueError("Sampler dimensions must be positive")
        if samples_per_species < 2:
            raise ValueError("SupCon requires samples_per_species >= 2")
        self.genera_per_batch = int(genera_per_batch)
        self.species_per_genus = int(species_per_genus)
        self.samples_per_species = int(samples_per_species)
        self.seed = int(seed)
        self.class_sampling = str(class_sampling).strip().lower()
        self.sampling_power = float(sampling_power)
        if self.class_sampling not in {"uniform", "power"}:
            raise ValueError("class_sampling must be 'uniform' or 'power'")
        if self.sampling_power < 0:
            raise ValueError("sampling_power must be non-negative")
        self.hard_negative_probability = float(hard_negative_probability)
        if not 0 <= self.hard_negative_probability <= 1:
            raise ValueError("hard_negative_probability must be in [0, 1]")
        if self.hard_negative_probability > 0 and self.species_per_genus < 2:
            raise ValueError("Hard-negative sampling requires species_per_genus >= 2")
        self.epoch = 0
        self.index: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for item_index, (genus, species) in enumerate(zip(genus_labels, species_labels)):
            self.index[int(genus)][int(species)].append(item_index)
        if not self.index:
            raise ValueError("Cannot build a sampler for an empty dataset")
        species_to_genus = {
            species: genus for genus, species_index in self.index.items() for species in species_index
        }
        self.hard_negative_groups: list[tuple[int, list[int]]] = []
        for raw_group in hard_negative_groups or []:
            group = list(dict.fromkeys(int(value) for value in raw_group))
            missing = [species for species in group if species not in species_to_genus]
            if missing:
                raise ValueError(f"Hard-negative species IDs absent from train: {missing}")
            if len(group) < 2:
                raise ValueError("Every hard-negative group must contain at least two species")
            genera = {species_to_genus[species] for species in group}
            if len(genera) != 1:
                raise ValueError("A hard-negative group must contain species of one genus")
            self.hard_negative_groups.append((next(iter(genera)), group))
        if self.hard_negative_probability > 0 and not self.hard_negative_groups:
            raise ValueError("hard_negative_probability requires at least one valid group")
        self.batch_size = self.genera_per_batch * self.species_per_genus * self.samples_per_species
        self.batches_per_epoch = batches_per_epoch or max(1, math.ceil(len(genus_labels) / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _sample(rng: random.Random, population: Sequence[int], count: int) -> list[int]:
        values = list(population)
        if len(values) >= count:
            return rng.sample(values, count)
        return [rng.choice(values) for _ in range(count)]

    @staticmethod
    def _weighted_sample(
        rng: random.Random, population: Sequence[int], weights: Sequence[float], count: int
    ) -> list[int]:
        values = list(population)
        current_weights = list(weights)
        if len(values) < count:
            return rng.choices(values, weights=current_weights, k=count)
        selected: list[int] = []
        for _ in range(count):
            choice = rng.choices(range(len(values)), weights=current_weights, k=1)[0]
            selected.append(values.pop(choice))
            current_weights.pop(choice)
        return selected

    def _sample_classes(
        self, rng: random.Random, population: Sequence[int], weights: Sequence[float], count: int
    ) -> list[int]:
        if self.class_sampling == "uniform":
            return self._sample(rng, population, count)
        return self._weighted_sample(rng, population, weights, count)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        genera = list(self.index)
        species_weights = {
            genus: [len(self.index[genus][species]) ** self.sampling_power for species in self.index[genus]]
            for genus in genera
        }
        genus_weights = [sum(species_weights[genus]) for genus in genera]
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            selected_genera = self._sample_classes(rng, genera, genus_weights, self.genera_per_batch)
            forced_species: dict[int, list[int]] = {}
            if self.hard_negative_groups and rng.random() < self.hard_negative_probability:
                hard_genus, hard_group = rng.choice(self.hard_negative_groups)
                forced_species[hard_genus] = self._sample(rng, hard_group, 2)
                if hard_genus not in selected_genera:
                    selected_genera[-1] = hard_genus
            for genus in selected_genera:
                species_values = list(self.index[genus])
                selected_species = list(forced_species.get(genus, []))
                if len(selected_species) < self.species_per_genus:
                    available = [value for value in species_values if value not in selected_species]
                    available_weights = [
                        len(self.index[genus][species]) ** self.sampling_power for species in available
                    ]
                    population = available or species_values
                    weights = available_weights or species_weights[genus]
                    selected_species.extend(
                        self._sample_classes(
                            rng,
                            population,
                            weights,
                            self.species_per_genus - len(selected_species),
                        )
                    )
                for species in selected_species:
                    candidates = self.index[genus][species]
                    batch.extend(self._sample(rng, candidates, self.samples_per_species))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


class ReplayHierarchicalBatchSampler:
    """Compose every PK batch from fixed adaptation and replay genus slots."""

    def __init__(
        self, genus_labels: Sequence[int], species_labels: Sequence[int], roles: Sequence[str],
        *, replay_fraction: float, genera_per_batch: int, species_per_genus: int,
        samples_per_species: int, batches_per_epoch: int | None = None, seed: int = 42,
        class_sampling: str = "uniform", sampling_power: float = 0.5,
    ) -> None:
        if len(roles) != len(genus_labels):
            raise ValueError("roles and labels must have equal length")
        if not 0 < replay_fraction < 1:
            raise ValueError("replay_fraction must be in (0, 1)")
        replay_genera = max(1, min(genera_per_batch - 1, round(genera_per_batch * replay_fraction)))
        adaptation_genera = genera_per_batch - replay_genera
        self._indices: dict[str, list[int]] = {
            role: [index for index, value in enumerate(roles) if str(value) == role]
            for role in ("adaptation", "replay")
        }
        if any(not values for values in self._indices.values()):
            raise ValueError("Replay sampler requires both adaptation and replay rows")
        total_batch = genera_per_batch * species_per_genus * samples_per_species
        epochs = batches_per_epoch or max(1, math.ceil(len(genus_labels) / total_batch))
        common = dict(
            species_per_genus=species_per_genus, samples_per_species=samples_per_species,
            batches_per_epoch=epochs, class_sampling=class_sampling, sampling_power=sampling_power,
            hard_negative_probability=0.0,
        )
        self._samplers = {
            "adaptation": HierarchicalBatchSampler(
                [genus_labels[i] for i in self._indices["adaptation"]],
                [species_labels[i] for i in self._indices["adaptation"]],
                genera_per_batch=adaptation_genera, seed=seed, **common,
            ),
            "replay": HierarchicalBatchSampler(
                [genus_labels[i] for i in self._indices["replay"]],
                [species_labels[i] for i in self._indices["replay"]],
                genera_per_batch=replay_genera, seed=seed + 100_003, **common,
            ),
        }
        self.batch_size = total_batch
        self.batches_per_epoch = epochs
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for sampler in self._samplers.values():
            sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        iterators = {key: iter(value) for key, value in self._samplers.items()}
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for role in ("adaptation", "replay"):
                local = next(iterators[role])
                batch.extend(self._indices[role][index] for index in local)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
