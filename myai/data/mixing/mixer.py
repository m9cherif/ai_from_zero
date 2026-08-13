"""Multi-dataset mixing with configurable sampling ratios."""

import random
from typing import Dict, List, Iterator, Optional, Tuple, Any
from ...core.logging import logger


class DatasetMixer:
    """Combines multiple data sources with configurable sampling ratios.

    Supports weighted sampling, curriculum learning schedules,
    and dynamic dataset rebalancing during training.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._datasets: Dict[str, Iterator] = {}
        self._weights: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def add_dataset(self, name: str, iterator: Iterator, weight: float = 1.0) -> None:
        """Add a dataset with an optional sampling weight."""
        self._datasets[name] = iterator
        self._weights[name] = weight
        self._counts[name] = 0
        logger.info(f"Added dataset '{name}' with weight {weight}")

    def remove_dataset(self, name: str) -> None:
        self._datasets.pop(name, None)
        self._weights.pop(name, None)
        self._counts.pop(name, None)

    def set_weight(self, name: str, weight: float) -> None:
        if name in self._weights:
            self._weights[name] = weight

    def sample(self) -> Tuple[Optional[str], Any]:
        """Sample an item from the datasets according to weights."""
        if not self._datasets:
            return None, None

        # Weighted random selection of dataset
        names = list(self._datasets.keys())
        weights = [self._weights[n] for n in names]
        total = sum(weights)
        if total == 0:
            return None, None

        chosen = self._rng.choices(names, weights=weights, k=1)[0]

        # Get next item from chosen dataset
        iterator = self._datasets[chosen]
        try:
            item = next(iterator)
            self._counts[chosen] += 1
            return chosen, item
        except StopIteration:
            logger.warning(f"Dataset '{chosen}' exhausted")
            del self._datasets[chosen]
            del self._weights[chosen]
            del self._counts[chosen]
            return self.sample()

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterate indefinitely over mixed datasets."""
        while self._datasets:
            name, item = self.sample()
            if item is not None:
                yield name, item

    def get_counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def reset_counts(self) -> None:
        self._counts = {n: 0 for n in self._counts}
