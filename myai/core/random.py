"""Random state management for full reproducibility."""

import random
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple


class RandomStateManager:
    """Manages random states across all libraries for deterministic execution."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._states: Dict[str, Any] = {}

    def seed_all(self, seed: Optional[int] = None) -> int:
        """Seed all random number generators and return the seed used."""
        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return self._seed

    def capture_state(self) -> Dict[str, Any]:
        """Capture current random states for checkpointing."""
        state = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_random": torch.get_rng_state(),
            "seed": self._seed,
        }
        if torch.cuda.is_available():
            state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore random states from a captured state dict."""
        if "seed" in state:
            self._seed = state["seed"]
        if "python_random" in state:
            random.setstate(state["python_random"])
        if "numpy_random" in state:
            np.random.set_state(state["numpy_random"])
        if "torch_random" in state:
            torch.set_rng_state(state["torch_random"])
        if "torch_cuda_random" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_random"])

    @property
    def seed(self) -> int:
        return self._seed


_default_rng = RandomStateManager(seed=42)


def get_default_rng() -> RandomStateManager:
    return _default_rng


def set_seed(seed: int) -> int:
    return _default_rng.seed_all(seed)
