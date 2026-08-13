"""Learning rate schedulers implemented from scratch."""

import math
from typing import Optional, Callable
from .optimizer import Optimizer
from ..core.logging import logger


class LRScheduler:
    """Base learning rate scheduler."""

    def __init__(self, optimizer: Optimizer, last_step: int = -1):
        self._optimizer = optimizer
        self._base_lrs = [group["lr"] for group in optimizer._param_groups]
        self._last_step = last_step
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1
        self._last_step = self._step_count
        for i, group in enumerate(self._optimizer._param_groups):
            group["lr"] = self._get_lr(i)

    def _get_lr(self, group_idx: int) -> float:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {
            "base_lrs": self._base_lrs,
            "last_step": self._last_step,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._base_lrs = state_dict["base_lrs"]
        self._last_step = state_dict["last_step"]
        self._step_count = state_dict["step_count"]


class ConstantLR(LRScheduler):
    """Constant learning rate."""

    def __init__(self, optimizer: Optimizer, last_step: int = -1):
        super().__init__(optimizer, last_step)

    def _get_lr(self, group_idx: int) -> float:
        return self._base_lrs[group_idx]


class LinearLR(LRScheduler):
    """Linear decay from base_lr to 0."""

    def __init__(self, optimizer: Optimizer, total_steps: int, last_step: int = -1):
        self._total_steps = total_steps
        super().__init__(optimizer, last_step)

    def _get_lr(self, group_idx: int) -> float:
        progress = min(1.0, self._step_count / max(self._total_steps, 1))
        return self._base_lrs[group_idx] * (1.0 - progress)


class CosineLR(LRScheduler):
    """Cosine annealing learning rate."""

    def __init__(self, optimizer: Optimizer, total_steps: int, min_lr_ratio: float = 0.1, last_step: int = -1):
        self._total_steps = total_steps
        self._min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_step)

    def _get_lr(self, group_idx: int) -> float:
        progress = min(1.0, self._step_count / max(self._total_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self._base_lrs[group_idx] * (self._min_lr_ratio + (1.0 - self._min_lr_ratio) * cosine)


class WarmupCosineLR(LRScheduler):
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1, last_step: int = -1):
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps
        self._min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_step)

    def _get_lr(self, group_idx: int) -> float:
        base_lr = self._base_lrs[group_idx]

        if self._step_count < self._warmup_steps:
            # Linear warmup
            return base_lr * (self._step_count / max(self._warmup_steps, 1))
        else:
            # Cosine decay
            progress = (self._step_count - self._warmup_steps) / max(self._total_steps - self._warmup_steps, 1)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_lr * (self._min_lr_ratio + (1.0 - self._min_lr_ratio) * cosine)


class WarmupLinearLR(LRScheduler):
    """Linear warmup followed by linear decay."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1, last_step: int = -1):
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps
        self._min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_step)

    def _get_lr(self, group_idx: int) -> float:
        base_lr = self._base_lrs[group_idx]

        if self._step_count < self._warmup_steps:
            return base_lr * (self._step_count / max(self._warmup_steps, 1))
        else:
            progress = (self._step_count - self._warmup_steps) / max(self._total_steps - self._warmup_steps, 1)
            progress = min(1.0, progress)
            return base_lr * (self._min_lr_ratio + (1.0 - self._min_lr_ratio) * (1.0 - progress))
