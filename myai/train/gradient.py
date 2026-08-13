"""Gradient management: clipping and accumulation."""

import torch
from typing import Iterable, Optional, List
from ..nn.parameter import Parameter
from ..core.types import Tensor
from ..core.logging import logger


class GradientClipper:
    """Gradient clipping to prevent exploding gradients.

    Rescales every gradient by a single coefficient so the global norm across
    all parameters is at most ``max_norm``. Computing that norm with
    ``_foreach_norm`` keeps the work on-device: the previous per-tensor
    ``.item()`` call forced a host synchronization for every parameter, which
    on GPU costs more than the clipping itself.
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self._max_norm = max_norm
        self._norm_type = norm_type

    def clip(self, parameters: Iterable[Parameter]) -> float:
        """Clip gradients in place and return the pre-clip global norm."""
        grads: List[Tensor] = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return 0.0

        norms = torch._foreach_norm(grads, self._norm_type)
        total_norm = torch.linalg.vector_norm(torch.stack(norms), self._norm_type)

        # clamp(max=1.0) makes this branch-free: scaling by 1.0 is a no-op when
        # the norm is already under the limit.
        clip_coef = (self._max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(grads, clip_coef)

        return total_norm.item()


class GradientAccumulator:
    """Accumulates gradients over multiple micro-batches.

    Simulates a large batch on limited memory: run N micro-batches, scale each
    loss by 1/N so the summed gradient equals the mean over the full batch, and
    step the optimizer only on the last one.
    """

    def __init__(self, num_micro_batches: int = 1):
        if num_micro_batches < 1:
            raise ValueError(f"num_micro_batches must be >= 1, got {num_micro_batches}")
        self._num_micro_batches = num_micro_batches
        self._current_batch = 0

    @property
    def num_micro_batches(self) -> int:
        return self._num_micro_batches

    @property
    def current_batch(self) -> int:
        return self._current_batch

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss so accumulated gradients average rather than sum."""
        if self._num_micro_batches == 1:
            return loss
        return loss / self._num_micro_batches

    def should_step(self) -> bool:
        """Advance the micro-batch counter; True when the window is complete."""
        self._current_batch += 1
        if self._current_batch >= self._num_micro_batches:
            self._current_batch = 0
            return True
        return False

    @property
    def is_pending(self) -> bool:
        """True when gradients are held from an incomplete accumulation window."""
        return self._current_batch > 0

    def reset(self) -> None:
        self._current_batch = 0
