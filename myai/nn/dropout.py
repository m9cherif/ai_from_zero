"""Dropout layer implemented from scratch."""

import torch
from .module import Module
from ..core.types import Tensor


class Dropout(Module):
    """Dropout regularization layer.

    Randomly zeroes elements during training with probability p.
    Scales remaining elements by 1/(1-p) to preserve expected value.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self._p = p

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout.

        Args:
            x: Input tensor of any shape

        Returns:
            Tensor of the same shape with dropout applied
        """
        if not self._training or self._p == 0.0:
            return x

        # bernoulli_ writes the mask directly in the input dtype - one
        # allocation instead of the rand + bool + float() chain.
        keep_prob = 1.0 - self._p
        mask = torch.empty_like(x).bernoulli_(keep_prob)

        # Inverted dropout: scale at train time so inference needs no rescaling.
        return x * mask * (1.0 / keep_prob)
