"""Activation functions implemented from scratch.

Each activation is defined mathematically and implemented
using only basic tensor operations.
"""

import math
import torch
from .module import Module
from ..core.types import Tensor


class ReLU(Module):
    """Rectified Linear Unit: f(x) = max(0, x)."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.maximum(x, torch.zeros_like(x))


class GELU(Module):
    """Gaussian Error Linear Unit.

    Implementation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self._approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        if self._approximate == "tanh":
            # GELU approximation from the original paper
            coeff = math.sqrt(2.0 / math.pi)
            return 0.5 * x * (1.0 + torch.tanh(coeff * (x + 0.044715 * x ** 3)))
        else:
            # Exact GELU using the error function
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SiLU(Module):
    """Sigmoid Linear Unit (also called Swish).

    SiLU(x) = x * sigmoid(x)
    """

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(Module):
    """SwiGLU activation: SwiGLU(x, gate) = x * sigmoid(gate) * gate.

    Combines Swish activation with gated linear unit structure.
    Typically used in the feed-forward network of modern LLMs.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        from .linear import Linear
        self._gate_proj = Linear(dim, hidden_dim, bias=False)
        self._register_module("gate_proj", self._gate_proj)

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        """SwiGLU(x, gate) = x * sigmoid(gate) * gate."""
        gate = self._gate_proj(x)
        return hidden * torch.sigmoid(gate) * gate
