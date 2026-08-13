"""Normalization layers implemented from scratch.

LayerNorm and RMSNorm implementations using only basic tensor operations.
"""

import math
import torch
from .module import Module
from .parameter import Parameter
from .init import zeros_, ones_
from ..core.types import Tensor


class LayerNorm(Module):
    """Layer Normalization.

    LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

    Normalizes across the last dimension.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self._normalized_shape = normalized_shape
        self._eps = eps
        self._elementwise_affine = elementwise_affine

        if elementwise_affine:
            gamma_data = torch.empty(normalized_shape)
            gamma = Parameter(gamma_data)
            ones_(gamma_data)
            self._register_parameter("gamma", gamma)

            beta_data = torch.empty(normalized_shape)
            beta = Parameter(beta_data)
            zeros_(beta_data)
            self._register_parameter("beta", beta)

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of the same shape
        """
        # Compute mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(variance + self._eps)

        if self._elementwise_affine:
            gamma = self._parameters["gamma"].data
            beta = self._parameters["beta"].data
            x_normalized = x_normalized * gamma + beta

        return x_normalized


class RMSNorm(Module):
    """Root Mean Square Layer Normalization.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

    A simpler alternative to LayerNorm that only scales (no shift).
    Used in many modern LLMs (LLaMA, etc.).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self._normalized_shape = normalized_shape
        self._eps = eps
        self._elementwise_affine = elementwise_affine

        if elementwise_affine:
            gamma_data = torch.empty(normalized_shape)
            gamma = Parameter(gamma_data)
            ones_(gamma_data)
            self._register_parameter("gamma", gamma)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of the same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self._eps)

        # Normalize: x / RMS
        x_normalized = x / rms

        if self._elementwise_affine:
            gamma = self._parameters["gamma"].data
            x_normalized = x_normalized * gamma

        return x_normalized
