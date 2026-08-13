"""Embedding layer implemented from scratch.

Implements token embedding lookup using a manually managed
embedding matrix.
"""

import math
from typing import Optional
import torch
from .module import Module
from .parameter import Parameter
from .init import normal_
from ..core.types import Tensor


class Embedding(Module):
    """Token embedding layer.

    Maps token IDs to dense vectors through a learnable embedding matrix.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._padding_idx = padding_idx

        # Embedding matrix: shape (num_embeddings, embedding_dim)
        weight_data = torch.empty(num_embeddings, embedding_dim)
        weight = Parameter(weight_data)

        # Initialize with small normal distribution
        std = 1.0 / math.sqrt(embedding_dim)
        normal_(weight_data, mean=0.0, std=std)

        if padding_idx is not None:
            with torch.no_grad():
                weight_data[padding_idx].zero_()

        self._register_parameter("weight", weight)

    def forward(self, indices: Tensor) -> Tensor:
        """Forward pass: look up embeddings for given indices.

        Args:
            indices: Long tensor of shape (..., seq_len)

        Returns:
            Embedding tensor of shape (..., seq_len, embedding_dim)
        """
        weight = self._parameters["weight"].data

        if self._padding_idx is not None:
            # Create a mask for padding positions
            mask = indices == self._padding_idx
            # Use clamped indices to avoid OOB, then zero out padding positions
            clamped = indices.clamp(0, self._num_embeddings - 1)
            embedded = torch.nn.functional.embedding(clamped, weight, padding_idx=self._padding_idx)
            # Zero out the masked positions
            embedded = embedded * (~mask).unsqueeze(-1).float()
        else:
            embedded = torch.nn.functional.embedding(indices, weight)

        return embedded

    @property
    def weight(self) -> Tensor:
        return self._parameters["weight"].data

    def __repr__(self) -> str:
        return f"Embedding({self._num_embeddings}, {self._embedding_dim})"
