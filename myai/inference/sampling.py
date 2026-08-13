"""Sampling algorithms for text generation.

Thin object wrappers over the vectorized processors in ``myai.nn.logits`` so a
sampling strategy can be configured once and reused across calls.
"""

import torch
from typing import Optional, List
from ..core.types import Tensor
from ..core.errors import InferenceError
from ..nn.logits import apply_top_k, apply_top_p, apply_min_p, sample_from_logits


class Sampler:
    """Base sampler class for token selection."""

    def sample(self, logits: Tensor) -> Tensor:
        """Sample a token from logits.

        Args:
            logits: Tensor of shape (batch_size, vocab_size)

        Returns:
            Sampled token IDs of shape (batch_size, 1)
        """
        raise NotImplementedError


class TemperatureSampler(Sampler):
    """Temperature-controlled sampling.

    Temperature > 1: more random, Temperature < 1: more deterministic.
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise InferenceError(f"Temperature must be > 0, got {temperature}")
        self._temperature = temperature

    def sample(self, logits: Tensor) -> Tensor:
        return sample_from_logits(logits, temperature=self._temperature)


class TopKSampler(Sampler):
    """Top-k sampling: sample only from the k most likely tokens."""

    def __init__(self, k: int = 50, temperature: float = 1.0):
        if k <= 0:
            raise InferenceError(f"k must be > 0, got {k}")
        self._k = k
        self._temperature = temperature

    def sample(self, logits: Tensor) -> Tensor:
        return sample_from_logits(logits, temperature=self._temperature, top_k=self._k)


class TopPSampler(Sampler):
    """Top-p (nucleus) sampling: sample from the smallest set of tokens
    whose cumulative probability exceeds p.
    """

    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        if p <= 0 or p > 1:
            raise InferenceError(f"p must be in (0, 1], got {p}")
        self._p = p
        self._temperature = temperature

    def sample(self, logits: Tensor) -> Tensor:
        return sample_from_logits(logits, temperature=self._temperature, top_p=self._p)


class MinPSampler(Sampler):
    """Min-p sampling: keep tokens with probability >= min_p * p_max.

    The cutoff scales with the model's confidence, so it stays permissive on
    genuinely ambiguous next tokens and tight where the model is certain.
    """

    def __init__(self, min_p: float = 0.05, temperature: float = 1.0):
        if min_p <= 0 or min_p > 1:
            raise InferenceError(f"min_p must be in (0, 1], got {min_p}")
        self._min_p = min_p
        self._temperature = temperature

    def sample(self, logits: Tensor) -> Tensor:
        return sample_from_logits(logits, temperature=self._temperature, min_p=self._min_p)


class GreedySampler(Sampler):
    """Greedy (argmax) sampling - always picks the most likely token."""

    def sample(self, logits: Tensor) -> Tensor:
        return logits.argmax(dim=-1, keepdim=True)


class CompositeSampler(Sampler):
    """Applies temperature, top-k, top-p and min-p together, in that order."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
    ):
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._min_p = min_p

    def sample(self, logits: Tensor) -> Tensor:
        return sample_from_logits(
            logits,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            min_p=self._min_p,
        )
