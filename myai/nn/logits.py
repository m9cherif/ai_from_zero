"""Vectorized logit processors for sampling.

Every operation here is batched: no Python loop over batch elements or over
vocabulary entries. Sampling runs once per generated token, so a per-token
Python loop over a 50k vocabulary dominates generation time.
"""

from typing import Optional
import torch
from ..core.types import Tensor


def apply_repetition_penalty(
    logits: Tensor,
    generated: Tensor,
    penalty: float = 1.0,
) -> Tensor:
    """Penalize tokens that already appeared (Keskar et al., 2019).

    Positive logits are divided by the penalty and negative ones multiplied, so
    the score always moves toward zero regardless of sign.

    Args:
        logits: (batch, vocab_size)
        generated: (batch, seq_len) token IDs produced so far
        penalty: >1.0 discourages repetition, 1.0 disables

    Returns:
        (batch, vocab_size)
    """
    if penalty == 1.0:
        return logits

    # Gather is a single batched kernel; the previous implementation looped
    # over batch x unique tokens in Python.
    score = torch.gather(logits, 1, generated)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, generated, score)


def apply_frequency_presence_penalty(
    logits: Tensor,
    generated: Tensor,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Tensor:
    """OpenAI-style frequency and presence penalties.

    frequency_penalty scales with how often a token was used; presence_penalty
    applies a flat cost the moment a token appears at all.
    """
    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits

    counts = torch.zeros_like(logits)
    counts.scatter_add_(1, generated, torch.ones_like(generated, dtype=logits.dtype))

    if frequency_penalty:
        logits = logits - frequency_penalty * counts
    if presence_penalty:
        logits = logits - presence_penalty * (counts > 0).to(logits.dtype)
    return logits


def apply_top_k(logits: Tensor, k: Optional[int]) -> Tensor:
    """Keep only the k highest-scoring tokens."""
    if k is None or k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: Tensor, p: Optional[float]) -> Tensor:
    """Nucleus sampling: keep the smallest set whose probability mass exceeds p."""
    if p is None or p >= 1.0 or p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Drop everything past the cutoff, but always keep the top token.
    remove = cumulative - torch.softmax(sorted_logits, dim=-1) > p
    remove[..., 0] = False

    # scatter back to the original ordering - no Python loop over the batch.
    mask = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
    return logits.masked_fill(mask, float("-inf"))


def apply_min_p(logits: Tensor, min_p: Optional[float]) -> Tensor:
    """Min-p sampling: keep tokens with probability >= min_p * p_max.

    Adapts the cutoff to the model's confidence - wide when the distribution is
    flat, tight when the model is sure. Often better than top-p at high
    temperature.
    """
    if min_p is None or min_p <= 0.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    top_prob = probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < min_p * top_prob, float("-inf"))


def sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Turn raw logits into one sampled token per batch row.

    Args:
        logits: (batch, vocab_size)
        temperature: 0 (or less) means greedy/argmax
        top_k / top_p / min_p: optional truncation filters
        generator: optional RNG for reproducible sampling

    Returns:
        (batch, 1) token IDs
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature != 1.0:
        logits = logits / temperature

    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    logits = apply_min_p(logits, min_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
