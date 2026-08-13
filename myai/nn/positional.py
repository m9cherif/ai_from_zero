"""Positional encodings implemented from scratch.

A transformer's attention is permutation-equivariant: without one of these,
"dog bites man" and "man bites dog" produce the same representation. Three
strategies are provided, matching what production LLMs actually use:

* ``RoPE``       - rotary embeddings (LLaMA, Qwen, Mistral, GPT-NeoX). Default.
* ``ALiBi``      - linear attention bias (BLOOM, MPT).
* ``Sinusoidal`` / ``Learned`` - the original absolute-position approaches.
"""

import math
from typing import Optional, Tuple
import torch
from .module import Module
from .parameter import Parameter
from .init import normal_
from ..core.types import Tensor


class SinusoidalPositionalEncoding(Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Uses fixed (non-learnable) sinusoidal functions, added to the embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self._d_model = d_model
        self._max_len = max_len

        # Precompute positional encodings: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            offset: Position of the first token (non-zero when decoding with a cache)

        Returns:
            Tensor of the same shape with positional encoding added
        """
        seq_len = x.shape[1]
        pe = self._buffers["pe"]
        if offset + seq_len > pe.shape[0]:
            raise ValueError(
                f"Sequence position {offset + seq_len} exceeds max_len={pe.shape[0]}"
            )
        pe = pe[offset:offset + seq_len, :].to(device=x.device, dtype=x.dtype)
        return x + pe.unsqueeze(0)


class LearnedPositionalEmbedding(Module):
    """Learned absolute position embeddings (GPT-2 style)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self._d_model = d_model
        self._max_len = max_len

        weight_data = torch.empty(max_len, d_model)
        normal_(weight_data, mean=0.0, std=0.02)
        self._register_parameter("weight", Parameter(weight_data))

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq_len = x.shape[1]
        if offset + seq_len > self._max_len:
            raise ValueError(
                f"Sequence position {offset + seq_len} exceeds max_len={self._max_len}"
            )
        pos = self._parameters["weight"].data[offset:offset + seq_len]
        return x + pos.unsqueeze(0)


class RoPE(Module):
    """Rotary Position Embedding (Su et al., 2021).

    Instead of adding a position vector to the token embedding, RoPE *rotates*
    each (even, odd) pair of query/key channels by an angle proportional to the
    token's position. The dot product between a query at position m and a key at
    position n then depends only on (m - n), giving attention a built-in notion
    of relative distance.

    Applied per-head, so ``dim`` is the head dimension - not d_model.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {dim}")

        self._dim = dim
        self._max_seq_len = max_seq_len
        self._base = base
        # >1.0 stretches the position grid, letting a model trained at a short
        # context extrapolate to a longer one (linear position interpolation).
        self._scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cached", None)
        self.register_buffer("sin_cached", None)
        self._cached_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Precompute cos/sin tables once and reuse them across every step."""
        cos = self._buffers.get("cos_cached")
        if (
            cos is not None
            and self._cached_len >= seq_len
            and cos.device == device
            and cos.dtype == dtype
        ):
            return

        inv_freq = self._buffers["inv_freq"].to(device=device, dtype=torch.float32)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        if self._scaling_factor != 1.0:
            t = t / self._scaling_factor

        freqs = torch.outer(t, inv_freq)          # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)   # (seq_len, dim)

        self.register_buffer("cos_cached", emb.cos().to(dtype))
        self.register_buffer("sin_cached", emb.sin().to(dtype))
        self._cached_len = seq_len

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Map (x1, x2) -> (-x2, x1) over the split halves of the last dim."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_kv_heads, seq_len, head_dim)
            offset: Position of the first token (non-zero when decoding with a cache)

        Returns:
            (rotated_q, rotated_k) with the same shapes as the inputs
        """
        seq_len = q.shape[-2]
        total_len = offset + seq_len

        self._build_cache(max(total_len, self._max_seq_len), q.device, q.dtype)

        cos = self._buffers["cos_cached"][offset:total_len].unsqueeze(0).unsqueeze(0)
        sin = self._buffers["sin_cached"][offset:total_len].unsqueeze(0).unsqueeze(0)

        q_out = q * cos + self._rotate_half(q) * sin
        k_out = k * cos + self._rotate_half(k) * sin
        return q_out, k_out


class ALiBi(Module):
    """Attention with Linear Biases (Press et al., 2022).

    Adds a per-head bias of ``-slope * (query_pos - key_pos)`` to the attention
    scores, penalizing distant tokens. No embedding is added anywhere, and the
    model extrapolates to sequences longer than it was trained on.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 512):
        super().__init__()
        self._n_heads = n_heads
        self._max_seq_len = max_seq_len
        self.register_buffer("slopes", self._build_slopes(n_heads))
        self.register_buffer("bias_cached", None)
        self._cached_len = 0

    @staticmethod
    def _build_slopes(n_heads: int) -> Tensor:
        """Geometric sequence of per-head slopes, as specified in the paper."""
        def power_of_two_slopes(n: int) -> list:
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = power_of_two_slopes(n_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(n_heads))
            slopes = power_of_two_slopes(closest_power)
            extra = power_of_two_slopes(2 * closest_power)[0::2][: n_heads - closest_power]
            slopes = slopes + extra
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, seq_len_q: int, seq_len_k: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build the additive attention bias of shape (1, n_heads, seq_len_q, seq_len_k)."""
        q_pos = torch.arange(seq_len_k - seq_len_q, seq_len_k, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
        distance = (k_pos - q_pos).clamp(max=0).to(torch.float32)  # 0 or negative
        slopes = self._buffers["slopes"].to(device=device, dtype=torch.float32)
        bias = distance.unsqueeze(0) * slopes.view(-1, 1, 1)
        return bias.unsqueeze(0).to(dtype)


def build_position_encoder(
    kind: str,
    d_model: int,
    head_dim: int,
    n_heads: int,
    max_seq_len: int,
    rope_base: float = 10000.0,
    rope_scaling: float = 1.0,
) -> Optional[Module]:
    """Factory returning the position module for a given strategy name."""
    kind = (kind or "none").lower()
    if kind == "rope":
        return RoPE(head_dim, max_seq_len=max_seq_len, base=rope_base, scaling_factor=rope_scaling)
    if kind == "alibi":
        return ALiBi(n_heads, max_seq_len=max_seq_len)
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
    if kind == "learned":
        return LearnedPositionalEmbedding(d_model, max_len=max_seq_len)
    if kind == "none":
        return None
    raise ValueError(
        f"Unknown position encoding '{kind}'. "
        f"Expected one of: rope, alibi, sinusoidal, learned, none."
    )
