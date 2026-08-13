"""Attention mechanisms implemented from scratch.

Includes the textbook multi-head attention and an optimized causal
self-attention with the techniques modern LLMs rely on:

* **Fused QKV projection** - one matmul instead of three.
* **Grouped-Query Attention (GQA)** - fewer key/value heads than query heads,
  shrinking the KV cache (LLaMA-2 70B, Mistral, Qwen).
* **RoPE** - relative position information applied to Q and K.
* **KV caching** - generation becomes O(n) instead of O(n^2).
* **Cached causal mask** - built once, sliced per step.

The attention math itself is written out by hand. An opt-in fast path can
delegate to PyTorch's fused SDPA (FlashAttention) kernels instead; both paths
are tested to produce the same numbers.
"""

import math
from typing import Optional, Tuple, List
import torch
from .module import Module
from .parameter import Parameter
from .linear import Linear
from .dropout import Dropout
from .init import zeros_, normal_
from ..core.types import Tensor
from ..core.errors import NNError

# PyTorch's fused attention kernel, used only when explicitly enabled.
_HAS_SDPA = hasattr(torch.nn.functional, "scaled_dot_product_attention")


class KVCache:
    """Preallocated key/value cache for autoregressive decoding.

    Without a cache, generating token n re-encodes all n-1 previous tokens, so
    producing N tokens costs O(N^2) forward passes' worth of work. The cache
    keeps every layer's projected keys and values, so each new token only
    attends against stored state: O(N) total.

    Buffers are allocated once up front - growing a tensor per step would
    reallocate and copy the whole cache on every token.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        self._k: List[Tensor] = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self._v: List[Tensor] = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self._max_seq_len = max_seq_len
        self._length = 0

    @property
    def length(self) -> int:
        """Number of tokens currently stored."""
        return self._length

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Write this step's keys/values and return the full history."""
        new_len = k.shape[2]
        end = self._length + new_len
        if end > self._max_seq_len:
            raise NNError(
                f"KV cache overflow: {end} tokens requested, capacity is {self._max_seq_len}"
            )
        self._k[layer_idx][:, :, self._length:end] = k
        self._v[layer_idx][:, :, self._length:end] = v
        return self._k[layer_idx][:, :, :end], self._v[layer_idx][:, :, :end]

    def advance(self, num_tokens: int) -> None:
        """Commit a step. Called once per forward pass, after every layer."""
        self._length += num_tokens

    def reset(self) -> None:
        self._length = 0


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand grouped key/value heads to match the number of query heads.

    Uses expand + reshape rather than repeat_interleave: the expand is a view,
    so only the final reshape materializes memory.
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class MultiHeadAttention(Module):
    """Multi-Head Attention as described in "Attention Is All You Need".

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Kept as the reference implementation with separate Q/K/V projections and
    support for cross-attention (distinct query/key/value inputs).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise NNError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self._d_model = d_model
        self._n_heads = n_heads
        self._head_dim = d_model // n_heads
        self._scale = 1.0 / math.sqrt(self._head_dim)

        self._q_proj = Linear(d_model, d_model, bias=bias)
        self._k_proj = Linear(d_model, d_model, bias=bias)
        self._v_proj = Linear(d_model, d_model, bias=bias)
        self._out_proj = Linear(d_model, d_model, bias=bias)

        self._dropout = Dropout(dropout)

        self._register_module("q_proj", self._q_proj)
        self._register_module("k_proj", self._k_proj)
        self._register_module("v_proj", self._v_proj)
        self._register_module("out_proj", self._out_proj)
        self._register_module("dropout", self._dropout)

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """Reshape from (batch, seq_len, d_model) to (batch, n_heads, seq_len, head_dim)."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self._n_heads, self._head_dim)
        return x.transpose(1, 2)

    def _scaled_dot_product(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        """Compute scaled dot-product attention.

        Args:
            q: (batch, n_heads, seq_len_q, head_dim)
            k: (batch, n_heads, seq_len_k, head_dim)
            v: (batch, n_heads, seq_len_v, head_dim)
            mask: Optional attention mask (batch, 1, seq_len_q, seq_len_k)

        Returns:
            Context tensor of shape (batch, n_heads, seq_len_q, head_dim)
        """
        # Scale the query rather than the full score matrix: same result,
        # but the multiply touches seq_len x head_dim instead of seq_len^2.
        scores = torch.matmul(q * self._scale, k.transpose(-2, -1))

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                scores = scores + mask

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self._dropout(attention_weights)

        return torch.matmul(attention_weights, v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """Multi-head attention forward pass.

        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: Optional mask of shape (batch, 1, seq_len_q, seq_len_k)

        Returns:
            Output tensor of shape (batch, seq_len_q, d_model)
        """
        batch_size = query.shape[0]

        q = self._reshape_for_attention(self._q_proj(query))
        k = self._reshape_for_attention(self._k_proj(key))
        v = self._reshape_for_attention(self._v_proj(value))

        context = self._scaled_dot_product(q, k, v, mask)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self._d_model)

        return self._out_proj(context)


class CausalSelfAttention(Module):
    """Optimized causal self-attention for autoregressive language modeling.

    Each position attends only to itself and earlier positions. Supports
    grouped-query attention, rotary embeddings, ALiBi biases, and KV caching.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 512,
        rope: Optional[Module] = None,
        alibi: Optional[Module] = None,
        use_flash: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise NNError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % n_kv_heads != 0:
            raise NNError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )

        self._d_model = d_model
        self._n_heads = n_heads
        self._n_kv_heads = n_kv_heads
        self._n_rep = n_heads // n_kv_heads
        self._head_dim = d_model // n_heads
        self._scale = 1.0 / math.sqrt(self._head_dim)
        self._dropout_p = dropout
        self._max_seq_len = max_seq_len
        self._rope = rope
        self._alibi = alibi
        self._use_flash = use_flash and _HAS_SDPA

        # One fused projection for Q, K and V. A single (d_model x fused_dim)
        # matmul is meaningfully faster than three smaller ones: better GEMM
        # shapes, one bias add, one kernel launch.
        self._q_dim = n_heads * self._head_dim
        self._kv_dim = n_kv_heads * self._head_dim
        self._qkv_proj = Linear(d_model, self._q_dim + 2 * self._kv_dim, bias=bias)
        self._out_proj = Linear(d_model, d_model, bias=bias)

        self._dropout = Dropout(dropout)
        self._resid_dropout = Dropout(dropout)

        self._register_module("qkv_proj", self._qkv_proj)
        self._register_module("out_proj", self._out_proj)
        self._register_module("dropout", self._dropout)
        self._register_module("resid_dropout", self._resid_dropout)

        # Causal mask built once, then sliced. Rebuilding a (seq, seq) triangular
        # matrix every forward pass is pure waste.
        causal = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal)

    @property
    def n_kv_heads(self) -> int:
        return self._n_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    def _get_causal_mask(self, seq_len_q: int, seq_len_k: int, device: torch.device) -> Tensor:
        """Slice the cached mask for the current query/key lengths.

        With a KV cache the query block sits at the end of the key sequence, so
        the relevant rows are the last ``seq_len_q`` of the full mask.
        """
        mask = self._buffers["causal_mask"]
        if mask.device != device:
            mask = mask.to(device)
            self.register_buffer("causal_mask", mask)
        if seq_len_k > mask.shape[0]:
            mask = torch.triu(
                torch.ones(seq_len_k, seq_len_k, dtype=torch.bool, device=device), diagonal=1
            )
            self.register_buffer("causal_mask", mask)
        start = seq_len_k - seq_len_q
        return mask[start:seq_len_k, :seq_len_k]

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        offset: int = 0,
    ) -> Tensor:
        """Causal self-attention forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional additive padding mask (batch, 1, 1, seq_len_k)
            cache: Optional KV cache for incremental decoding
            layer_idx: This layer's index, used to address the cache
            offset: Position of the first token in the sequence

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Fused projection, then split into Q, K, V.
        qkv = self._qkv_proj(x)
        q, k, v = qkv.split([self._q_dim, self._kv_dim, self._kv_dim], dim=-1)

        q = q.view(batch_size, seq_len, self._n_heads, self._head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self._n_kv_heads, self._head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self._n_kv_heads, self._head_dim).transpose(1, 2)

        # Rotary embeddings are applied before caching, so cached keys already
        # carry their absolute position and stay valid for later steps.
        if self._rope is not None:
            q, k = self._rope(q, k, offset=offset)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        seq_len_k = k.shape[2]

        # Grouped-query attention: broadcast each KV head across its query group.
        k = repeat_kv(k, self._n_rep)
        v = repeat_kv(v, self._n_rep)

        causal_mask = self._get_causal_mask(seq_len, seq_len_k, x.device)

        if self._use_flash:
            attn_mask = None
            if mask is not None:
                attn_mask = mask.masked_fill(causal_mask, float("-inf"))
            else:
                attn_mask = torch.zeros(
                    seq_len, seq_len_k, device=x.device, dtype=x.dtype
                ).masked_fill(causal_mask, float("-inf"))
            context = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self._dropout_p if self.training else 0.0,
            )
        else:
            scores = torch.matmul(q * self._scale, k.transpose(-2, -1))

            if self._alibi is not None:
                scores = scores + self._alibi(seq_len, seq_len_k, x.device, scores.dtype)

            if mask is not None:
                scores = scores + mask

            # -inf would make fully-masked rows produce NaN after softmax;
            # the dtype's most negative finite value keeps them well-defined.
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

            attn = torch.softmax(scores, dim=-1)
            attn = self._dropout(attn)
            context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self._d_model)
        return self._resid_dropout(self._out_proj(context))
