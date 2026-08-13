"""Transformer decoder blocks implemented from scratch."""

import math
from typing import Optional
import torch
from .module import Module
from .attention import CausalSelfAttention, KVCache
from .feedforward import FeedForward, SwiGLUFeedForward
from .normalization import LayerNorm, RMSNorm
from .dropout import Dropout
from .residual import ResidualConnection
from .positional import build_position_encoder, RoPE, ALiBi
from ..core.types import Tensor
from ..core.errors import NNError


def _make_norm(norm_type: str, d_model: int) -> Module:
    if norm_type == "layernorm":
        return LayerNorm(d_model)
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    raise NNError(f"Unknown norm_type: {norm_type}")


class TransformerBlock(Module):
    """A single transformer decoder block.

    Consists of:
    1. Self-attention with residual connection
    2. Feed-forward network with residual connection

    Supports both pre-norm and post-norm architectures. Pre-norm is the default
    and is what every modern LLM uses: the residual stream stays un-normalized
    end to end, so gradients reach early layers without a warmup crutch.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "swiglu",
        norm_type: str = "rmsnorm",
        pre_norm: bool = True,
        bias: bool = True,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 512,
        rope: Optional[Module] = None,
        alibi: Optional[Module] = None,
        use_flash: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()

        self._layer_idx = layer_idx

        self._attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len,
            rope=rope,
            alibi=alibi,
            use_flash=use_flash,
        )

        if activation == "swiglu":
            # SwiGLU has three matrices instead of two, so the hidden size is
            # scaled by 2/3 to keep the parameter count comparable (LLaMA).
            hidden = int(2 * d_ff / 3)
            hidden = 64 * ((hidden + 63) // 64)  # round up for GEMM alignment
            ff_layer = SwiGLUFeedForward(d_model, hidden, bias=False)
        else:
            ff_layer = FeedForward(d_model, d_ff, activation, bias)

        self._attn_norm = _make_norm(norm_type, d_model)
        self._ff_norm = _make_norm(norm_type, d_model)
        self._dropout = Dropout(dropout)

        self._register_module("attention", self._attention)
        self._ff_layer = ff_layer
        self._register_module("ff_layer", ff_layer)
        self._register_module("attn_norm", self._attn_norm)
        self._register_module("ff_norm", self._ff_norm)
        self._register_module("dropout", self._dropout)

        self._pre_norm = pre_norm

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
        offset: int = 0,
    ) -> Tensor:
        """Forward pass through one transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional additive attention mask
            cache: Optional KV cache for incremental decoding
            offset: Position of the first token in the sequence

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self._pre_norm:
            attn_out = self._attention(
                self._attn_norm(x), mask, cache=cache, layer_idx=self._layer_idx, offset=offset
            )
            x = x + attn_out

            ff_out = self._dropout(self._ff_layer(self._ff_norm(x)))
            x = x + ff_out
        else:
            attn_out = self._attention(
                x, mask, cache=cache, layer_idx=self._layer_idx, offset=offset
            )
            x = self._attn_norm(x + attn_out)

            ff_out = self._dropout(self._ff_layer(x))
            x = self._ff_norm(x + ff_out)

        return x


class TransformerDecoder(Module):
    """Stack of transformer decoder blocks forming the core of the language model.

    Owns the position encoder so that a single RoPE/ALiBi table is shared by
    every layer instead of each block building its own copy.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        activation: str = "swiglu",
        norm_type: str = "rmsnorm",
        pre_norm: bool = True,
        bias: bool = True,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 512,
        position_encoding: str = "rope",
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        use_flash: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self._n_layers = n_layers
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self._head_dim = d_model // n_heads
        self._gradient_checkpointing = gradient_checkpointing

        # One position encoder, shared across all layers.
        self._position_encoder = build_position_encoder(
            kind=position_encoding,
            d_model=d_model,
            head_dim=self._head_dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
        )
        rope = self._position_encoder if isinstance(self._position_encoder, RoPE) else None
        alibi = self._position_encoder if isinstance(self._position_encoder, ALiBi) else None
        # Absolute encodings act on the embeddings, not on Q/K - the model adds
        # those before the stack runs.
        self._additive_position = self._position_encoder if (rope is None and alibi is None) else None

        if self._position_encoder is not None:
            self._register_module("position_encoder", self._position_encoder)

        for i in range(n_layers):
            block = TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type,
                pre_norm=pre_norm,
                bias=bias,
                n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len,
                rope=rope,
                alibi=alibi,
                use_flash=use_flash,
                layer_idx=i,
            )
            self._register_module(f"layer_{i}", block)

        self._final_norm = _make_norm(norm_type, d_model)
        self._register_module("final_norm", self._final_norm)

    @property
    def additive_position_encoder(self) -> Optional[Module]:
        """The sinusoidal/learned encoder, if this stack uses one."""
        return self._additive_position

    @property
    def n_kv_heads(self) -> int:
        return self._n_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Trade compute for memory: recompute activations during backward.

        Roughly 30-40% slower per step, but cuts activation memory by about the
        depth of the stack - the standard way to fit a larger batch or context.
        """
        self._gradient_checkpointing = enabled

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
        offset: int = 0,
    ) -> Tensor:
        """Forward pass through the transformer decoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional additive attention mask
            cache: Optional KV cache for incremental decoding
            offset: Position of the first token in the sequence

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        use_checkpoint = self._gradient_checkpointing and self._training and cache is None

        for i in range(self._n_layers):
            block = self._modules[f"layer_{i}"]
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, mask, None, offset, use_reentrant=False
                )
            else:
                x = block(x, mask, cache=cache, offset=offset)

        return self._final_norm(x)
