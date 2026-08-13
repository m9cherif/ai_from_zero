"""Complete language model assembly.

Combines embedding, positional encoding, transformer decoder,
and output projection into a single end-to-end model.
"""

import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
import torch
from .module import Module
from .embedding import Embedding
from .transformer import TransformerDecoder
from .attention import KVCache, CausalSelfAttention
from .linear import Linear
from .loss import CrossEntropyLoss
from .init import zeros_, normal_
from .logits import (
    apply_repetition_penalty,
    apply_frequency_presence_penalty,
    sample_from_logits,
)
from ..core.types import Tensor
from ..core.errors import NNError


@dataclass
class LMConfig:
    """Language model configuration.

    This is a lightweight config for internal use; full config
    system is in myai.config.
    """
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    activation: str = "swiglu"
    norm_type: str = "rmsnorm"
    pre_norm: bool = True
    tie_embeddings: bool = True
    bias: bool = False
    padding_idx: Optional[int] = None

    # Grouped-query attention: n_kv_heads < n_heads shrinks the KV cache
    # proportionally. None means standard multi-head attention.
    n_kv_heads: Optional[int] = None

    # "rope" | "alibi" | "sinusoidal" | "learned" | "none"
    position_encoding: str = "rope"
    rope_base: float = 10000.0
    rope_scaling: float = 1.0

    # Opt-in fused SDPA/FlashAttention kernels. The hand-written attention path
    # is the default so the model stays "from scratch"; flip this on for speed.
    use_flash: bool = False
    gradient_checkpointing: bool = False

    init_std: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LMConfig":
        """Build a config, ignoring keys this version does not know about."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class LanguageModel(Module):
    """Complete autoregressive language model.

    Architecture:
    1. Token embeddings (+ absolute position embeddings, if configured)
    2. Transformer decoder stack (RoPE/ALiBi applied inside attention)
    3. Output projection (LM head), optionally tied to the embeddings

    All components are manually implemented.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self._config = config

        self._embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.padding_idx,
        )

        self._decoder = TransformerDecoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_layers,
            dropout=config.dropout,
            activation=config.activation,
            norm_type=config.norm_type,
            pre_norm=config.pre_norm,
            bias=config.bias,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            position_encoding=config.position_encoding,
            rope_base=config.rope_base,
            rope_scaling=config.rope_scaling,
            use_flash=config.use_flash,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        self._lm_head = Linear(config.d_model, config.vocab_size, bias=False)

        self._register_module("embedding", self._embedding)
        self._register_module("decoder", self._decoder)
        self._register_module("lm_head", self._lm_head)

        # Only used when an absolute position encoding is added to embeddings;
        # RoPE and ALiBi act inside attention and leave the embedding alone.
        self._embed_scale = math.sqrt(config.d_model)

        self._init_weights()

        # Weight tying shares the *Parameter object*, not just the tensor data,
        # so both modules stay bound through .to() and every optimizer step, and
        # the shared weight is counted and updated exactly once.
        if config.tie_embeddings:
            self._lm_head._parameters["weight"] = self._embedding._parameters["weight"]

        self._loss_fn = CrossEntropyLoss(reduction="mean")

    def _init_weights(self) -> None:
        """Initialize weights, scaling down the layers that write to the residual stream.

        Each layer adds into the residual stream, so after N layers its variance
        has grown by a factor of N. Scaling the output projections by
        1/sqrt(2*n_layers) keeps activations stable at depth (GPT-2's scheme).
        """
        std = self._config.init_std
        residual_std = std / math.sqrt(2.0 * max(self._config.n_layers, 1))

        for name, param in self.named_parameters():
            data = param.data
            if data.dim() < 2:
                continue  # biases and norm gains keep their own init
            is_residual_output = name.endswith("out_proj.weight") or name.endswith("down_proj.weight")
            normal_(data, mean=0.0, std=residual_std if is_residual_output else std)

        if self._config.padding_idx is not None:
            with torch.no_grad():
                self._embedding._parameters["weight"].data[self._config.padding_idx].zero_()

    def _build_padding_mask(self, attention_mask: Optional[Tensor], dtype: torch.dtype) -> Optional[Tensor]:
        """Turn a (batch, seq_len) 0/1 mask into an additive attention bias."""
        if attention_mask is None:
            return None
        mask = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
        offset: int = 0,
        num_logits_to_keep: int = 0,
    ) -> Dict[str, Tensor]:
        """Forward pass through the entire language model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            labels: Target token IDs of shape (batch_size, seq_len) for loss
            attention_mask: Optional 0/1 mask of shape (batch_size, seq_len)
            cache: Optional KV cache for incremental decoding
            offset: Position of the first token (non-zero when decoding)
            num_logits_to_keep: If > 0, only project the last N positions through
                the LM head. During generation only the final position matters,
                and the vocabulary projection is the most expensive layer in the
                model, so this avoids computing seq_len x vocab_size logits.

        Returns:
            Dictionary containing:
                - "logits": Raw logits of shape (batch_size, kept_len, vocab_size)
                - "loss": Scalar loss if labels are provided
        """
        x = self._embedding(input_ids)

        # Absolute (sinusoidal / learned) encodings act on the embeddings.
        additive_pos = self._decoder.additive_position_encoder
        if additive_pos is not None:
            # Scale embeddings by sqrt(d_model) before adding the position
            # signal, as in Vaswani et al. Sinusoidal components live in
            # [-1, 1] while embeddings are initialized at std=0.02; without
            # this the position vector is ~100x larger than the token vector
            # and the normalization layer discards token identity entirely.
            x = x * self._embed_scale
            x = additive_pos(x, offset=offset)

        attn_bias = self._build_padding_mask(attention_mask, x.dtype)

        x = self._decoder(x, attn_bias, cache=cache, offset=offset)

        if cache is not None:
            cache.advance(input_ids.shape[1])

        if num_logits_to_keep > 0:
            x = x[:, -num_logits_to_keep:, :]

        logits = self._lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            # Predict token t+1 from position t.
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            result["loss"] = self._loss_fn(shift_logits, shift_labels)

        return result

    def build_cache(self, batch_size: int, max_seq_len: Optional[int] = None) -> KVCache:
        """Allocate a KV cache sized for this model."""
        max_seq_len = max_seq_len or self._config.max_seq_len
        param = next(self.parameters())
        return KVCache(
            n_layers=self._config.n_layers,
            batch_size=batch_size,
            n_kv_heads=self._decoder.n_kv_heads,
            head_dim=self._decoder.head_dim,
            max_seq_len=max_seq_len,
            device=param.device,
            dtype=param.dtype,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        use_cache: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Autoregressive generation with KV caching.

        Args:
            input_ids: Prompt token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature; <= 0 selects greedy decoding
            top_k: Sample only from the top-k tokens
            top_p: Nucleus sampling threshold
            min_p: Confidence-relative probability floor
            eos_token_id: Stop a sequence once this token is produced
            pad_token_id: Token used to fill finished sequences
            repetition_penalty: >1.0 discourages repeats
            frequency_penalty: Per-occurrence logit penalty
            presence_penalty: Flat penalty for any already-used token
            use_cache: Enable KV caching (O(n) instead of O(n^2))
            generator: Optional RNG for reproducible sampling

        Returns:
            Generated token IDs including the prompt
        """
        was_training = self.training
        self.eval()

        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        max_seq_len = self._config.max_seq_len

        if prompt_len > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
            prompt_len = max_seq_len

        generated = input_ids
        pad_token_id = pad_token_id if pad_token_id is not None else (eos_token_id or 0)

        # Tracks which rows have already emitted EOS, so a finished sequence is
        # padded rather than continuing to sample noise.
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        cache = None
        if use_cache:
            cache = self.build_cache(
                batch_size=batch_size,
                max_seq_len=min(max_seq_len, prompt_len + max_new_tokens),
            )

        # Prefill: run the whole prompt once, keeping only the final logits.
        step_input = generated
        offset = 0

        for _ in range(max_new_tokens):
            outputs = self(
                step_input,
                cache=cache,
                offset=offset,
                num_logits_to_keep=1,
            )
            next_token_logits = outputs["logits"][:, -1, :].float()

            next_token_logits = apply_repetition_penalty(
                next_token_logits, generated, repetition_penalty
            )
            next_token_logits = apply_frequency_presence_penalty(
                next_token_logits, generated, frequency_penalty, presence_penalty
            )

            next_token = sample_from_logits(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                generator=generator,
            )

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, pad_token_id),
                    next_token,
                )
                finished = finished | (next_token.squeeze(1) == eos_token_id)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and bool(finished.all()):
                break

            if cache is not None:
                # With a cache only the new token is fed back in.
                offset = cache.length
                step_input = next_token
                if offset >= max_seq_len:
                    break
            else:
                step_input = generated[:, -max_seq_len:]

        if was_training:
            self.train()

        return generated

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Recompute activations in the backward pass to save memory."""
        self._decoder.set_gradient_checkpointing(enabled)
        self._config.gradient_checkpointing = enabled

    def set_flash_attention(self, enabled: bool = True) -> None:
        """Switch every attention layer between the manual and fused kernels."""
        from .attention import _HAS_SDPA
        if enabled and not _HAS_SDPA:
            raise NNError("Fused SDPA is unavailable in this PyTorch build")
        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                module._use_flash = enabled
        self._config.use_flash = enabled

    def num_parameters_excluding_embeddings(self) -> int:
        """Parameter count without the token embedding table.

        This is the number usually quoted for a model's "size", since embedding
        parameters scale with vocabulary rather than with capacity.
        """
        total = self.num_parameters()
        emb = self._embedding._parameters["weight"].data.numel()
        # Untied models carry a second vocab-sized matrix in the LM head.
        return total - (emb if self._config.tie_embeddings else 2 * emb)

    def estimate_flops_per_token(self) -> int:
        """Rough forward+backward FLOPs per token (6 * non-embedding params)."""
        return 6 * self.num_parameters_excluding_embeddings()

    @property
    def config(self) -> LMConfig:
        return self._config

    @property
    def device(self) -> torch.device:
        for param in self.parameters():
            return param.data.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for param in self.parameters():
            return param.data.dtype
        return torch.float32

    def __repr__(self) -> str:
        c = self._config
        return (
            f"LanguageModel(params={self.num_parameters():,}, "
            f"d_model={c.d_model}, n_layers={c.n_layers}, n_heads={c.n_heads}, "
            f"n_kv_heads={c.n_kv_heads or c.n_heads}, vocab={c.vocab_size}, "
            f"pos={c.position_encoding}, act={c.activation}, tied={c.tie_embeddings})"
        )
