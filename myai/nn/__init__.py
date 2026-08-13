"""Neural network module - all components implemented from scratch.

Every building block is manually implemented using only PyTorch's
basic tensor operations (no torch.nn.Linear, torch.nn.LayerNorm, etc.).
"""

from .parameter import Parameter, ParameterList
from .module import Module
from .init import zeros_, ones_, uniform_, normal_, kaiming_uniform_, xavier_uniform_
from .linear import Linear
from .embedding import Embedding
from .activation import ReLU, GELU, SiLU, SwiGLU
from .normalization import LayerNorm, RMSNorm
from .attention import MultiHeadAttention, CausalSelfAttention, KVCache, repeat_kv
from .dropout import Dropout
from .residual import ResidualConnection
from .positional import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RoPE,
    ALiBi,
    build_position_encoder,
)
from .feedforward import FeedForward, SwiGLUFeedForward
from .loss import CrossEntropyLoss
from .logits import (
    apply_repetition_penalty,
    apply_frequency_presence_penalty,
    apply_top_k,
    apply_top_p,
    apply_min_p,
    sample_from_logits,
)
from .transformer import TransformerBlock, TransformerDecoder
from .model import LanguageModel, LMConfig

__all__ = [
    "Parameter", "ParameterList",
    "Module",
    "zeros_", "ones_", "uniform_", "normal_", "kaiming_uniform_", "xavier_uniform_",
    "Linear",
    "Embedding",
    "ReLU", "GELU", "SiLU", "SwiGLU",
    "LayerNorm", "RMSNorm",
    "MultiHeadAttention", "CausalSelfAttention", "KVCache", "repeat_kv",
    "Dropout",
    "ResidualConnection",
    "SinusoidalPositionalEncoding", "LearnedPositionalEmbedding",
    "RoPE", "ALiBi", "build_position_encoder",
    "FeedForward", "SwiGLUFeedForward",
    "CrossEntropyLoss",
    "apply_repetition_penalty", "apply_frequency_presence_penalty",
    "apply_top_k", "apply_top_p", "apply_min_p", "sample_from_logits",
    "TransformerBlock", "TransformerDecoder",
    "LanguageModel", "LMConfig",
]
