"""Feed-forward network implementations from scratch."""

from .module import Module
from .linear import Linear
from .activation import ReLU, GELU, SiLU
from ..core.errors import NNError
from ..core.types import Tensor


class FeedForward(Module):
    """Standard feed-forward network with one hidden layer.

    FFN(x) = activation(x @ W1 + b1) @ W2 + b2

    Used in the original transformer architecture.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu",
        bias: bool = True,
    ):
        super().__init__()

        self._gate_proj = Linear(d_model, d_ff, bias=bias)
        self._down_proj = Linear(d_ff, d_model, bias=bias)

        if activation == "relu":
            self._activation = ReLU()
        elif activation == "gelu":
            self._activation = GELU()
        elif activation == "silu":
            self._activation = SiLU()
        else:
            raise NNError(f"Unknown activation: {activation}")

        self._register_module("gate_proj", self._gate_proj)
        self._register_module("down_proj", self._down_proj)
        self._register_module("activation", self._activation)

    def forward(self, x: Tensor) -> Tensor:
        """FFN forward pass.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        hidden = self._gate_proj(x)
        hidden = self._activation(hidden)
        output = self._down_proj(hidden)
        return output


class SwiGLUFeedForward(Module):
    """Feed-forward network using SwiGLU activation.

    SwiGLU(x) = (activation(x @ W_gate) * (x @ W_up)) @ W_down

    Used in LLaMA, PaLM, and other modern LLMs.
    Unlike standard FFN, this has three weight matrices (gate, up, down).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False,
    ):
        super().__init__()

        # Standard SwiGLU uses 8/3 * d_model for the hidden dimension
        # d_ff here is the effective intermediate size
        self._gate_proj = Linear(d_model, d_ff, bias=bias)
        self._up_proj = Linear(d_model, d_ff, bias=bias)
        self._down_proj = Linear(d_ff, d_model, bias=bias)
        self._activation = SiLU()

        self._register_module("gate_proj", self._gate_proj)
        self._register_module("up_proj", self._up_proj)
        self._register_module("down_proj", self._down_proj)
        self._register_module("activation", self._activation)

    def forward(self, x: Tensor) -> Tensor:
        """SwiGLU FFN forward pass.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        gate = self._gate_proj(x)
        gate = self._activation(gate)
        up = self._up_proj(x)
        hidden = gate * up
        output = self._down_proj(hidden)
        return output
