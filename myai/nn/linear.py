"""Linear (dense) layer implemented from scratch.

Implements y = x @ W^T + b using only basic tensor operations.
"""

import math
import torch
from .module import Module
from .parameter import Parameter
from .init import kaiming_uniform_, zeros_
from ..core.types import Tensor


class Linear(Module):
    """A fully-connected linear transformation: y = x @ W^T + b.

    All parameters (weight, bias) are manually defined and initialized.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._use_bias = bias

        # Manually create weight parameter: shape (out_features, in_features)
        weight_data = torch.empty(out_features, in_features)
        weight = Parameter(weight_data)
        kaiming_uniform_(weight_data, fan=in_features, nonlinearity="linear")
        self._register_parameter("weight", weight)

        if bias:
            bias_data = torch.empty(out_features)
            bias_param = Parameter(bias_data)
            zeros_(bias_data)
            self._register_parameter("bias", bias_param)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = x @ W^T + b.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        weight = self._parameters["weight"].data

        if not self._use_bias:
            return x @ weight.t()

        bias = self._parameters["bias"].data

        # addmm fuses the matmul and the bias add into one kernel, avoiding the
        # full-size intermediate that `x @ W.t() + b` allocates. It only accepts
        # 2-D inputs, so collapse the leading dims and restore them after.
        if x.dim() == 2:
            return torch.addmm(bias, x, weight.t())

        flat = x.reshape(-1, x.shape[-1])
        out = torch.addmm(bias, flat, weight.t())
        return out.view(*x.shape[:-1], weight.shape[0])

    def __repr__(self) -> str:
        return f"Linear(in={self._in_features}, out={self._out_features}, bias={self._use_bias})"
