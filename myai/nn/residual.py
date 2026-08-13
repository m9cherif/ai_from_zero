"""Residual connection module."""

from .module import Module
from ..core.types import Tensor


class ResidualConnection(Module):
    """Residual connection with optional pre/post normalization.

    Implements: output = norm(sub_layer(x)) + x (pre-norm)
    or: output = x + sub_layer(x) followed by norm (post-norm)

    Pre-norm is the default for most modern LLMs.
    """

    def __init__(self, sub_layer: Module, norm_layer: Module = None, pre_norm: bool = True):
        super().__init__()
        self._sub_layer = sub_layer
        self._norm_layer = norm_layer
        self._pre_norm = pre_norm

        self._register_module("sub_layer", sub_layer)
        if norm_layer is not None:
            self._register_module("norm", norm_layer)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Apply residual connection.

        For pre-norm: output = x + sub_layer(norm(x))
        For post-norm: output = norm(x + sub_layer(x))
        """
        if self._pre_norm:
            # Pre-norm: normalize before the sub-layer
            if self._norm_layer is not None:
                normalized = self._norm_layer(x)
            else:
                normalized = x

            sub_output = self._sub_layer(normalized, *args, **kwargs)
            return x + sub_output
        else:
            # Post-norm: normalize after the residual addition
            sub_output = self._sub_layer(x, *args, **kwargs)
            residual = x + sub_output

            if self._norm_layer is not None:
                residual = self._norm_layer(residual)

            return residual
