"""Weight initialization functions implemented from scratch.

These use PyTorch tensor operations but implement the
mathematical initialization schemes manually.
"""

import math
from typing import Optional
import torch
from ..core.types import Tensor


def _check_tensor(t: Tensor) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected Tensor, got {type(t)}")


def zeros_(tensor: Tensor) -> Tensor:
    """Fill tensor with zeros."""
    _check_tensor(tensor)
    with torch.no_grad():
        tensor.zero_()
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    """Fill tensor with ones."""
    _check_tensor(tensor)
    with torch.no_grad():
        tensor.fill_(1.0)
    return tensor


def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """Fill tensor with uniform distribution U(low, high)."""
    _check_tensor(tensor)
    with torch.no_grad():
        tensor.uniform_(low, high)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill tensor with normal distribution N(mean, std)."""
    _check_tensor(tensor)
    with torch.no_grad():
        tensor.normal_(mean, std)
    return tensor


def kaiming_uniform_(tensor: Tensor, fan: Optional[int] = None, nonlinearity: str = "relu") -> Tensor:
    """Kaiming (He) uniform initialization."""
    _check_tensor(tensor)
    if fan is None:
        fan = tensor.shape[0] if len(tensor.shape) >= 2 else tensor.numel()

    # He et al. 2015: Var = 2 / fan_in
    gain = {
        "relu": math.sqrt(2.0),
        "leaky_relu": math.sqrt(2.0 / (1 + 0.01 ** 2)),
        "tanh": 1.0,
        "sigmoid": 1.0,
        "linear": 1.0,
        "silu": math.sqrt(2.0),
        "gelu": math.sqrt(2.0),
    }.get(nonlinearity, 1.0)

    std = gain / math.sqrt(max(fan, 1))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier (Glorot) uniform initialization."""
    _check_tensor(tensor)
    if len(tensor.shape) < 2:
        fan_in = tensor.numel()
        fan_out = tensor.numel()
    else:
        fan_in = tensor.shape[1] if len(tensor.shape) == 2 else tensor.shape[1] * (tensor.shape[2] if len(tensor.shape) > 2 else 1)
        fan_out = tensor.shape[0] if len(tensor.shape) == 2 else tensor.shape[0] * (tensor.shape[2] if len(tensor.shape) > 2 else 1) // (tensor.shape[1] if len(tensor.shape) > 2 else 1)

    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier (Glorot) normal initialization."""
    _check_tensor(tensor)
    if len(tensor.shape) < 2:
        fan_in = tensor.numel()
        fan_out = tensor.numel()
    else:
        fan_in = tensor.shape[1] if len(tensor.shape) == 2 else tensor.shape[1] * (tensor.shape[2] if len(tensor.shape) > 2 else 1)
        fan_out = tensor.shape[0] if len(tensor.shape) == 2 else tensor.shape[0] * (tensor.shape[2] if len(tensor.shape) > 2 else 1) // (tensor.shape[1] if len(tensor.shape) > 2 else 1)

    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)
