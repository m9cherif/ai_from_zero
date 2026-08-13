"""Parameter management for manual neural network implementation."""

import torch
from typing import List, Optional, Any, Iterator
from ..core.types import Tensor, Shape


class Parameter:
    """A trainable parameter tensor.

    Wraps a tensor and tracks whether it requires gradients.
    This is the basic building block of all learnable components.
    """

    def __init__(self, data: Tensor, requires_grad: bool = True):
        self._data = data
        self._data.requires_grad_(requires_grad)

    @property
    def data(self) -> Tensor:
        return self._data

    @data.setter
    def data(self, value: Tensor) -> None:
        self._data = value

    @property
    def raw_data(self) -> Tensor:
        return self._data.data

    @property
    def grad(self) -> Optional[Tensor]:
        return self._data.grad

    @property
    def requires_grad(self) -> bool:
        return self._data.requires_grad

    @property
    def shape(self) -> Shape:
        return tuple(self._data.shape)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self):
        return self._data.device

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear the gradient.

        set_to_none=True releases the gradient buffer instead of filling it
        with zeros. This is faster and is what modern training loops do: the
        next backward() allocates a fresh buffer rather than accumulating
        into a zeroed one.
        """
        if set_to_none:
            self._data.grad = None
        elif self._data.grad is not None:
            self._data.grad.zero_()

    def detach(self) -> Tensor:
        return self._data.detach()

    def numpy(self):
        return self._data.detach().cpu().numpy()

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Move/cast the parameter, preserving its leaf status.

        A plain ``tensor.to(...)`` is an autograd *operation*: its result is a
        non-leaf tensor, and .grad is never populated on it. Moving a model to
        GPU that way produces a model that runs but silently never learns.
        We therefore detach, move, and re-mark the result as a leaf.
        """
        if device is None and dtype is None:
            return self

        target_device = torch.device(device) if device is not None else self._data.device
        target_dtype = dtype if dtype is not None else self._data.dtype

        # Nothing to do - avoid needlessly rebuilding the autograd leaf, which
        # would break parameters shared between modules (e.g. tied embeddings).
        if self._data.device == target_device and self._data.dtype == target_dtype:
            return self

        requires_grad = self._data.requires_grad
        old_grad = self._data.grad

        moved = self._data.detach().to(device=target_device, dtype=target_dtype)
        moved.requires_grad_(requires_grad)

        if old_grad is not None:
            moved.grad = old_grad.detach().to(device=target_device, dtype=target_dtype)

        self._data = moved
        return self

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, requires_grad={self._data.requires_grad})"


class ParameterList:
    """A list of parameters with convenient access methods."""

    def __init__(self, parameters: Optional[List[Parameter]] = None):
        self._parameters = list(parameters) if parameters else []

    def append(self, param: Parameter) -> None:
        self._parameters.append(param)

    def extend(self, params: List[Parameter]) -> None:
        self._parameters.extend(params)

    def __getitem__(self, idx: int) -> Parameter:
        return self._parameters[idx]

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._parameters:
            p.zero_grad(set_to_none=set_to_none)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        for p in self._parameters:
            p.to(device=device, dtype=dtype)
        return self

    def parameters(self) -> List[Parameter]:
        return self._parameters
