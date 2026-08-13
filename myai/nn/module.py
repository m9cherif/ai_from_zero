"""Base module class for all neural network components.

Provides parameter registration, buffer registration, device management,
training/evaluation mode, and state dict serialization.
"""

from typing import Dict, Iterator, List, Optional, Tuple, Any, Set
import torch
from .parameter import Parameter
from ..core.types import Tensor, Device


class Module:
    """Base class for all neural network modules.

    Every component (Linear, Embedding, Attention, etc.) inherits from this.
    Provides automatic parameter tracking, device management, and state dict.
    """

    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._buffers: Dict[str, Tensor] = {}
        self._modules: Dict[str, "Module"] = {}
        self._training: bool = True
        self._name: str = ""

    def _register_parameter(self, name: str, param: Parameter) -> None:
        self._parameters[name] = param

    def _register_module(self, name: str, module: "Module") -> None:
        self._modules[name] = module
        module._name = name

    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        """Register a non-trainable tensor that still follows the module's device.

        Buffers (RoPE tables, causal masks, ...) are not parameters: they carry
        no gradient and are not optimized, but they must live on the same device
        as the parameters or every forward pass pays a host->device copy.
        """
        if not hasattr(self, "_buffers"):
            self._buffers = {}
        self._buffers[name] = tensor

    def get_buffer(self, name: str) -> Optional[Tensor]:
        return self._buffers.get(name)

    def parameters(self, recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Parameter]:
        """Yield all parameters in the module.

        Shared parameters (e.g. tied input/output embeddings) are yielded once,
        so optimizers do not apply the same update twice.
        """
        for _, param in self.named_parameters(recurse=recurse, remove_duplicate=remove_duplicate):
            yield param

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
        _memo: Optional[Set[int]] = None,
    ) -> Iterator[Tuple[str, Parameter]]:
        """Yield (name, parameter) pairs."""
        if _memo is None:
            _memo = set()

        for name, param in self._parameters.items():
            if remove_duplicate:
                if id(param) in _memo:
                    continue
                _memo.add(id(param))
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param

        if recurse:
            for module_name, module in self._modules.items():
                child_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_parameters(
                    prefix=child_prefix,
                    recurse=True,
                    remove_duplicate=remove_duplicate,
                    _memo=_memo,
                )

    def modules(self) -> Iterator["Module"]:
        """Yield all submodules."""
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        """Yield (name, module) pairs for this module and all descendants."""
        yield prefix, self
        for module_name, module in self._modules.items():
            child_prefix = f"{prefix}.{module_name}" if prefix else module_name
            yield from module.named_modules(prefix=child_prefix)

    def train(self, mode: bool = True) -> "Module":
        """Set training mode."""
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode."""
        return self.train(False)

    @property
    def training(self) -> bool:
        return self._training

    def to(self, device: Optional[Device] = None, dtype: Optional[torch.dtype] = None) -> "Module":
        """Move all parameters and buffers to device/dtype."""
        for param in self.parameters():
            param.to(device=device, dtype=dtype)
        for module in self.modules():
            for name, buf in getattr(module, "_buffers", {}).items():
                if buf is None:
                    continue
                # Integer/bool buffers (masks, position ids) keep their dtype.
                buf_dtype = dtype if (dtype is not None and buf.is_floating_point()) else None
                module._buffers[name] = buf.to(device=device, dtype=buf_dtype)
        return self

    def cuda(self) -> "Module":
        return self.to(device=torch.device("cuda"))

    def cpu(self) -> "Module":
        return self.to(device=torch.device("cpu"))

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear all parameter gradients."""
        for param in self.parameters():
            param.zero_grad(set_to_none=set_to_none)

    def state_dict(self, prefix: str = "") -> Dict[str, Tensor]:
        """Serialize module state to a dictionary of tensors.

        Buffers are deliberately excluded: every buffer in this codebase is a
        derived table (RoPE frequencies, causal masks) that is recomputed from
        the config on load, so persisting them would only bloat checkpoints and
        break loading at a different sequence length.
        """
        state = {}
        for name, param in self._parameters.items():
            state[f"{prefix}{name}"] = param.data
        for module_name, module in self._modules.items():
            state.update(module.state_dict(prefix=f"{prefix}{module_name}."))
        return state

    def load_state_dict(self, state_dict: Dict[str, Tensor], prefix: str = "", strict: bool = True) -> None:
        """Load module state from a dictionary of tensors."""
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            if key in state_dict:
                incoming = state_dict[key]
                if tuple(incoming.shape) != tuple(param.data.shape):
                    raise ValueError(
                        f"Shape mismatch for '{key}': checkpoint has {tuple(incoming.shape)}, "
                        f"model expects {tuple(param.data.shape)}"
                    )
                with torch.no_grad():
                    param.data.copy_(incoming.to(device=param.device, dtype=param.dtype))
            elif strict:
                raise KeyError(f"Missing key '{key}' in state_dict")

        for module_name, module in self._modules.items():
            module.load_state_dict(state_dict, prefix=f"{prefix}{module_name}.", strict=strict)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters (shared parameters counted once)."""
        total = 0
        for param in self.parameters():
            if not trainable_only or param.data.requires_grad:
                total += param.data.numel()
        return total

    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.num_parameters()})"
