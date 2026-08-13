"""Optimizers implemented from scratch.

SGD, Adam, and AdamW written with basic tensor operations, but using PyTorch's
``_foreach_*`` primitives so a single update touches every parameter tensor at
once. A model with hundreds of small tensors otherwise spends most of the step
in Python loop and kernel-launch overhead rather than in arithmetic; the fused
form is typically several times faster for the same math.
"""

from typing import Iterable, Optional, Dict, Any, Callable, List, Tuple, Union
from collections import defaultdict
import torch
from ..core.types import Tensor
from ..nn.parameter import Parameter
from ..nn.module import Module
from ..core.logging import logger


ParamsT = Union[Iterable[Parameter], List[Dict[str, Any]]]


def build_param_groups(
    model: Module,
    weight_decay: float = 0.1,
    no_decay_ndim: int = 2,
) -> List[Dict[str, Any]]:
    """Split parameters into decayed and non-decayed groups.

    Standard practice for transformers: apply weight decay only to matrices.
    Biases, LayerNorm/RMSNorm gains, and other 1-D parameters are excluded -
    decaying them shrinks the network's scale and normalization behaviour
    rather than acting as the intended regularizer.
    """
    decay: List[Parameter] = []
    no_decay: List[Parameter] = []

    for _, param in model.named_parameters():
        if param.data.dim() < no_decay_ndim:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


class Optimizer:
    """Base optimizer class."""

    def __init__(self, parameters: ParamsT, defaults: Dict[str, Any]):
        self._defaults = dict(defaults)
        self._param_groups: List[Dict[str, Any]] = []
        self._state: Dict[int, Dict[str, Any]] = {}
        self._step_count = 0

        parameters = list(parameters)
        if not parameters:
            raise ValueError("Optimizer received an empty parameter list")

        if isinstance(parameters[0], dict):
            for group in parameters:
                merged = dict(defaults)
                merged.update(group)
                merged["params"] = list(group["params"])
                self._param_groups.append(merged)
        else:
            group = dict(defaults)
            group["params"] = parameters
            self._param_groups.append(group)

        self._parameter_list: List[Parameter] = [
            p for group in self._param_groups for p in group["params"]
        ]

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        return self._param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients.

        Releasing the buffers (the default) is faster than zeroing them and lets
        backward() write into a fresh allocation.
        """
        for p in self._parameter_list:
            p.zero_grad(set_to_none=set_to_none)

    @staticmethod
    def _bucket_by_device_dtype(
        params: List[Parameter],
    ) -> Dict[Tuple[torch.device, torch.dtype], List[Parameter]]:
        """Group tensors so each _foreach_ call sees one device and dtype."""
        buckets: Dict[Tuple[torch.device, torch.dtype], List[Parameter]] = defaultdict(list)
        for p in params:
            buckets[(p.data.device, p.data.dtype)].append(p)
        return buckets

    def step(self) -> None:
        """Perform a single optimization step."""
        self._step_count += 1
        for group in self._param_groups:
            active = [p for p in group["params"] if p.grad is not None]
            if not active:
                continue
            group["step"] = group.get("step", 0) + 1
            for bucket in self._bucket_by_device_dtype(active).values():
                self._update_group(bucket, group)

    def _update_group(self, params: List[Parameter], group: Dict[str, Any]) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Serialize optimizer state.

        This is a method (not a property) to match the torch convention and the
        way every call site - checkpointing included - invokes it.
        """
        packed = {}
        for i, p in enumerate(self._parameter_list):
            entry = self._state.get(id(p))
            if entry is None:
                continue
            packed[str(i)] = {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in entry.items()
            }
        return {
            "version": 2,
            "step": self._step_count,
            "defaults": self._defaults,
            "group_steps": [g.get("step", 0) for g in self._param_groups],
            "group_hyperparams": [
                {k: v for k, v in g.items() if k != "params"} for g in self._param_groups
            ],
            "state": packed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore optimizer state, tolerating checkpoints from the old format."""
        self._step_count = state_dict.get("step", 0)

        for group, saved in zip(self._param_groups, state_dict.get("group_hyperparams", [])):
            for key, value in saved.items():
                if key != "params":
                    group[key] = value

        for group, step in zip(self._param_groups, state_dict.get("group_steps", [])):
            group["step"] = step

        packed = state_dict.get("state")
        if packed is None:
            # Version 1 layout: entries stored as top-level "param_{i}" keys.
            packed = {
                key.split("_", 1)[1]: value
                for key, value in state_dict.items()
                if key.startswith("param_")
            }

        for i, p in enumerate(self._parameter_list):
            entry = packed.get(str(i))
            if entry is None:
                continue
            self._state[id(p)] = {
                k: (v.to(device=p.device, dtype=p.dtype) if torch.is_tensor(v) else v)
                for k, v in entry.items()
            }

    def _moment_buffers(
        self, params: List[Parameter], keys: Tuple[str, ...]
    ) -> Tuple[List[Tensor], ...]:
        """Fetch (creating on first use) the per-parameter moment buffers."""
        out: Tuple[List[Tensor], ...] = tuple([] for _ in keys)
        for p in params:
            state = self._state.get(id(p))
            if state is None:
                state = {k: torch.zeros_like(p.data) for k in keys}
                self._state[id(p)] = state
            for slot, key in zip(out, keys):
                slot.append(state[key])
        return out


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum.

    param = param - lr * grad (optionally with momentum and Nesterov).
    """

    def __init__(
        self,
        parameters: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay, "nesterov": nesterov}
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def _update_group(self, params: List[Parameter], group: Dict[str, Any]) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        nesterov = group["nesterov"]

        datas = [p.data for p in params]
        grads = [p.grad for p in params]

        if weight_decay != 0:
            # Coupled (L2) decay: fold the penalty into the gradient.
            grads = torch._foreach_add(grads, datas, alpha=weight_decay)

        if momentum != 0:
            (bufs,) = self._moment_buffers(params, ("momentum_buffer",))
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)
            if nesterov:
                grads = torch._foreach_add(grads, bufs, alpha=momentum)
            else:
                grads = bufs

        torch._foreach_add_(datas, grads, alpha=-lr)


class _AdamBase(Optimizer):
    """Shared Adam machinery; subclasses only differ in how decay is applied."""

    _decoupled_decay = False

    @torch.no_grad()
    def _update_group(self, params: List[Parameter], group: Dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        step = group["step"]

        datas = [p.data for p in params]
        grads = [p.grad for p in params]

        if weight_decay != 0:
            if self._decoupled_decay:
                # AdamW: shrink the weights directly, independent of the
                # gradient's second-moment scaling.
                torch._foreach_mul_(datas, 1.0 - lr * weight_decay)
            else:
                grads = torch._foreach_add(grads, datas, alpha=weight_decay)

        exp_avgs, exp_avg_sqs = self._moment_buffers(params, ("exp_avg", "exp_avg_sq"))

        # m = beta1*m + (1-beta1)*g   (lerp is the fused form)
        torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
        # v = beta2*v + (1-beta2)*g^2
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        step_size = lr / bias_correction1

        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denom, bias_correction2 ** 0.5)
        torch._foreach_add_(denom, eps)

        torch._foreach_addcdiv_(datas, exp_avgs, denom, value=-step_size)


class Adam(_AdamBase):
    """Adam optimizer (Kingma & Ba, 2015).

    Maintains first and second moment estimates with bias correction.
    Weight decay is coupled (added to the gradient, i.e. L2 regularization).
    """

    _decoupled_decay = False

    def __init__(
        self,
        parameters: ParamsT,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(parameters, defaults)


class AdamW(_AdamBase):
    """AdamW optimizer (Loshchilov & Hutter, 2019).

    Adam with decoupled weight decay - the decay is applied to the weights
    directly rather than through the adaptive gradient scaling, which is what
    makes it the default for transformer training.
    """

    _decoupled_decay = True

    def __init__(
        self,
        parameters: ParamsT,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(parameters, defaults)
