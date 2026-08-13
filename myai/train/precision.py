"""Mixed precision training: autocast plus a hand-written dynamic loss scaler."""

import contextlib
from typing import Optional, Iterable, List
import torch
from ..core.types import Tensor
from ..core.logging import logger
from ..nn.parameter import Parameter

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class DynamicLossScaler:
    """Dynamic loss scaling for fp16 training.

    fp16's smallest normal magnitude is ~6e-5, so gradients below that flush to
    zero and the model silently stops learning. Multiplying the loss by a large
    constant shifts them into representable range; the gradients are divided
    back out before the optimizer sees them.

    The scale is adapted at runtime: halve it whenever a step overflows to
    inf/NaN (and skip that step), double it after a long clean stretch.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._good_steps = 0

    @property
    def scale(self) -> float:
        return self._scale

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * self._scale

    def unscale_(self, parameters: Iterable[Parameter]) -> bool:
        """Divide gradients by the current scale. Returns True if any is non-finite."""
        grads: List[Tensor] = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return False

        torch._foreach_mul_(grads, 1.0 / self._scale)

        # One reduction over the stacked per-tensor norms: a non-finite value
        # anywhere makes the total non-finite.
        total = torch.stack([g.norm() for g in grads]).sum()
        return not bool(torch.isfinite(total))

    def update(self, found_inf: bool) -> None:
        """Adapt the scale after a step."""
        if found_inf:
            self._scale = max(self._scale * self._backoff_factor, 1.0)
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps >= self._growth_interval:
                self._scale *= self._growth_factor
                self._good_steps = 0

    def state_dict(self) -> dict:
        return {
            "scale": self._scale,
            "good_steps": self._good_steps,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
        }

    def load_state_dict(self, state: dict) -> None:
        self._scale = state.get("scale", self._scale)
        self._good_steps = state.get("good_steps", 0)
        self._growth_factor = state.get("growth_factor", self._growth_factor)
        self._backoff_factor = state.get("backoff_factor", self._backoff_factor)
        self._growth_interval = state.get("growth_interval", self._growth_interval)


class MixedPrecisionManager:
    """Manages mixed precision training with gradient scaling.

    Runs the forward pass in fp16/bf16 while master weights stay fp32. bf16 has
    fp32's exponent range, so it needs no loss scaling and is preferred wherever
    the hardware supports it.
    """

    def __init__(
        self,
        enabled: bool = False,
        dtype: str = "bfloat16",
        device_type: str = "cuda",
    ):
        device_type = str(device_type)
        self._device_type = "cuda" if device_type.startswith("cuda") else device_type
        self._dtype_name = dtype
        self._amp_dtype = _DTYPES.get(dtype, torch.bfloat16)
        self._scaler: Optional[DynamicLossScaler] = None
        self._found_inf = False

        if enabled and self._device_type == "cpu" and self._amp_dtype == torch.float16:
            logger.warning("float16 autocast is not supported on CPU; using bfloat16")
            self._amp_dtype = torch.bfloat16
            self._dtype_name = "bfloat16"

        if enabled and self._amp_dtype not in (torch.float16, torch.bfloat16):
            logger.warning(
                f"Mixed precision requested with dtype={dtype}; disabling (needs fp16 or bf16)"
            )
            enabled = False

        self._enabled = enabled

        if enabled:
            if self._amp_dtype == torch.float16:
                self._scaler = DynamicLossScaler()
            logger.info(
                f"Mixed precision enabled: dtype={self._dtype_name}, device={self._device_type}, "
                f"loss_scaling={self._scaler is not None}"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dtype(self) -> torch.dtype:
        return self._amp_dtype

    def get_forward_context(self):
        """Return the autocast context manager for the forward pass."""
        if not self._enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._device_type, dtype=self._amp_dtype)

    def scale_loss(self, loss: Tensor) -> Tensor:
        if self._scaler is not None:
            return self._scaler.scale_loss(loss)
        return loss

    def unscale_gradients(self, optimizer) -> None:
        """Undo loss scaling so clipping and the update see true magnitudes."""
        if self._scaler is None or optimizer is None:
            return
        params = [p for group in optimizer.param_groups for p in group["params"]]
        self._found_inf = self._scaler.unscale_(params)

    def step_optimizer(self, optimizer, closure=None) -> bool:
        """Step the optimizer. Returns False when the step was skipped for inf/NaN."""
        if self._scaler is None:
            if optimizer is not None:
                optimizer.step()
            return True

        stepped = not self._found_inf
        if stepped and optimizer is not None:
            optimizer.step()

        self._scaler.update(self._found_inf)
        self._found_inf = False
        return stepped

    def state_dict(self) -> dict:
        return self._scaler.state_dict() if self._scaler is not None else {}

    def load_state_dict(self, state_dict: dict) -> None:
        if self._scaler is not None and state_dict:
            self._scaler.load_state_dict(state_dict)
