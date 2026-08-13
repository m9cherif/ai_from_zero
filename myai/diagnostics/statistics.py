"""Parameter and gradient statistics for model analysis."""

from typing import Dict, List, Optional
import torch
from ..core.types import Tensor
from ..nn.module import Module
from ..core.logging import logger


class ParameterStatistics:
    """Computes and reports parameter and gradient statistics."""

    @staticmethod
    def compute_param_stats(model: Module) -> Dict[str, float]:
        """Compute statistics for all model parameters."""
        total_params = 0
        total_norm = 0.0
        max_abs = 0.0
        min_abs = float('inf')
        mean_abs = 0.0

        for param in model.parameters():
            data = param.data
            numel = data.numel()
            total_params += numel

            p_norm = data.norm().item()
            total_norm += p_norm ** 2

            p_max_abs = data.abs().max().item()
            max_abs = max(max_abs, p_max_abs)

            p_min_abs = data.abs().min().item()
            min_abs = min(min_abs, p_min_abs)

            mean_abs += data.abs().sum().item()

        total_norm = total_norm ** 0.5
        mean_abs /= max(total_params, 1)

        return {
            "total_parameters": total_params,
            "param_norm": total_norm,
            "param_max_abs": max_abs,
            "param_min_abs": min_abs if min_abs != float('inf') else 0.0,
            "param_mean_abs": mean_abs,
        }

    @staticmethod
    def compute_grad_stats(model: Module) -> Dict[str, float]:
        """Compute statistics for model gradients."""
        total_norm = 0.0
        max_abs = 0.0
        zero_grad = 0
        total_grads = 0
        max_grad_ratio = 0.0

        for param in model.parameters():
            if param.grad is None:
                continue

            grad = param.grad
            numel = grad.numel()
            total_grads += numel

            g_norm = grad.norm().item()
            total_norm += g_norm ** 2

            g_max_abs = grad.abs().max().item()
            max_abs = max(max_abs, g_max_abs)

            zero_grad += (grad == 0).sum().item()

            # Ratio of gradient norm to parameter norm
            p_norm = param.data.norm().item()
            if p_norm > 0:
                max_grad_ratio = max(max_grad_ratio, g_norm / p_norm)

        total_norm = total_norm ** 0.5

        return {
            "grad_norm": total_norm,
            "grad_max_abs": max_abs,
            "zero_grad_ratio": zero_grad / max(total_grads, 1) if total_grads > 0 else 0.0,
            "grad_param_ratio": max_grad_ratio,
        }

    @staticmethod
    def log_stats(model: Module, prefix: str = "") -> None:
        """Log comprehensive parameter and gradient statistics."""
        param_stats = ParameterStatistics.compute_param_stats(model)
        grad_stats = ParameterStatistics.compute_grad_stats(model)

        logger.info(f"{prefix}Parameter stats: {param_stats}")
        logger.info(f"{prefix}Gradient stats: {grad_stats}")
