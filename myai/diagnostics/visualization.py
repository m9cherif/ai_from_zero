"""Visualization utilities for training metrics and model analysis."""

from typing import Dict, List, Optional, Tuple
from ..core.logging import logger


class MetricsVisualizer:
    """Generates text-based visualizations of training metrics."""

    @staticmethod
    def format_metrics_table(metrics: Dict[str, float], precision: int = 4) -> str:
        """Format metrics as a text table."""
        lines = ["-" * 50, f"{'Metric':<25} {'Value':<15}", "-" * 50]
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                lines.append(f"{key:<25} {value:.{precision}f}")
            else:
                lines.append(f"{key:<25} {value:<15}")
        lines.append("-" * 50)
        return "\n".join(lines)

    @staticmethod
    def progress_bar(current: int, total: int, bar_length: int = 30) -> str:
        """Create a text progress bar."""
        fraction = min(1.0, current / max(total, 1))
        filled = int(bar_length * fraction)
        bar = "█" * filled + "░" * (bar_length - filled)
        return f"|{bar}| {current}/{total} ({fraction * 100:.1f}%)"

    @staticmethod
    def loss_curve(losses: List[float], width: int = 60, height: int = 10) -> str:
        """Create a simple ASCII loss curve."""
        if not losses:
            return "[no data]"

        min_loss = min(losses)
        max_loss = max(losses)
        loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

        lines = []
        for h in range(height, 0, -1):
            threshold = min_loss + (h / height) * loss_range
            row = ""
            for loss in losses:
                row += "█" if loss >= threshold else " "
            lines.append(row)

        return "\n".join(lines)
