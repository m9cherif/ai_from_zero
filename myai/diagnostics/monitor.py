"""Runtime monitoring of training metrics."""

import time
from typing import Dict, List, Optional, Any
from collections import deque
from ..core.logging import logger


class TrainingMonitor:
    """Monitors and reports training metrics in real-time."""

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._metrics: Dict[str, deque] = {}
        self._start_time: float = 0.0
        self._step_times: deque = deque(maxlen=window_size)

    def start(self) -> None:
        """Start the monitoring timer."""
        self._start_time = time.time()

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self._window_size)
        self._metrics[name].append(value)

    def record_step_time(self) -> None:
        """Record the time for a single training step."""
        self._step_times.append(time.time())

    def get_metric(self, name: str, average: bool = True) -> Optional[float]:
        """Get a metric value (latest or average)."""
        if name not in self._metrics or not self._metrics[name]:
            return None
        if average:
            return sum(self._metrics[name]) / len(self._metrics[name])
        return self._metrics[name][-1]

    def get_all_metrics(self, average: bool = True) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        result = {}
        for name, values in self._metrics.items():
            if values:
                if average:
                    result[name] = sum(values) / len(values)
                else:
                    result[name] = values[-1]
        return result

    @property
    def elapsed_time(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    @property
    def steps_per_second(self) -> float:
        if len(self._step_times) < 2:
            return 0.0
        times = list(self._step_times)
        if len(times) >= 2:
            elapsed = times[-1] - times[0]
            return (len(times) - 1) / max(elapsed, 0.001)
        return 0.0
