"""Performance profiling tools."""

import time
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict
from ..core.logging import logger


class PerformanceProfiler:
    """Simple performance profiler for measuring execution times."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._current: Dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing an operation."""
        if not self._enabled:
            return
        self._current[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing an operation and return duration."""
        if not self._enabled or name not in self._current:
            return 0.0
        duration = time.perf_counter() - self._current[name]
        self._timers[name].append(duration)
        del self._current[name]
        return duration

    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get timing statistics for a named operation."""
        if name not in self._timers or not self._timers[name]:
            return None
        times = self._timers[name]
        total = sum(times)
        count = len(times)
        return {
            "count": count,
            "total": total,
            "mean": total / count,
            "min": min(times),
            "max": max(times),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        return {
            name: self.get_stats(name)
            for name in self._timers
            if self.get_stats(name) is not None
        }

    def log_summary(self) -> None:
        """Log a summary of all profiling results."""
        stats = self.get_all_stats()
        if not stats:
            logger.info("No profiling data collected")
            return

        logger.info("=" * 60)
        logger.info("Performance Profile Summary")
        logger.info(f"{'Operation':<25} {'Count':<8} {'Total (s)':<10} {'Mean (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
        logger.info("-" * 60)

        for name, stat in sorted(stats.items()):
            logger.info(
                f"{name:<25} {stat['count']:<8} {stat['total']:<10.3f} "
                f"{stat['mean']*1000:<10.3f} {stat['min']*1000:<10.3f} {stat['max']*1000:<10.3f}"
            )
        logger.info("=" * 60)

    def reset(self) -> None:
        """Clear all profiling data."""
        self._timers.clear()
        self._current.clear()
