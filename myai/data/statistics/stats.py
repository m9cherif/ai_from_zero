"""Dataset statistics: computes and logs dataset properties."""

import math
import time
from collections import Counter
from typing import List, Dict, Optional, Any, Tuple
from ...core.logging import logger


class DatasetStatistics:
    """Computes and reports comprehensive dataset statistics."""

    def __init__(self):
        self._reset()

    def _reset(self) -> None:
        self._num_samples = 0
        self._total_tokens = 0
        self._total_chars = 0
        self._seq_lengths: List[int] = []
        self._token_frequencies: Counter = Counter()
        self._vocab_coverage: float = 0.0
        self._duplication_rate: float = 0.0
        self._start_time: float = 0.0
        self._language_stats: Dict[str, int] = {}

    def begin(self) -> None:
        self._reset()
        self._start_time = time.time()

    def record_sample(self, token_ids: List[int], text_length: int) -> None:
        """Record statistics for a single sample."""
        self._num_samples += 1
        self._total_tokens += len(token_ids)
        self._total_chars += text_length
        self._seq_lengths.append(len(token_ids))
        for tid in token_ids:
            self._token_frequencies[tid] += 1

    def set_vocab_coverage(self, coverage: float) -> None:
        self._vocab_coverage = coverage

    def set_duplication_rate(self, rate: float) -> None:
        self._duplication_rate = rate

    def record_language(self, language: str) -> None:
        self._language_stats[language] = self._language_stats.get(language, 0) + 1

    def compute(self) -> Dict[str, Any]:
        """Compute and return all statistics."""
        elapsed = time.time() - self._start_time

        stats = {
            "num_samples": self._num_samples,
            "total_tokens": self._total_tokens,
            "total_chars": self._total_chars,
            "elapsed_seconds": round(elapsed, 2),
        }

        if self._num_samples > 0:
            stats["tokens_per_second"] = round(self._total_tokens / max(elapsed, 0.001), 2)

        if self._seq_lengths:
            sorted_lengths = sorted(self._seq_lengths)
            n = len(sorted_lengths)
            stats["seq_length_mean"] = round(sum(self._seq_lengths) / n, 2)
            stats["seq_length_median"] = sorted_lengths[n // 2]
            stats["seq_length_min"] = sorted_lengths[0]
            stats["seq_length_max"] = sorted_lengths[-1]
            stats["seq_length_p90"] = sorted_lengths[int(n * 0.9)]
            stats["seq_length_std"] = round(
                (sum((x - stats["seq_length_mean"]) ** 2 for x in self._seq_lengths) / n) ** 0.5,
                2,
            )

        if self._token_frequencies:
            stats["unique_tokens"] = len(self._token_frequencies)
            # Entropy
            total = sum(self._token_frequencies.values())
            entropy = -sum((c / total) * math.log2(c / total) for c in self._token_frequencies.values())
            stats["token_entropy"] = round(entropy, 4)

        if self._vocab_coverage:
            stats["vocab_coverage"] = round(self._vocab_coverage, 4)

        if self._duplication_rate:
            stats["duplication_rate"] = round(self._duplication_rate, 4)

        if self._language_stats:
            stats["languages"] = dict(self._language_stats)

        return stats

    def log_summary(self) -> None:
        """Log a comprehensive statistics summary."""
        stats = self.compute()
        logger.info("=" * 50)
        logger.info("Dataset Statistics Summary")
        logger.info("=" * 50)
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 50)
