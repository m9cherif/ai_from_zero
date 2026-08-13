"""Quality scoring and filtering of text data."""

import re
import math
from typing import Optional, Callable, List, Tuple


class QualityScorer:
    """Computes quality scores for text samples."""

    @staticmethod
    def character_repetition_ratio(text: str) -> float:
        """Ratio of repeated character n-grams (high = likely low quality)."""
        if len(text) < 10:
            return 0.0
        for n in [3, 5]:
            ngrams = [text[i:i+n] for i in range(len(text)-n)]
            if ngrams:
                unique = len(set(ngrams))
                ratio = 1.0 - (unique / len(ngrams))
                if ratio > 0.5:
                    return ratio
        return 0.0

    @staticmethod
    def line_repetition_ratio(text: str) -> float:
        """Ratio of duplicated lines."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 2:
            return 0.0
        unique_lines = len(set(lines))
        return 1.0 - (unique_lines / len(lines))

    @staticmethod
    def symbol_to_word_ratio(text: str) -> float:
        """Ratio of non-alphanumeric characters to words."""
        words = text.split()
        if not words:
            return 1.0
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return symbols / max(len(text), 1)

    @staticmethod
    def avg_word_length(text: str) -> float:
        """Average word length."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    @staticmethod
    def perplexity_estimate(text: str) -> float:
        """Crude perplexity estimate based on character distribution entropy."""
        if not text:
            return float('inf')
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        total = len(text)
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
        return 2.0 ** entropy

    @staticmethod
    def compute_score(text: str) -> float:
        """Compute an overall quality score between 0 (low) and 1 (high)."""
        if not text or len(text) < 20:
            return 0.0

        score = 1.0

        # Penalize excessive character repetition
        char_rep = QualityScorer.character_repetition_ratio(text)
        score -= char_rep * 0.3

        # Penalize line repetition
        line_rep = QualityScorer.line_repetition_ratio(text)
        score -= line_rep * 0.3

        # Penalize too many symbols
        symbol_ratio = QualityScorer.symbol_to_word_ratio(text)
        if symbol_ratio > 0.5:
            score -= (symbol_ratio - 0.5) * 0.3

        # Penalize very short or very long words
        avg_wl = QualityScorer.avg_word_length(text)
        if avg_wl < 2 or avg_wl > 20:
            score -= 0.1

        return max(0.0, min(1.0, score))


class QualityFilter:
    """Filters text based on configurable quality thresholds."""

    def __init__(
        self,
        min_score: float = 0.3,
        min_length: int = 50,
        max_length: int = 100000,
        max_char_repetition: float = 0.3,
        max_line_repetition: float = 0.3,
        max_symbol_ratio: float = 0.5,
    ):
        self._scorer = QualityScorer()
        self._min_score = min_score
        self._min_length = min_length
        self._max_length = max_length
        self._max_char_repetition = max_char_repetition
        self._max_line_repetition = max_line_repetition
        self._max_symbol_ratio = max_symbol_ratio

    def is_quality(self, text: str) -> Tuple[bool, float]:
        """Check if text meets quality thresholds. Returns (pass, score)."""
        if len(text) < self._min_length:
            return False, 0.0
        if len(text) > self._max_length:
            return False, 0.0

        score = self._scorer.compute_score(text)

        if score < self._min_score:
            return False, score

        if self._scorer.character_repetition_ratio(text) > self._max_char_repetition:
            return False, score

        if self._scorer.line_repetition_ratio(text) > self._max_line_repetition:
            return False, score

        if self._scorer.symbol_to_word_ratio(text) > self._max_symbol_ratio:
            return False, score

        return True, score

    def filter(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Filter a list of texts, returning (text, score) for passing ones."""
        results = []
        for text in texts:
            passed, score = self.is_quality(text)
            if passed:
                results.append((text, score))
        return results
