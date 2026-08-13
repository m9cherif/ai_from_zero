"""Evaluation metrics for language model performance."""

import math
from typing import List, Dict, Optional, Any
import torch
from ..core.types import Tensor
from ..core.logging import logger


class PerplexityMetric:
    """Perplexity: exp(loss) - measures model uncertainty."""

    @staticmethod
    def compute(loss: float) -> float:
        return math.exp(min(loss, 100.0))

    @staticmethod
    def from_logits(logits: Tensor, targets: Tensor) -> float:
        """Compute perplexity directly from logits and targets."""
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return math.exp(min(loss.item(), 100.0))


class AccuracyMetric:
    """Token prediction accuracy."""

    @staticmethod
    def compute(logits: Tensor, targets: Tensor) -> float:
        """Compute token-level accuracy."""
        predictions = logits.argmax(dim=-1)
        mask = targets != -100
        correct = (predictions == targets) & mask
        total = mask.sum().item()
        if total == 0:
            return 0.0
        return correct.sum().item() / total


class TextGenerationMetrics:
    """Collection of text generation quality metrics."""

    @staticmethod
    def distinct_ngrams(text: str, n: int = 2) -> float:
        """Compute distinct n-gram ratio (diversity metric)."""
        words = text.split()
        if len(words) < n:
            return 1.0
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngrams.add(" ".join(words[i:i + n]))
        return len(ngrams) / max(len(words) - n + 1, 1)

    @staticmethod
    def avg_sentence_length(text: str) -> float:
        """Compute average sentence length in words."""
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    @staticmethod
    def repetition_rate(text: str, n: int = 4) -> float:
        """Compute n-gram repetition rate."""
        words = text.split()
        if len(words) < n + 1:
            return 0.0
        ngrams = {}
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        repeated = sum(1 for v in ngrams.values() if v > 1)
        return repeated / max(len(ngrams), 1)

    @staticmethod
    def compute_all(text: str) -> Dict[str, float]:
        """Compute all text generation metrics."""
        return {
            "distinct_1gram": TextGenerationMetrics.distinct_ngrams(text, 1),
            "distinct_2gram": TextGenerationMetrics.distinct_ngrams(text, 2),
            "avg_sentence_length": TextGenerationMetrics.avg_sentence_length(text),
            "repetition_rate": TextGenerationMetrics.repetition_rate(text, 4),
        }
