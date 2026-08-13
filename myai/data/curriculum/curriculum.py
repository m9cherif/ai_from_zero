"""Curriculum learning: ordering data from simple to complex."""

import math
from typing import Callable, List, Optional, Tuple, Any
from ...core.logging import logger


class CurriculumScheduler:
    """Orders data from simple to complex based on configurable heuristics.

    Supports multiple difficulty metrics and annealing schedules.
    """

    def __init__(
        self,
        schedule: str = "linear",
        total_steps: int = 100000,
        initial_difficulty: float = 0.0,
        final_difficulty: float = 1.0,
    ):
        self._schedule = schedule
        self._total_steps = total_steps
        self._initial_difficulty = initial_difficulty
        self._final_difficulty = final_difficulty
        self._current_step = 0

    def get_difficulty(self, step: Optional[int] = None) -> float:
        """Get the current difficulty threshold based on training step."""
        s = step if step is not None else self._current_step
        progress = min(1.0, s / max(self._total_steps, 1))

        if self._schedule == "linear":
            difficulty = self._initial_difficulty + (self._final_difficulty - self._initial_difficulty) * progress
        elif self._schedule == "exponential":
            difficulty = self._initial_difficulty + (self._final_difficulty - self._initial_difficulty) * (1 - math.exp(-5 * progress))
        elif self._schedule == "logarithmic":
            difficulty = self._initial_difficulty + (self._final_difficulty - self._initial_difficulty) * math.log(1 + 9 * progress) / math.log(10)
        elif self._schedule == "step":
            difficulty = self._initial_difficulty if progress < 0.5 else self._final_difficulty
        else:
            difficulty = self._initial_difficulty

        return max(0.0, min(1.0, difficulty))

    def step(self) -> None:
        self._current_step += 1

    def score_text(self, text: str) -> float:
        """Compute a difficulty score for a text (higher = more complex)."""
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0

        # Average word length
        avg_word_len = sum(len(w) for w in words) / len(words)

        # Vocabulary diversity
        unique_words = len(set(w.lower() for w in words))
        diversity = unique_words / max(len(words), 1)

        # Sentence length (simple heuristic)
        sentences = text.count(".") + text.count("!") + text.count("?")
        avg_sent_len = len(words) / max(sentences, 1)

        # Composite score (normalized roughly)
        score = (avg_word_len / 20) * 0.3 + diversity * 0.4 + min(avg_sent_len / 100, 1) * 0.3
        return min(1.0, score)

    def should_include(self, text: str) -> bool:
        """Check if a text should be included at the current difficulty level."""
        score = self.score_text(text)
        difficulty = self.get_difficulty()
        return score <= difficulty

    def filter_by_difficulty(self, texts: List[str]) -> List[str]:
        """Filter texts based on current curriculum difficulty."""
        difficulty = self.get_difficulty()
        filtered = [t for t in texts if self.score_text(t) <= difficulty]
        logger.info(f"Curriculum filter: {len(texts)} -> {len(filtered)} (difficulty={difficulty:.3f})")
        return filtered

    def state_dict(self) -> dict:
        return {
            "current_step": self._current_step,
            "schedule": self._schedule,
            "total_steps": self._total_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self._current_step = state.get("current_step", 0)
        self._schedule = state.get("schedule", "linear")
        self._total_steps = state.get("total_steps", 100000)
