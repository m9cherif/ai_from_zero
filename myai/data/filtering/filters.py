"""Configurable text filters for quality, safety, and language."""

import re
from typing import List, Optional, Set, Callable, Tuple


class TextFilter:
    """Configurable text filtering with chainable operations."""

    def __init__(self):
        self._filters: List[Callable[[str], bool]] = []

    def add_filter(self, filter_fn: Callable[[str], bool], name: str = "") -> "TextFilter":
        self._filters.append(filter_fn)
        return self

    def filter(self, text: str) -> Tuple[bool, List[str]]:
        """Apply all filters. Returns (pass, list of failed filter reasons)."""
        failures = []
        for fn in self._filters:
            if not fn(text):
                name = getattr(fn, "__name__", str(fn))
                failures.append(name)
        return len(failures) == 0, failures

    def filter_batch(self, texts: List[str]) -> List[Tuple[str, bool, List[str]]]:
        """Filter a batch of texts."""
        return [(t, *self.filter(t)) for t in texts]

    @staticmethod
    def min_length_filter(min_chars: int) -> Callable[[str], bool]:
        return lambda text: len(text) >= min_chars

    @staticmethod
    def max_length_filter(max_chars: int) -> Callable[[str], bool]:
        return lambda text: len(text) <= max_chars

    @staticmethod
    def min_words_filter(min_words: int) -> Callable[[str], bool]:
        return lambda text: len(text.split()) >= min_words

    @staticmethod
    def max_words_filter(max_words: int) -> Callable[[str], bool]:
        return lambda text: len(text.split()) <= max_words

    @staticmethod
    def language_filter(allowed_languages: Set[str]) -> Callable[[str], bool]:
        """Simple language detection by checking character scripts."""
        def _check(text: str) -> bool:
            if "english" in allowed_languages or "en" in allowed_languages:
                # Check if mostly ASCII
                ascii_chars = sum(1 for c in text if ord(c) < 128)
                if text and ascii_chars / len(text) < 0.5:
                    return False
            return True
        return _check

    @staticmethod
    def profanity_filter(profanity_list: Optional[Set[str]] = None) -> Callable[[str], bool]:
        """Filter text containing profanity (configurable word list)."""
        bad_words = profanity_list or set()
        if not bad_words:
            return lambda text: True

        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in bad_words) + r')\b',
            re.IGNORECASE,
        )
        return lambda text: pattern.search(text) is None

    @staticmethod
    def boilerplate_filter(max_boilerplate_ratio: float = 0.3) -> Callable[[str], bool]:
        """Filter text with excessive boilerplate patterns."""
        patterns = [
            r'copyright\s+\d{4}',
            r'all rights reserved',
            r'terms\s+(and|of)\s+(service|use|conditions)',
            r'privacy\s+policy',
            r'click\s+here',
            r'subscribe',
            r'newsletter',
            r'cookie',
            r'advertisement',
        ]
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

        def _check(text: str) -> bool:
            if not text:
                return True
            matches = sum(1 for p in compiled if p.search(text))
            ratio = matches / max(len(patterns), 1)
            return ratio <= max_boilerplate_ratio

        return _check

    @staticmethod
    def url_ratio_filter(max_url_ratio: float = 0.2) -> Callable[[str], bool]:
        url_pattern = re.compile(r'https?://\S+')

        def _check(text: str) -> bool:
            urls = url_pattern.findall(text)
            if not urls:
                return True
            url_length = sum(len(u) for u in urls)
            return url_length / max(len(text), 1) <= max_url_ratio

        return _check

    @staticmethod
    def ellipsis_ratio_filter(max_ratio: float = 0.1) -> Callable[[str], bool]:
        def _check(text: str) -> bool:
            ellipsis_count = text.count('...') + text.count('…')
            return ellipsis_count / max(len(text.split()), 1) <= max_ratio

        return _check
