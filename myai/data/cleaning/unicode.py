"""Unicode normalization and cleanup."""

import unicodedata
import re
from typing import Optional


class UnicodeNormalizer:
    """Normalizes Unicode text to a consistent representation."""

    def __init__(self, form: str = "NFKC", remove_control: bool = True):
        if form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError(f"Unknown Unicode normalization form: {form}")
        self._form = form
        self._remove_control = remove_control

    def normalize(self, text: str) -> str:
        """Apply Unicode normalization."""
        text = unicodedata.normalize(self._form, text)
        if self._remove_control:
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xad]', '', text)
            text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u2064\ufeff]', '', text)
        return text

    def __call__(self, text: str) -> str:
        return self.normalize(text)
