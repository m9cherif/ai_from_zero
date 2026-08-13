"""Full text sanitization combining Unicode, whitespace, and character cleanup."""

from .unicode import UnicodeNormalizer
from .whitespace import WhitespaceNormalizer


class TextSanitizer:
    """Combined text sanitization pipeline."""

    def __init__(
        self,
        unicode_form: str = "NFKC",
        collapse_spaces: bool = True,
        strip_lines: bool = True,
        remove_control: bool = True,
    ):
        self._unicode = UnicodeNormalizer(form=unicode_form, remove_control=remove_control)
        self._whitespace = WhitespaceNormalizer()

    def sanitize(self, text: str) -> str:
        """Full sanitization pipeline."""
        text = self._unicode.normalize(text)
        text = self._whitespace.normalize(
            text,
            collapse_spaces=True,
            strip_lines=True,
        )
        return text

    def __call__(self, text: str) -> str:
        return self.sanitize(text)
