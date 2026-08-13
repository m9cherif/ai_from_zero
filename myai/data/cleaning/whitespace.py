"""Whitespace standardization and normalization."""

import re


class WhitespaceNormalizer:
    """Standardizes whitespace characters and patterns."""

    @staticmethod
    def normalize(text: str, collapse_spaces: bool = True, strip_lines: bool = True) -> str:
        """Normalize whitespace in text."""
        if collapse_spaces:
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
        if strip_lines:
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)
        text = text.strip()
        return text

    @staticmethod
    def normalize_spaces(text: str) -> str:
        """Collapse multiple spaces into one."""
        return re.sub(r' {2,}', ' ', text)

    @staticmethod
    def normalize_line_breaks(text: str) -> str:
        """Standardize line breaks to \n."""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return re.sub(r'\n{3,}', '\n\n', text)

    def __call__(self, text: str) -> str:
        return self.normalize(text)
