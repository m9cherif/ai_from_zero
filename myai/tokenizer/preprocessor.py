"""Text preprocessing for tokenization: normalization, cleaning, splitting."""

import re
import unicodedata
from typing import List, Optional

# Word-boundary marker (U+2581, "lower one eighth block"), the same convention
# SentencePiece uses. Spaces are rewritten to this character before tokenizing,
# so whitespace is carried *inside* tokens and survives the decode round trip.
# A plain str.split() would discard it, which is why the old encoder could not
# reconstruct spaces.
WORD_BOUNDARY = "▁"

# Pretokenizer: runs of boundary markers plus the following word, bare words, or
# runs of other whitespace (newlines, tabs). Merges are only ever learned within
# one pretoken, never across a word boundary.
_PRETOKEN_PATTERN = re.compile(
    rf"{WORD_BOUNDARY}+[^{WORD_BOUNDARY}\s]*|[^{WORD_BOUNDARY}\s]+|\s+"
)


class TextPreprocessor:
    """Handles text normalization before tokenization."""

    @staticmethod
    def normalize_unicode(text: str, form: str = "NFKC") -> str:
        """Normalize Unicode to a standard form."""
        return unicodedata.normalize(form, text)

    @staticmethod
    def strip_control_characters(text: str) -> str:
        """Remove control characters except newlines, tabs, and carriage returns."""
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace characters into single spaces."""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def remove_invalid_chars(text: str) -> str:
        """Remove characters that are invalid or problematic."""
        return ''.join(c for c in text if c == '\n' or c == '\t' or c == '\r' or (
            unicodedata.category(c) not in ('Cc', 'Cf', 'Cs', 'Cn')
        ))

    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase (optional, use with caution)."""
        return text.lower()

    @staticmethod
    def split_on_whitespace(text: str) -> List[str]:
        """Split text into whitespace-delimited tokens."""
        return text.split()

    @staticmethod
    def split_on_punctuation(text: str) -> List[str]:
        """Split text on punctuation boundaries while keeping punctuation."""
        tokens = []
        for word in text.split():
            current = ""
            for char in word:
                if unicodedata.category(char).startswith("P"):
                    if current:
                        tokens.append(current)
                        current = ""
                    tokens.append(char)
                else:
                    current += char
            if current:
                tokens.append(current)
        return tokens

    @staticmethod
    def byte_pair_split(text: str) -> List[str]:
        """Split text into individual characters for BPE initialization."""
        return list(text)

    @staticmethod
    def mark_word_boundaries(text: str) -> str:
        """Rewrite spaces as the word-boundary marker."""
        return text.replace(" ", WORD_BOUNDARY)

    @staticmethod
    def restore_word_boundaries(text: str) -> str:
        """Inverse of mark_word_boundaries - used when decoding."""
        return text.replace(WORD_BOUNDARY, " ")

    @staticmethod
    def pretokenize(text: str) -> List[str]:
        """Split marked text into pretokens (words, punctuation runs, newlines).

        The concatenation of the result always reproduces the input exactly,
        which is what makes encode/decode lossless.
        """
        return _PRETOKEN_PATTERN.findall(text)

    @staticmethod
    def preprocess(
        text: str,
        lowercase: bool = False,
        normalize_unicode: bool = True,
        preserve_whitespace: bool = True,
    ) -> str:
        """Full preprocessing pipeline.

        preserve_whitespace keeps newlines and indentation intact. Collapsing
        them (the old default) throws away real signal: for a language model,
        line structure is part of what it is meant to learn.
        """
        if normalize_unicode:
            text = TextPreprocessor.normalize_unicode(text)
        text = TextPreprocessor.strip_control_characters(text)
        text = TextPreprocessor.remove_invalid_chars(text)
        if not preserve_whitespace:
            text = TextPreprocessor.normalize_whitespace(text)
        if lowercase:
            text = TextPreprocessor.lowercase(text)
        return text
