"""Document segmentation into training units."""

import re
from typing import List, Iterator, Optional
from ...core.logging import logger


class TextSegmenter:
    """Splits documents into meaningful training segments."""

    def __init__(self, max_length: int = 512, overlap: int = 0):
        self._max_length = max_length
        self._overlap = overlap

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def sliding_window_chunks(self, text: str) -> Iterator[str]:
        """Generate sliding window chunks of max_length."""
        if len(text) <= self._max_length:
            yield text
            return

        start = 0
        while start < len(text):
            end = min(start + self._max_length, len(text))
            chunk = text[start:end]
            if chunk:
                yield chunk
            start += self._max_length - self._overlap

    def split_by_token_estimate(self, text: str, avg_chars_per_token: float = 4.0) -> List[str]:
        """Split text into chunks estimated to fit within max_length tokens."""
        max_chars = int(self._max_length * avg_chars_per_token)
        chunks = []
        for paragraph in self.split_into_paragraphs(text):
            if len(paragraph) <= max_chars:
                chunks.append(paragraph)
            else:
                # Split long paragraphs into sentences, then group
                sentences = self.split_into_sentences(paragraph)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current = (current + " " + sent).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
                if current:
                    chunks.append(current)
        return chunks

    def segment(self, text: str, strategy: str = "paragraph") -> List[str]:
        """Segment text using the specified strategy."""
        if strategy == "paragraph":
            return self.split_into_paragraphs(text)
        elif strategy == "sentence":
            return self.split_into_sentences(text)
        elif strategy == "sliding_window":
            return list(self.sliding_window_chunks(text))
        elif strategy == "token_estimate":
            return self.split_by_token_estimate(text)
        else:
            raise ValueError(f"Unknown segmentation strategy: {strategy}")
