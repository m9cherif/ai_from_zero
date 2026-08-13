"""Streaming dataset loader for memory-efficient data processing."""

import os
import random
import json
import pickle
from pathlib import Path
from typing import List, Iterator, Optional, Callable, Any, Dict, Tuple
from ...core.logging import logger
from ...core.types import TokenIds
from ..segmentation import TextSegmenter, SequencePacker
from ..ingestion import DataDiscoverer, TextReader, FormatDetector
from ..cleaning import TextSanitizer
from ..filtering import QualityFilter


class StreamingDataset:
    """Memory-efficient dataset that streams data from disk.

    Supports lazy loading, on-the-fly preprocessing, segmentation, token
    packing and shuffling.

    Reading a .txt file yields the whole file as a single string. Feeding that
    straight to the tokenizer and truncating to max_seq_len would throw away
    everything past the first few hundred tokens, so documents are segmented
    into paragraphs first and the token stream is then re-chunked.
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: Any,
        max_seq_len: int = 512,
        shuffle_buffer_size: int = 10000,
        sanitizer: Optional[TextSanitizer] = None,
        quality_filter: Optional[QualityFilter] = None,
        segmenter: Optional[TextSegmenter] = None,
        packer: Optional[SequencePacker] = None,
        pack_sequences: bool = True,
        segment_strategy: str = "paragraph",
        add_special_tokens: bool = False,
        cache_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self._data_paths = data_paths
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._shuffle_buffer_size = max(shuffle_buffer_size, 1)
        self._sanitizer = sanitizer or TextSanitizer()
        self._quality_filter = quality_filter
        self._segmenter = segmenter or TextSegmenter(max_length=max_seq_len * 4)
        self._packer = packer
        self._pack_sequences = pack_sequences
        self._segment_strategy = segment_strategy
        self._add_special_tokens = add_special_tokens
        self._cache_dir = cache_dir
        self._rng = random.Random(seed) if seed is not None else random

        self._discoverer = DataDiscoverer()
        self._reader = TextReader()

        self._file_list: List[str] = []
        self._index = 0
        self._epoch = 0
        self._shuffle_buffer: List[TokenIds] = []

    def discover_files(self) -> List[str]:
        """Discover all data files from configured paths."""
        self._file_list = self._discoverer.discover(self._data_paths)
        self._rng.shuffle(self._file_list)
        logger.info(f"Discovered {len(self._file_list)} files for streaming")
        return self._file_list

    def _iter_documents(self) -> Iterator[str]:
        """Yield cleaned, quality-filtered text segments from every file."""
        file_queue = list(self._file_list)
        self._rng.shuffle(file_queue)

        for file_path in file_queue:
            try:
                fmt = FormatDetector().detect(file_path)
                for raw in self._reader.read_as_text(file_path, format=fmt):
                    if not raw or not raw.strip():
                        continue

                    # A whole .txt file arrives as one string; split it so each
                    # piece is a plausible training unit rather than one
                    # truncated sample per file.
                    for segment in self._segmenter.segment(raw, self._segment_strategy):
                        segment = self._sanitizer(segment)
                        if not segment or not segment.strip():
                            continue

                        if self._quality_filter is not None:
                            passed, _ = self._quality_filter.is_quality(segment)
                            if not passed:
                                continue

                        yield segment
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")

    def _iter_token_sequences(self) -> Iterator[TokenIds]:
        """Tokenize documents and cut them into training sequences."""
        if self._pack_sequences:
            # Concatenate the token stream and slice fixed-length windows: no
            # padding at all, which is how GPT-style pretraining is done.
            eos = getattr(self._tokenizer, "eos_token_id", None)
            buffer: List[int] = []

            for segment in self._iter_documents():
                tokens = self._tokenizer.encode(
                    segment, add_special_tokens=self._add_special_tokens
                )
                if not tokens:
                    continue
                buffer.extend(tokens)
                if eos is not None:
                    buffer.append(eos)

                while len(buffer) >= self._max_seq_len:
                    yield buffer[:self._max_seq_len]
                    buffer = buffer[self._max_seq_len:]

            # Keep the tail only if it is long enough to be worth a step.
            if len(buffer) >= 16:
                yield buffer
        else:
            for segment in self._iter_documents():
                tokens = self._tokenizer.encode(
                    segment, add_special_tokens=self._add_special_tokens
                )
                if len(tokens) < 2:
                    continue
                for start in range(0, len(tokens), self._max_seq_len):
                    chunk = tokens[start:start + self._max_seq_len]
                    if len(chunk) >= 2:
                        yield chunk

    def __iter__(self) -> Iterator[TokenIds]:
        """Iterate over tokenized sequences, shuffled through a buffer."""
        self._shuffle_buffer = []
        if not self._file_list:
            self.discover_files()

        for sequence in self._iter_token_sequences():
            self._shuffle_buffer.append(sequence)
            if len(self._shuffle_buffer) >= self._shuffle_buffer_size:
                idx = self._rng.randrange(len(self._shuffle_buffer))
                # Swap-with-last then pop: O(1), where pop(idx) is O(n).
                self._shuffle_buffer[idx], self._shuffle_buffer[-1] = (
                    self._shuffle_buffer[-1],
                    self._shuffle_buffer[idx],
                )
                yield self._shuffle_buffer.pop()

        self._rng.shuffle(self._shuffle_buffer)
        while self._shuffle_buffer:
            yield self._shuffle_buffer.pop()

        self._epoch += 1

    def take(self, n: int) -> List[TokenIds]:
        """Materialize the first n sequences (useful for tests and probes)."""
        out = []
        for seq in self:
            out.append(seq)
            if len(out) >= n:
                break
        return out

    def __len__(self) -> int:
        return len(self._file_list)

    @property
    def epoch(self) -> int:
        return self._epoch

    def state_dict(self) -> dict:
        return {
            "file_list": self._file_list,
            "index": self._index,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self._file_list = state.get("file_list", [])
        self._index = state.get("index", 0)
        self._epoch = state.get("epoch", 0)
