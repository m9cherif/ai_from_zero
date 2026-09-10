"""BPE (Byte-Pair Encoding) tokenizer implementation from scratch."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
from ..core.errors import TokenizerError
from ..core.logging import logger
from .base import BaseTokenizer
from .vocabulary import Vocabulary
from .preprocessor import TextPreprocessor, WORD_BOUNDARY


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding tokenizer with learnable merge operations.

    Implements BPE from first principles: pretokenization, character-level
    splitting, merge application by learned rank, encoding, and decoding.

    Whitespace is preserved: spaces become the ``▁`` boundary marker before
    tokenizing and are restored on decode, so ``decode(encode(text)) == text``
    for any text whose characters are in the vocabulary.
    """

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        max_token_length: Optional[int] = None,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        self._vocab = vocab if vocab is not None else Vocabulary()
        self._merges: Dict[Tuple[str, str], str] = {}
        self._merge_priority: Dict[Tuple[str, str], int] = {}
        self._max_token_length = max_token_length
        self._preprocessor = preprocessor or TextPreprocessor()

        if merges:
            self.set_merges(merges)

        # Words repeat constantly in natural text, and applying merges is the
        # expensive part of encoding, so cache the result per pretoken. The
        # cache is bounded: an unbounded dict, reused across every file a
        # TokenCache builds from, grows with every unique pretoken it has ever
        # seen. A large, script-diverse corpus (many languages, many alphabets)
        # can hold millions of them - that filled 30 GB of Kaggle RAM on one
        # such run - so the cache is dropped and restarted once it gets big,
        # trading a burst of recompute for a hard memory ceiling.
        self._encode_cache: Dict[str, List[str]] = {}
        self._encode_cache_limit = 2_000_000
        self._has_initialized_special = False

    def _ensure_special_tokens(self) -> None:
        if not self._has_initialized_special:
            self._vocab.build_initial()
            self._has_initialized_special = True

    def _get_pair_stats(self, word: List[str]) -> Dict[Tuple[str, str], int]:
        """Count adjacent pair frequencies in a word represented as a list of tokens."""
        stats: Dict[Tuple[str, str], int] = {}
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _merge_pair(self, word: List[str], pair: Tuple[str, str], replacement: str) -> List[str]:
        """Apply a single merge operation to a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def _apply_merges(self, pretoken: str) -> List[str]:
        """Reduce one pretoken to its BPE symbols, cheapest merge first."""
        cached = self._encode_cache.get(pretoken)
        if cached is not None:
            return cached

        if len(self._encode_cache) >= self._encode_cache_limit:
            self._encode_cache.clear()

        symbols = list(pretoken)

        while len(symbols) > 1:
            best_pair = None
            best_rank = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_priority.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            symbols = self._merge_pair(symbols, best_pair, self._merges[best_pair])

        if self._max_token_length:
            symbols = self._split_long_symbols(symbols)

        self._encode_cache[pretoken] = symbols
        return symbols

    def _split_long_symbols(self, symbols: List[str]) -> List[str]:
        """Break any symbol longer than max_token_length back into characters."""
        out: List[str] = []
        for symbol in symbols:
            if len(symbol) > self._max_token_length:
                out.extend(list(symbol))
            else:
                out.append(symbol)
        return out

    def tokenize(self, text: str) -> List[str]:
        """Convert text into BPE token strings (no IDs)."""
        text = self._preprocessor.preprocess(text)
        text = self._preprocessor.mark_word_boundaries(text)

        tokens: List[str] = []
        for pretoken in self._preprocessor.pretokenize(text):
            tokens.extend(self._apply_merges(pretoken))
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs using BPE."""
        self._ensure_special_tokens()

        token_ids = [self._vocab.get_id(token) for token in self.tokenize(text)]

        if add_special_tokens:
            if self._vocab.bos_id is not None:
                token_ids = [self._vocab.bos_id] + token_ids
            if self._vocab.eos_id is not None:
                token_ids = token_ids + [self._vocab.eos_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text.

        Tokens are concatenated verbatim and the word-boundary marker is turned
        back into a space, so spacing and newlines come back exactly as encoded.
        """
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and self._vocab.is_special(tid):
                continue
            tokens.append(self._vocab.get_token(tid))

        return self._preprocessor.restore_word_boundaries("".join(tokens))

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int = None,
    ) -> List[List[int]]:
        """Encode a batch of texts."""
        batch = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

        if truncation and max_length is not None:
            batch = [seq[:max_length] for seq in batch]

        if padding:
            max_len = max((len(seq) for seq in batch), default=0) if max_length is None else max_length
            pad_id = self._vocab.pad_id
            batch = [seq + [pad_id] * (max_len - len(seq)) for seq in batch]

        return batch

    def decode_batch(self, batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in batch]

    @property
    def vocab_size(self) -> int:
        return self._vocab.size

    @property
    def pad_token_id(self) -> int:
        return self._vocab.pad_id

    @property
    def unk_token_id(self) -> int:
        return self._vocab.unk_id

    @property
    def bos_token_id(self) -> int:
        bos = self._vocab.bos_id
        return bos if bos is not None else 0

    @property
    def eos_token_id(self) -> int:
        eos = self._vocab.eos_id
        return eos if eos is not None else 0

    def get_vocabulary(self) -> Vocabulary:
        return self._vocab

    def set_merges(self, merges: List[Tuple[str, str]]) -> None:
        """Set merge operations from a list of (left, right) pairs, in rank order."""
        self._merges = {}
        self._merge_priority = {}
        for rank, (a, b) in enumerate(merges):
            self._merges[(a, b)] = a + b
            self._merge_priority[(a, b)] = rank
        self._encode_cache = {}

    def to_dict(self) -> dict:
        """Serialize the tokenizer state."""
        merges_list = [
            [a, b] for (a, b), _ in sorted(self._merge_priority.items(), key=lambda kv: kv[1])
        ]
        return {
            "type": "bpe",
            "vocab": self._vocab.to_dict(),
            "merges": merges_list,
            "max_token_length": self._max_token_length,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        """Restore a tokenizer from its serialized form."""
        vocab = Vocabulary.from_dict(data["vocab"])
        merges = [tuple(m) for m in data.get("merges", [])]
        tokenizer = cls(
            vocab=vocab,
            merges=merges,
            max_token_length=data.get("max_token_length"),
        )
        tokenizer._has_initialized_special = True
        return tokenizer

    def save(self, path: str) -> None:
        """Save tokenizer to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Tokenizer saved to {path} (vocab={self.vocab_size}, merges={len(self._merges)})")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
