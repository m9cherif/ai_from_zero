"""Character-level tokenizer.

The simplest tokenizer that works: one token per character. The vocabulary is
tiny (tens of entries for English), there are no unknown words, and encode and
decode are exact inverses. Sequences are much longer than with BPE for the same
text, so it is the right default for small models on small corpora and the
wrong one at scale.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from ..core.errors import TokenizerError
from ..core.logging import logger
from .base import BaseTokenizer
from .vocabulary import Vocabulary

DEFAULT_SPECIAL_TOKENS = {
    "pad": "[PAD]",
    "unk": "[UNK]",
    "bos": "[BOS]",
    "eos": "[EOS]",
    "mask": "[MASK]",
}


class CharTokenizer(BaseTokenizer):
    """Maps each character to a token ID."""

    def __init__(self, vocab: Optional[Vocabulary] = None):
        self._vocab = vocab if vocab is not None else Vocabulary(DEFAULT_SPECIAL_TOKENS)
        if self._vocab.size == 0:
            self._vocab.build_initial()

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        special_tokens: Optional[Dict[str, str]] = None,
        min_frequency: int = 1,
    ) -> "CharTokenizer":
        """Build a vocabulary from the characters present in the corpus."""
        counts: Dict[str, int] = {}
        for text in texts:
            for ch in text:
                counts[ch] = counts.get(ch, 0) + 1

        vocab = Vocabulary(special_tokens or DEFAULT_SPECIAL_TOKENS)
        vocab.build_initial()

        # Sorted for determinism: the same corpus always yields the same IDs.
        chars = sorted(ch for ch, n in counts.items() if n >= min_frequency)
        vocab.add_tokens(chars)

        logger.info(f"Character vocabulary built: {vocab.size} tokens from {len(counts)} distinct chars")
        return cls(vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self._vocab.get_id(ch) for ch in text]
        if add_special_tokens:
            if self._vocab.bos_id is not None:
                ids = [self._vocab.bos_id] + ids
            if self._vocab.eos_id is not None:
                ids = ids + [self._vocab.eos_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        out = []
        for tid in token_ids:
            if skip_special_tokens and self._vocab.is_special(tid):
                continue
            out.append(self._vocab.get_token(tid))
        return "".join(out)

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int = None,
    ) -> List[List[int]]:
        batch = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

        if truncation and max_length is not None:
            batch = [seq[:max_length] for seq in batch]

        if padding:
            target = max((len(s) for s in batch), default=0) if max_length is None else max_length
            pad_id = self._vocab.pad_id
            batch = [seq + [pad_id] * (target - len(seq)) for seq in batch]

        return batch

    def decode_batch(self, batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
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

    def to_dict(self) -> dict:
        return {"type": "char", "vocab": self._vocab.to_dict(), "merges": []}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(Vocabulary.from_dict(data["vocab"]))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Tokenizer saved to {path} (vocab={self.vocab_size})")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
