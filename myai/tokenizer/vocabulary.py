"""Vocabulary management: construction, serialization, token-id mapping."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Sequence, Iterator
from ..core.errors import TokenizerError
from ..core.logging import logger


class Vocabulary:
    """Maps tokens to IDs and vice versa. Handles special tokens."""

    def __init__(
        self,
        special_tokens: Optional[Dict[str, str]] = None,
        max_size: Optional[int] = None,
    ):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._max_size = max_size

        self._special_names: Dict[str, str] = {}
        self._special_tokens: Set[str] = set()

        if special_tokens:
            for name, token in special_tokens.items():
                self._special_names[name] = token

    def _add_token(self, token: str) -> int:
        """Add a single token and return its ID."""
        if token in self._token_to_id:
            return self._token_to_id[token]
        if self._max_size is not None and len(self._token_to_id) >= self._max_size:
            raise TokenizerError(f"Vocabulary exceeds maximum size {self._max_size}")
        tid = len(self._token_to_id)
        self._token_to_id[token] = tid
        self._id_to_token[tid] = token
        return tid

    def build_initial(self, special_tokens: Optional[Dict[str, str]] = None) -> None:
        """Initialize vocabulary with special tokens."""
        tokens = special_tokens or self._special_names
        for name, token in tokens.items():
            if token not in self._token_to_id:
                self._add_token(token)
            self._special_names[name] = token

    def add_tokens(self, tokens: Sequence[str]) -> List[int]:
        """Add multiple tokens to the vocabulary. Returns their IDs."""
        return [self._add_token(t) for t in tokens]

    def get_id(self, token: str) -> int:
        if token not in self._token_to_id:
            return self._token_to_id.get(self._special_names.get("unk", "[UNK]"), 0)
        return self._token_to_id[token]

    def get_token(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, self._special_names.get("unk", "[UNK]"))

    def get_special_id(self, name: str) -> Optional[int]:
        token = self._special_names.get(name)
        if token is None:
            return None
        return self._token_to_id.get(token)

    @property
    def pad_id(self) -> int:
        return self.get_special_id("pad") or 0

    @property
    def unk_id(self) -> int:
        return self.get_special_id("unk") or 0

    @property
    def bos_id(self) -> Optional[int]:
        return self.get_special_id("bos")

    @property
    def eos_id(self) -> Optional[int]:
        return self.get_special_id("eos")

    @property
    def mask_id(self) -> Optional[int]:
        return self.get_special_id("mask")

    @property
    def size(self) -> int:
        return len(self._token_to_id)

    @property
    def max_size(self) -> Optional[int]:
        return self._max_size

    def is_special(self, token_id: int) -> bool:
        token = self._id_to_token.get(token_id, "")
        return token in self._special_tokens or token in self._special_names.values()

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __iter__(self) -> Iterator[str]:
        return iter(self._token_to_id.keys())

    def items(self):
        return self._token_to_id.items()

    def to_dict(self) -> dict:
        """Serialize vocabulary to a dictionary."""
        return {
            "token_to_id": self._token_to_id,
            "special_names": self._special_names,
            "max_size": self._max_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        """Restore vocabulary from a dictionary."""
        vocab = cls(max_size=data.get("max_size"))
        vocab._token_to_id = data["token_to_id"]
        vocab._id_to_token = {int(k): v for k, v in data.get("id_to_token", {}).items()}
        if not vocab._id_to_token:
            vocab._id_to_token = {v: k for k, v in vocab._token_to_id.items()}
        vocab._special_names = data.get("special_names", {})
        vocab._special_tokens = set(vocab._special_names.values())
        return vocab

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary saved to {path} (size={self.size})")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
