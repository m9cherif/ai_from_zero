"""Base tokenizer interface defining encode/decode contracts."""

from abc import ABC, abstractmethod
from typing import List, Union
from ..core.types import TokenIds, TextString


class BaseTokenizer(ABC):
    """Abstract base for all tokenizer implementations."""

    @abstractmethod
    def encode(self, text: TextString, add_special_tokens: bool = True) -> TokenIds:
        """Convert text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: TokenIds, skip_special_tokens: bool = True) -> TextString:
        """Convert token IDs back to text."""
        pass

    @abstractmethod
    def encode_batch(self, texts: List[TextString], add_special_tokens: bool = True, padding: bool = False, truncation: bool = False, max_length: int = None) -> List[TokenIds]:
        """Encode a batch of texts."""
        pass

    @abstractmethod
    def decode_batch(self, batch: List[TokenIds], skip_special_tokens: bool = True) -> List[TextString]:
        """Decode a batch of token ID sequences."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Return the padding token ID."""
        pass

    @property
    @abstractmethod
    def unk_token_id(self) -> int:
        """Return the unknown token ID."""
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """Return the end-of-sequence token ID."""
        pass
