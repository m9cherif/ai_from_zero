"""Tokenizers: BPE and character-level, both implemented from scratch."""

import json
from typing import Union

from .base import BaseTokenizer
from .bpe import BPETokenizer
from .char import CharTokenizer
from .vocabulary import Vocabulary
from .preprocessor import TextPreprocessor, WORD_BOUNDARY
from .trainer import TokenizerTrainer, train_tokenizer_from_files


def load_tokenizer(path: str) -> Union[BPETokenizer, CharTokenizer]:
    """Load a tokenizer file, picking the implementation from its contents.

    Files written before the ``type`` field existed are classified by whether
    they carry any merges: no merges means character-level.
    """
    # A hf:// or https:// address is downloaded and cached, so a published
    # model's tokenizer travels with its weights.
    from ..checkpoint.remote import resolve_checkpoint
    path = resolve_checkpoint(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kind = data.get("type")
    if kind is None:
        kind = "bpe" if data.get("merges") else "char"

    if kind == "char":
        return CharTokenizer.from_dict(data)
    if kind == "bpe":
        return BPETokenizer.from_dict(data)
    raise ValueError(f"Unknown tokenizer type '{kind}' in {path}")


__all__ = [
    "BaseTokenizer",
    "BPETokenizer",
    "CharTokenizer",
    "Vocabulary",
    "TextPreprocessor",
    "WORD_BOUNDARY",
    "TokenizerTrainer",
    "train_tokenizer_from_files",
    "load_tokenizer",
]
