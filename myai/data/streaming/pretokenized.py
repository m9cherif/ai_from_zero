"""Pre-tokenized corpus cache.

``StreamingDataset`` tokenizes text every time it is iterated, so a multi-epoch
run pays the full BPE cost once per epoch, competing with the model for the same
cores. On CPU that overhead is a large fraction of the step budget - measured at
roughly 40% of throughput on a 4-core box.

Tokenizing once into a flat array of token ids and memory-mapping it removes
that cost entirely. The cache is a plain binary file of fixed-width integers
plus a small JSON sidecar, so it is inspectable and cheap to rebuild.
"""

import json
import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

from ...core.logging import logger

CACHE_VERSION = 1


def _dtype_for(vocab_size: int):
    """Smallest integer type that can hold every id in the vocabulary."""
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    return np.uint32


class TokenCache:
    """Builds and describes an on-disk array of token ids."""

    @staticmethod
    def build(
        paths: Sequence[str],
        tokenizer,
        cache_path: str,
        add_special_tokens: bool = False,
        overwrite: bool = False,
    ) -> str:
        """Tokenize every file in ``paths`` into one flat id array.

        Returns the path to the cache. Documents are separated by the EOS token
        when the tokenizer defines one, so the model gets a document boundary
        signal instead of silently running one book into the next.
        """
        cache = Path(cache_path)
        meta_path = cache.with_suffix(".json")

        if cache.exists() and meta_path.exists() and not overwrite:
            meta = json.loads(meta_path.read_text())
            if meta.get("version") == CACHE_VERSION and meta.get("vocab_size") == tokenizer.vocab_size:
                logger.info(f"Reusing token cache {cache} ({meta['n_tokens']:,} tokens)")
                return str(cache)

        files: List[str] = []
        for path in paths:
            if os.path.isdir(path):
                files.extend(
                    str(Path(path) / name) for name in sorted(os.listdir(path))
                    if name.endswith(".txt")
                )
            elif os.path.isfile(path):
                files.append(path)
        if not files:
            raise ValueError(f"No .txt files found in {list(paths)}")

        dtype = _dtype_for(tokenizer.vocab_size)
        eos = getattr(tokenizer, "eos_token_id", None)

        cache.parent.mkdir(parents=True, exist_ok=True)
        n_tokens = 0
        with open(cache, "wb") as out:
            for i, path in enumerate(files, 1):
                text = open(path, encoding="utf-8", errors="replace").read()
                ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
                if eos is not None:
                    ids = list(ids) + [eos]
                array = np.asarray(ids, dtype=dtype)
                array.tofile(out)
                n_tokens += array.size
                if i % 20 == 0 or i == len(files):
                    logger.info(f"  tokenized {i}/{len(files)} files, {n_tokens:,} tokens")

        meta_path.write_text(json.dumps({
            "version": CACHE_VERSION,
            "n_tokens": int(n_tokens),
            "dtype": np.dtype(dtype).name,
            "vocab_size": int(tokenizer.vocab_size),
            "n_files": len(files),
        }, indent=2))

        logger.info(f"Token cache written to {cache}: {n_tokens:,} tokens from {len(files)} files")
        return str(cache)


class PretokenizedDataset:
    """Iterates fixed-length windows over a pre-tokenized corpus.

    Yields lists of exactly ``max_seq_len`` ids, so batches never pad and every
    token in a batch contributes to the loss. Window order is shuffled per
    epoch; the underlying array is memory-mapped and never copied.
    """

    def __init__(
        self,
        cache_path: str,
        max_seq_len: int = 512,
        shuffle: bool = True,
        seed: Optional[int] = None,
        limit_windows: Optional[int] = None,
    ):
        cache = Path(cache_path)
        meta = json.loads(cache.with_suffix(".json").read_text())

        self._tokens = np.memmap(cache, dtype=np.dtype(meta["dtype"]), mode="r")
        self._max_seq_len = max_seq_len
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

        self._n_windows = len(self._tokens) // max_seq_len
        if limit_windows is not None:
            self._n_windows = min(self._n_windows, limit_windows)
        if self._n_windows == 0:
            raise ValueError(
                f"Corpus has {len(self._tokens)} tokens, too few for one "
                f"window of {max_seq_len}"
            )

    @property
    def n_tokens(self) -> int:
        return int(len(self._tokens))

    @property
    def n_windows(self) -> int:
        return int(self._n_windows)

    def __len__(self) -> int:
        return int(self._n_windows)

    def __iter__(self) -> Iterator[List[int]]:
        order = np.arange(self._n_windows)
        if self._shuffle:
            # Reseeded per epoch so successive epochs differ but a fixed seed
            # still reproduces the whole run.
            rng = np.random.default_rng(
                None if self._seed is None else self._seed + self._epoch
            )
            rng.shuffle(order)
        self._epoch += 1

        seq = self._max_seq_len
        for index in order:
            start = int(index) * seq
            yield self._tokens[start:start + seq].tolist()
