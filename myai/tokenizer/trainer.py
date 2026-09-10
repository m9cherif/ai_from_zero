"""Tokenizer training pipeline: learns BPE merges from raw text."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterator
from ..core.errors import TokenizerError
from ..core.logging import logger
from .vocabulary import Vocabulary
from .bpe import BPETokenizer
from .preprocessor import TextPreprocessor, WORD_BOUNDARY


class TokenizerTrainer:
    """Trains a BPE tokenizer from raw text data.

    Uses an incremental merge loop: rather than recounting every pair over the
    whole corpus after each merge (which makes training quadratic in vocabulary
    size), it keeps a pair -> words index and only touches the words that
    actually contained the merged pair.
    """

    def __init__(
        self,
        target_vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None,
        max_token_length: Optional[int] = None,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        self._target_vocab_size = target_vocab_size
        self._min_frequency = min_frequency
        self._special_tokens = special_tokens or {
            "pad": "[PAD]",
            "unk": "[UNK]",
            "bos": "[BOS]",
            "eos": "[EOS]",
            "mask": "[MASK]",
        }
        self._max_token_length = max_token_length
        self._preprocessor = preprocessor or TextPreprocessor()

    def _get_words_with_frequencies(self, texts: Iterator[str]) -> Counter:
        """Count pretoken frequencies across all texts.

        Pretokens carry their leading space as the ``▁`` marker, exactly as the
        encoder produces them - so every merge learned here is a merge the
        encoder can actually apply.
        """
        word_counts: Counter = Counter()
        for text in texts:
            text = self._preprocessor.preprocess(text)
            text = self._preprocessor.mark_word_boundaries(text)
            word_counts.update(self._preprocessor.pretokenize(text))
        return word_counts

    @staticmethod
    def _merge_symbols(symbols: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
        a, b = pair
        out: List[str] = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def train(
        self,
        texts: Iterator[str],
        verbose: bool = True,
    ) -> BPETokenizer:
        """Train a BPE tokenizer on the provided text iterator.

        This is the core BPE training loop:
        1. Initialize vocabulary with special tokens and all unique characters
        2. Count pretoken frequencies
        3. Iteratively merge the most frequent adjacent pair
        4. Stop at the target vocabulary size or the frequency floor
        """
        logger.info(
            f"Starting BPE training: target_vocab_size={self._target_vocab_size}, "
            f"min_frequency={self._min_frequency}"
        )

        vocab = Vocabulary(
            special_tokens=self._special_tokens,
            max_size=self._target_vocab_size,
        )
        vocab.build_initial()

        word_counts = self._get_words_with_frequencies(texts)
        logger.info(f"Collected {len(word_counts)} unique pretokens")

        # Ideally every character that appears anywhere is in the vocabulary,
        # so no position ever encodes to [UNK]. But a large, script-diverse web
        # corpus can contain more distinct characters than the target vocab
        # has room for at all - one real run hit over 4,096 distinct code
        # points (Greek and math symbols, mojibake, private-use glyphs from
        # scientific-web scrapes) and crashed outright with no merges learned
        # and no tokenizer produced. Fit what the budget allows instead:
        # weight each character by how often it actually occurs, and keep the
        # most common ones first, so a handful of one-off garbled bytes lose
        # out to the alphabet and punctuation that make up the vast majority
        # of the text. Whatever doesn't fit falls back to [UNK] at encode
        # time, same as any character this training run never saw at all.
        char_counts: Counter = Counter()
        for word, count in word_counts.items():
            for ch, occurrences in Counter(word).items():
                char_counts[ch] += occurrences * count

        budget = self._target_vocab_size - vocab.size
        if budget <= 0:
            raise TokenizerError(
                f"target_vocab_size={self._target_vocab_size} leaves no room "
                f"for characters after {vocab.size} special tokens; raise "
                f"target_vocab_size."
            )

        ranked = [ch for ch, _ in char_counts.most_common()]
        sorted_chars = sorted(ranked[:budget])
        dropped = len(ranked) - len(sorted_chars)
        if dropped > 0:
            logger.warning(
                f"Corpus has {len(ranked)} distinct characters but only "
                f"{budget} fit target_vocab_size={self._target_vocab_size}; "
                f"dropping the {dropped} rarest (they will encode to [UNK]). "
                f"Raise --vocab-size for full character coverage."
            )
        vocab.add_tokens(sorted_chars)

        splits: Dict[str, List[str]] = {word: list(word) for word in word_counts}
        pair_counts: Counter = Counter()
        pair_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for word, count in word_counts.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += count
                pair_words[pair].add(word)

        merges_performed: List[Tuple[str, str]] = []
        num_special = len(self._special_tokens)
        current_vocab_size = vocab.size

        logger.info(
            f"Initial vocab: {num_special} special + {len(sorted_chars)} characters "
            f"= {current_vocab_size}"
        )

        while current_vocab_size < self._target_vocab_size:
            if not pair_counts:
                logger.warning("No more pairs to merge; stopping early")
                break

            best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))

            if best_count < self._min_frequency:
                logger.info(
                    f"Pair frequency {best_count} below minimum {self._min_frequency}; stopping"
                )
                break

            merged_token = best_pair[0] + best_pair[1]
            if self._max_token_length and len(merged_token) > self._max_token_length:
                # Never merge past the cap; drop the pair and keep going.
                del pair_counts[best_pair]
                pair_words.pop(best_pair, None)
                continue

            try:
                vocab.add_tokens([merged_token])
            except TokenizerError:
                break

            # Rewrite only the words that actually contained this pair, and
            # adjust the affected pair counts in place.
            for word in list(pair_words.get(best_pair, ())):
                count = word_counts[word]
                symbols = splits[word]

                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_counts[pair] -= count
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    words_with_pair = pair_words.get(pair)
                    if words_with_pair is not None:
                        words_with_pair.discard(word)

                new_symbols = self._merge_symbols(symbols, best_pair, merged_token)
                splits[word] = new_symbols

                for i in range(len(new_symbols) - 1):
                    pair = (new_symbols[i], new_symbols[i + 1])
                    pair_counts[pair] += count
                    pair_words[pair].add(word)

            pair_counts.pop(best_pair, None)
            pair_words.pop(best_pair, None)

            merges_performed.append(best_pair)
            current_vocab_size = vocab.size

            if verbose and len(merges_performed) % 500 == 0:
                logger.info(
                    f"Merge {len(merges_performed)}: '{best_pair[0]}' + '{best_pair[1]}' -> "
                    f"'{merged_token}' (freq={best_count}, vocab={current_vocab_size})"
                )

        logger.info(
            f"BPE training complete: {len(merges_performed)} merges, vocab_size={vocab.size}"
        )

        return BPETokenizer(
            vocab=vocab,
            merges=merges_performed,
            max_token_length=self._max_token_length,
            preprocessor=self._preprocessor,
        )


def train_tokenizer_from_files(
    file_paths: List[str],
    target_vocab_size: int = 8192,
    min_frequency: int = 2,
    save_path: Optional[str] = None,
    **kwargs,
) -> BPETokenizer:
    """Convenience function to train a tokenizer from text files."""
    def text_iterator():
        for path in file_paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    yield line

    trainer = TokenizerTrainer(
        target_vocab_size=target_vocab_size,
        min_frequency=min_frequency,
        **kwargs,
    )
    tokenizer = trainer.train(text_iterator())

    if save_path:
        tokenizer.save(save_path)

    return tokenizer
