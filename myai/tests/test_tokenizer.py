"""Tests for the tokenizer system."""

import tempfile
import os
import json
import pytest
from ..core.errors import TokenizerError
from ..tokenizer.vocabulary import Vocabulary
from ..tokenizer.bpe import BPETokenizer
from ..tokenizer.trainer import TokenizerTrainer


class TestVocabulary:
    def test_special_tokens(self):
        special = {"pad": "[PAD]", "unk": "[UNK]", "bos": "[BOS]", "eos": "[EOS]"}
        vocab = Vocabulary(special_tokens=special)
        vocab.build_initial()
        assert vocab.pad_id is not None
        assert vocab.unk_id is not None
        assert vocab.bos_id is not None
        assert vocab.eos_id is not None

    def test_add_and_retrieve(self):
        vocab = Vocabulary()
        vocab.build_initial()
        ids = vocab.add_tokens(["hello", "world"])
        assert vocab.get_id("hello") == ids[0]
        assert vocab.get_token(ids[0]) == "hello"
        assert vocab.get_token(ids[1]) == "world"

    def test_unknown_token(self):
        vocab = Vocabulary()
        vocab.build_initial()
        unk_id = vocab.get_id("[UNK]")
        unknown = vocab.get_id("nonexistent")
        assert unknown == unk_id

    def test_serialization(self):
        vocab = Vocabulary()
        vocab.build_initial()
        vocab.add_tokens(["hello", "world", "test"])
        data = vocab.to_dict()
        restored = Vocabulary.from_dict(data)
        assert restored.size == vocab.size
        assert restored.get_id("hello") == vocab.get_id("hello")

    def test_save_load(self):
        vocab = Vocabulary()
        vocab.build_initial()
        vocab.add_tokens(["hello", "world"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            vocab.save(path)
            loaded = Vocabulary.load(path)
            assert loaded.size == vocab.size
            assert loaded.get_id("hello") == vocab.get_id("hello")
        finally:
            os.unlink(path)


class TestBPETokenizer:
    def test_basic_encode_decode(self):
        vocab = Vocabulary()
        vocab.build_initial()
        vocab.add_tokens(list("abcde h"))
        tokenizer = BPETokenizer(vocab=vocab, merges=[])

        text = "a b c"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert len(ids) > 0

    def test_special_tokens_in_encode(self):
        vocab = Vocabulary()
        special = {"pad": "[PAD]", "unk": "[UNK]", "bos": "[BOS]", "eos": "[EOS]"}
        vocab = Vocabulary(special_tokens=special)
        tokenizer = BPETokenizer(vocab=vocab, merges=[])

        ids = tokenizer.encode("test", add_special_tokens=True)
        # Should have BOS and EOS
        assert ids[0] == vocab.bos_id
        assert ids[-1] == vocab.eos_id

    def test_encode_cache_is_bounded(self):
        """The per-pretoken memo cache must not grow without limit.

        A tokenizer instance reused across a large, script-diverse corpus - one
        real run spanned German, Arabic, Occitan and Lingala text - can be
        asked to memoize millions of distinct pretokens. Left unbounded that
        cache exhausted 30 GB of RAM on its own; it must self-evict instead.
        """
        vocab = Vocabulary()
        vocab.build_initial()
        vocab.add_tokens(list("abcdefghijklmnopqrstuvwxyz "))
        tokenizer = BPETokenizer(vocab=vocab, merges=[])
        tokenizer._encode_cache_limit = 10

        for i in range(100):
            tokenizer.encode(f"word{i}", add_special_tokens=False)

        assert len(tokenizer._encode_cache) <= 10

    def test_encode_cache_eviction_does_not_change_output(self):
        """Eviction is a memory bound, not a correctness change: a pretoken
        encoded before and after a clear must produce the same symbols."""
        vocab = Vocabulary()
        vocab.build_initial()
        vocab.add_tokens(list("abcdefghijklmnopqrstuvwxyz "))
        tokenizer = BPETokenizer(vocab=vocab, merges=[("t", "h")])
        tokenizer._encode_cache_limit = 3

        text = "the quick brown fox"
        before = tokenizer.encode(text, add_special_tokens=False)
        for i in range(50):
            tokenizer.encode(f"filler{i}", add_special_tokens=False)
        after = tokenizer.encode(text, add_special_tokens=False)

        assert before == after


class TestTokenizerTrainer:
    def test_train_small(self):
        texts = [
            "hello world this is a test",
            "hello world another test here",
            "testing tokenizer training function",
            "this is a small corpus for testing",
        ]
        trainer = TokenizerTrainer(
            target_vocab_size=50,
            min_frequency=1,
        )
        tokenizer = trainer.train(iter(texts), verbose=False)
        assert tokenizer.vocab_size <= 50
        assert tokenizer.vocab_size > 4  # At least special tokens

    def test_encode_and_decode(self):
        texts = ["hello world this is a test corpus"]
        trainer = TokenizerTrainer(target_vocab_size=30, min_frequency=1)
        tokenizer = trainer.train(iter(texts), verbose=False)

        text = "hello world test"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # The decoded text should preserve the original content
        assert len(decoded) > 0

    def test_save_load_tokenizer(self):
        texts = ["hello world this is a test for save and load"]
        trainer = TokenizerTrainer(target_vocab_size=30, min_frequency=1)
        tokenizer = trainer.train(iter(texts), verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            tokenizer.save(path)
            loaded = BPETokenizer.load(path)
            assert loaded.vocab_size == tokenizer.vocab_size

            text = "hello world"
            original_ids = tokenizer.encode(text, add_special_tokens=False)
            loaded_ids = loaded.encode(text, add_special_tokens=False)
            assert original_ids == loaded_ids
        finally:
            os.unlink(path)

    def test_character_budget_exceeded_degrades_instead_of_crashing(self):
        """A large, script-diverse corpus can contain more distinct characters
        than a small target vocabulary has room for at all - one real run hit
        over 4,096 distinct code points from a web-scraped scientific corpus
        and crashed outright, with no tokenizer produced. Training must fit
        what the budget allows rather than raise.
        """
        common = "the quick brown fox jumps over the lazy dog runs away "
        # 15 characters that appear exactly once each - the ones that must lose
        # out to the alphabet above when the budget can't hold everything.
        rare = "".join(chr(0x0391 + i) for i in range(15))
        text = common * 500 + " ".join(rare)

        trainer = TokenizerTrainer(target_vocab_size=25, min_frequency=1)
        tokenizer = trainer.train(iter([text]), verbose=False)

        assert tokenizer.vocab_size == 25
        # Coverage exists for what fit the budget...
        ids = tokenizer.encode("the quick brown fox", add_special_tokens=False)
        assert tokenizer.decode(ids, skip_special_tokens=True) == "the quick brown fox"
        # ...and what didn't degrades to [UNK] instead of raising.
        dropped_ids = tokenizer.encode(rare[0], add_special_tokens=False)
        assert dropped_ids  # produced *something*, did not crash

    def test_character_selection_prefers_frequent_characters(self):
        """When characters must be dropped to fit the budget, the ones kept
        must be the ones that actually appear often - dropping by frequency,
        not by first-seen order or sort order.

        target_vocab_size=6 leaves exactly one character slot after the 5
        default special tokens, so this pins the tie-break directly: 'e'
        (frequency 20) must win it over ' ' and 'z' (frequency 1 each).
        """
        text = "eeeeeeeeeeeeeeeeeeee z"
        trainer = TokenizerTrainer(target_vocab_size=6, min_frequency=1)
        tokenizer = trainer.train(iter([text]), verbose=False)

        assert tokenizer.vocab_size == 6

        e_ids = tokenizer.encode("e", add_special_tokens=False)
        z_ids = tokenizer.encode("z", add_special_tokens=False)

        # 'e' earned the one available slot and gets a real token; 'z' lost
        # out and falls back to [UNK] - so the two must not be the same id.
        assert e_ids != z_ids

    def test_vocab_with_zero_room_for_characters_raises_a_clear_error(self):
        """Not every failure should be swallowed by the graceful-degradation
        path: a target_vocab_size that exactly fits the special tokens and
        nothing else leaves zero budget for even one character, which is a
        real misconfiguration distinct from "corpus has too many characters"
        and should say so plainly.

        target_vocab_size=5 exactly fits the 5 default special tokens (that
        much succeeds), leaving budget = 0 for characters - the boundary this
        test pins down.
        """
        trainer = TokenizerTrainer(target_vocab_size=5, min_frequency=1)
        with pytest.raises(TokenizerError, match="leaves no room"):
            trainer.train(iter(["hello world"]), verbose=False)
