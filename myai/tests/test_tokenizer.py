"""Tests for the tokenizer system."""

import tempfile
import os
import json
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
