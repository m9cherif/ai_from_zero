"""Tests for the pre-tokenized corpus cache."""

from pathlib import Path

import numpy as np
import pytest

from ..data.streaming import PretokenizedDataset, TokenCache
from ..tokenizer import CharTokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog.\n"
    "a language model learns to predict the next token.\n"
) * 60


@pytest.fixture
def tokenizer():
    return CharTokenizer.train([CORPUS])


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text(CORPUS, encoding="utf-8")
    (tmp_path / "b.txt").write_text(CORPUS[::-1], encoding="utf-8")
    return tmp_path


class TestTokenCache:
    def test_build_writes_cache_and_metadata(self, corpus_dir, tmp_path, tokenizer):
        cache = TokenCache.build([str(corpus_dir)], tokenizer, str(tmp_path / "c" / "train.bin"))
        assert Path(cache).exists()
        meta = Path(cache).with_suffix(".json")
        assert meta.exists()
        assert "n_tokens" in meta.read_text()

    def test_cache_is_reused_not_rebuilt(self, corpus_dir, tmp_path, tokenizer):
        path = str(tmp_path / "train.bin")
        TokenCache.build([str(corpus_dir)], tokenizer, path)
        mtime = Path(path).stat().st_mtime_ns
        TokenCache.build([str(corpus_dir)], tokenizer, path)
        assert Path(path).stat().st_mtime_ns == mtime

    def test_overwrite_forces_rebuild(self, corpus_dir, tmp_path, tokenizer):
        path = str(tmp_path / "train.bin")
        TokenCache.build([str(corpus_dir)], tokenizer, path)
        before = Path(path).stat().st_mtime_ns
        TokenCache.build([str(corpus_dir)], tokenizer, path, overwrite=True)
        assert Path(path).stat().st_mtime_ns != before

    def test_empty_input_is_rejected(self, tmp_path, tokenizer):
        with pytest.raises(ValueError):
            TokenCache.build([str(tmp_path / "nothing")], tokenizer, str(tmp_path / "x.bin"))

    def test_ids_round_trip_through_the_cache(self, tmp_path, tokenizer):
        """Cached ids must decode back to the original text."""
        (tmp_path / "only.txt").write_text(CORPUS, encoding="utf-8")
        path = str(tmp_path / "train.bin")
        TokenCache.build([str(tmp_path)], tokenizer, path)

        ids = np.fromfile(path, dtype=np.uint16).tolist()
        eos = tokenizer.eos_token_id
        if eos is not None and ids and ids[-1] == eos:
            ids = ids[:-1]
        assert tokenizer.decode(ids, skip_special_tokens=True) == CORPUS


class TestPretokenizedDataset:
    def _dataset(self, corpus_dir, tmp_path, tokenizer, **kwargs):
        cache = TokenCache.build([str(corpus_dir)], tokenizer, str(tmp_path / "train.bin"))
        return PretokenizedDataset(cache, **kwargs)

    def test_every_window_is_exactly_max_seq_len(self, corpus_dir, tmp_path, tokenizer):
        data = self._dataset(corpus_dir, tmp_path, tokenizer, max_seq_len=64)
        windows = list(data)
        assert windows, "dataset yielded nothing"
        assert all(len(w) == 64 for w in windows)

    def test_window_count_matches_token_count(self, corpus_dir, tmp_path, tokenizer):
        data = self._dataset(corpus_dir, tmp_path, tokenizer, max_seq_len=64)
        assert len(list(data)) == len(data) == data.n_tokens // 64

    def test_shuffle_changes_order_but_not_content(self, corpus_dir, tmp_path, tokenizer):
        data = self._dataset(corpus_dir, tmp_path, tokenizer, max_seq_len=32, seed=0)
        first = list(data)
        second = list(data)  # a second epoch reshuffles
        assert first != second
        assert sorted(map(tuple, first)) == sorted(map(tuple, second))

    def test_unshuffled_is_contiguous(self, corpus_dir, tmp_path, tokenizer):
        data = self._dataset(corpus_dir, tmp_path, tokenizer, max_seq_len=32, shuffle=False)
        flat = [t for window in data for t in window]
        raw = np.fromfile(str(tmp_path / "train.bin"), dtype=np.uint16).tolist()
        assert flat == raw[:len(flat)]

    def test_too_short_corpus_is_rejected(self, corpus_dir, tmp_path, tokenizer):
        cache = TokenCache.build([str(corpus_dir)], tokenizer, str(tmp_path / "train.bin"))
        with pytest.raises(ValueError, match="too few"):
            PretokenizedDataset(cache, max_seq_len=10_000_000)

    def test_feeds_a_training_step(self, corpus_dir, tmp_path, tokenizer):
        """The loop only needs an iterable of id lists - prove it drives one."""
        from ..nn.model import LanguageModel, LMConfig
        from ..train.loop import iter_batches

        data = self._dataset(corpus_dir, tmp_path, tokenizer, max_seq_len=32)
        model = LanguageModel(LMConfig(
            vocab_size=tokenizer.vocab_size, d_model=32, n_heads=4, n_kv_heads=2,
            n_layers=2, d_ff=64, max_seq_len=32, dropout=0.0,
        ))
        batch = next(iter(iter_batches(data, batch_size=4, length_bucketing=False)))
        assert len(batch) == 4

        import torch
        ids = torch.tensor(batch)
        out = model(ids, labels=ids)
        assert torch.isfinite(out["loss"])
