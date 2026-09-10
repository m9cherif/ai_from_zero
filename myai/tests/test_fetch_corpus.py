"""Tests for scripts/fetch_corpus.py's pure logic.

scripts/ is not a package (no __init__.py, and it isn't meant to be imported
from - it's a set of entry points), so the module is loaded by path rather
than imported normally.
"""

import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "fetch_corpus.py")
_SPEC = importlib.util.spec_from_file_location("fetch_corpus", _PATH)
fetch_corpus = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_corpus)


class TestIsNonEnglishLocale:
    """Multi-locale datasets (wikimedia/wikipedia, hltcoe/megawika) pack every
    language into one repo as separate file paths, which a dataset-level
    language tag cannot see through. One run that missed this pulled German,
    Arabic, Occitan and Lingala Wikipedia into an English corpus and, via the
    exotic character inventory that produced, contributed to exhausting 30 GB
    of RAM building the tokenizer's vocabulary.
    """

    @pytest.mark.parametrize("path", [
        "20231101.de/train-00000-of-00001.parquet",
        "20231101.oc/train-00000-of-00001.parquet",
        "20231101.ln/train-00000-of-00001.parquet",
        "20231101.hsb/train-00000-of-00001.parquet",
        "data/ar/ar-00413-of-02834.jsonl",
        "data/fr/fr-00001-of-00010.jsonl",
    ])
    def test_catches_the_datasets_that_caused_the_incident(self, path):
        assert fetch_corpus.is_non_english_locale(path)

    @pytest.mark.parametrize("path", [
        "20231101.en/train-00000-of-00001.parquet",
        "data/en/en-00001-of-00010.jsonl",
    ])
    def test_english_locale_is_not_flagged(self, path):
        assert not fetch_corpus.is_non_english_locale(path)

    @pytest.mark.parametrize("path", [
        "data/CC-MAIN-2013-48/004_00016.parquet",
        "wikitext-103-raw-v1/train-00001-of-00002.parquet",
        "TinyStoriesV2-GPT4-valid.txt",
        "main/train-00000-of-00001.parquet",
    ])
    def test_ordinary_shard_paths_are_left_alone(self, path):
        """The common case - a numbered shard, no locale segment at all -
        must not be swept up by an over-eager pattern."""
        assert not fetch_corpus.is_non_english_locale(path)

    def test_keep_language_is_configurable(self):
        assert fetch_corpus.is_non_english_locale("20231101.de/x.parquet", keep="de") is False
        assert fetch_corpus.is_non_english_locale("20231101.en/x.parquet", keep="de") is True

    def test_empty_keep_disables_filtering(self):
        assert fetch_corpus.is_non_english_locale("20231101.de/x.parquet", keep="") is False


class TestFetchReturnCode:
    """--max-time can abort a curl transfer after the response headers have
    already arrived: the status line it prints can still read "200" while
    curl's own exit code (28, on timeout) says the body never finished. A
    truncated multi-hundred-megabyte parquet shard slipped past a check that
    only looked at the status code, and that is what corrupted one run's
    corpus.
    """

    def test_nonzero_exit_is_a_failure_even_with_status_200(self, tmp_path, monkeypatch):
        name = "partial.parquet"
        (tmp_path / name).write_bytes(b"x" * 10_000)  # what curl wrote before aborting

        class FakeResult:
            returncode = 28
            stdout = "200"

        monkeypatch.setattr(fetch_corpus.subprocess, "run", lambda *a, **k: FakeResult())

        _, ok = fetch_corpus.fetch(str(tmp_path), name, "https://example.com/x.parquet")

        assert ok is False
        assert not (tmp_path / name).exists()  # the truncated file must not linger

    def test_clean_exit_and_status_200_is_kept(self, tmp_path, monkeypatch):
        name = "ok.parquet"
        (tmp_path / name).write_bytes(b"x" * 10_000)

        class FakeResult:
            returncode = 0
            stdout = "200"

        monkeypatch.setattr(fetch_corpus.subprocess, "run", lambda *a, **k: FakeResult())

        _, ok = fetch_corpus.fetch(str(tmp_path), name, "https://example.com/x.parquet")

        assert ok is True
        assert (tmp_path / name).exists()


class TestFetchHfResilience:
    def test_one_corrupt_file_does_not_abort_the_batch(self, tmp_path, monkeypatch):
        """The actual failure mode from the incident: an unhandled
        pyarrow.ArrowInvalid on one truncated shard propagated straight out
        of fetch_hf and killed the whole run - after twelve other files had
        already downloaded successfully and were sitting right there,
        discarded along with it.
        """
        def fake_fetch(dest_dir, name, url):
            open(os.path.join(dest_dir, name), "w").write("placeholder")
            return name, True

        def fake_extract(src, dst, limit_bytes=0):
            if "bad" in src:
                raise ValueError("Parquet magic bytes not found in footer")
            with open(dst, "w") as f:
                f.write("x" * 6000)
            return 6000

        monkeypatch.setattr(fetch_corpus, "fetch", fake_fetch)
        monkeypatch.setattr(fetch_corpus, "extract_text_to", fake_extract)

        kept = fetch_corpus.fetch_hf(
            str(tmp_path),
            specs=["owner/bad-repo:shard.parquet", "owner/good-repo:shard.parquet"],
            urls=None, max_files=4, workers=2,
        )

        assert kept == 1
        txt_files = [f for f in os.listdir(tmp_path) if f.endswith(".txt")]
        assert len(txt_files) == 1
        assert "good" in txt_files[0]
        assert not any("bad" in f for f in os.listdir(tmp_path))  # no leftover raw or partial file

    def test_all_files_bad_returns_zero_without_raising(self, tmp_path, monkeypatch):
        def fake_fetch(dest_dir, name, url):
            open(os.path.join(dest_dir, name), "w").write("placeholder")
            return name, True

        def fake_extract(src, dst, limit_bytes=0):
            raise ValueError("Parquet magic bytes not found in footer")

        monkeypatch.setattr(fetch_corpus, "fetch", fake_fetch)
        monkeypatch.setattr(fetch_corpus, "extract_text_to", fake_extract)

        kept = fetch_corpus.fetch_hf(
            str(tmp_path), specs=["owner/repo:shard.parquet"],
            urls=None, max_files=4, workers=2,
        )

        assert kept == 0
