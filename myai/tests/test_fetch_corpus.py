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
