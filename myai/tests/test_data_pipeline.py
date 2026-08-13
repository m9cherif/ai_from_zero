"""Tests for the data pipeline components."""

import tempfile
import os
from ..data.cleaning.unicode import UnicodeNormalizer
from ..data.cleaning.whitespace import WhitespaceNormalizer
from ..data.cleaning.sanitizer import TextSanitizer
from ..data.filtering.quality import QualityFilter
from ..data.filtering.dedup import Deduplicator
from ..data.segmentation.segmenter import TextSegmenter
from ..data.segmentation.packer import SequencePacker
from ..data.streaming.index import DatasetIndex
from ..data.streaming.cache import PreprocessingCache
from ..data.validation.validator import DatasetValidator
from ..data.versioning.versioning import DataVersionTracker


class TestTextCleaning:
    def test_unicode_normalizer(self):
        normalizer = UnicodeNormalizer()
        text = "café\u0301 résumé"
        normalized = normalizer(text)
        assert len(normalized) > 0

    def test_whitespace_normalizer(self):
        normalizer = WhitespaceNormalizer()
        text = "hello    world\n\n\nnew paragraph"
        normalized = normalizer(text)
        assert "    " not in normalized
        assert "\n\n\n" not in normalized

    def test_sanitizer(self):
        sanitizer = TextSanitizer()
        text = "Hello\x00 World!\x1fTest"
        sanitized = sanitizer(text)
        assert "\x00" not in sanitized
        assert "\x1f" not in sanitized


class TestQualityFilter:
    def test_quality_score(self):
        qf = QualityFilter(min_length=10, max_length=1000)
        passed, score = qf.is_quality("This is a high quality text with proper length and no repetition.")
        assert passed
        assert score > 0.3

    def test_low_quality(self):
        qf = QualityFilter(min_length=10, max_length=1000, max_char_repetition=0.1)
        passed, score = qf.is_quality("aaaaa bbbbb ccccc ddddd eeeee " * 20)
        assert not passed


class TestDeduplication:
    def test_exact_dedup(self):
        dedup = Deduplicator()
        docs = ["hello world", "hello world", "unique text", "hello world"]
        unique = dedup.deduplicate_documents(docs)
        assert len(unique) == 2


class TestSegmentation:
    def test_paragraph_split(self):
        seg = TextSegmenter()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = seg.split_into_paragraphs(text)
        assert len(paragraphs) == 3
        assert "First paragraph." in paragraphs[0]

    def test_sentence_split(self):
        seg = TextSegmenter()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = seg.split_into_sentences(text)
        assert len(sentences) == 3


class TestSequencePacker:
    def test_packing(self):
        packer = SequencePacker(max_length=10, pad_token_id=0)
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        packed = packer.pack(sequences)
        for seq in packed:
            assert len(seq) == 10


class TestDatasetIndex:
    def test_index_build_and_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".idx", delete=False) as f:
            idx_path = f.name
        bin_path = idx_path.replace(".idx", ".bin")
        try:
            sequences = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
            index = DatasetIndex()
            index.build_from_sequences(sequences, idx_path)

            loaded = DatasetIndex(idx_path)
            assert len(loaded) == 3

            seq0 = loaded.get_sequence(0)
            assert seq0 == [1, 2, 3]
        finally:
            for p in [idx_path, bin_path]:
                if os.path.exists(p):
                    os.unlink(p)


class TestPreprocessingCache:
    def test_cache_set_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PreprocessingCache(tmpdir)
            result = cache.get_or_compute("test_key", lambda: [1, 2, 3])
            assert result == [1, 2, 3]

            # Should return cached
            result2 = cache.get_or_compute("test_key", lambda: [4, 5, 6])
            assert result2 == [1, 2, 3]


class TestDatasetValidator:
    def test_empty_samples(self):
        validator = DatasetValidator()
        ok, empty, total = validator.check_empty_samples([[1, 2], [], [3, 4, 5]])
        assert empty == 1

    def test_invalid_tokens(self):
        validator = DatasetValidator()
        ok, count = validator.check_invalid_tokens([[1, 2], [3, 100]], vocab_size=50)
        assert count == 1


class TestDataVersionTracker:
    def test_snapshot(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("test data")
            path = f.name

        try:
            tracker = DataVersionTracker()
            snapshot = tracker.snapshot(data_paths=[path])
            assert snapshot["num_files"] >= 1
            assert "total_hash" in snapshot
        finally:
            os.unlink(path)
