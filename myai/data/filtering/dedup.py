"""Multi-level deduplication system for text data."""

import hashlib
from typing import Dict, List, Set, Optional, Tuple, Iterator
from collections import defaultdict
from ...core.logging import logger


class Deduplicator:
    """Multi-level text deduplication using hashing strategies."""

    def __init__(self, window_size: int = 5, threshold: float = 0.8):
        self._window_size = window_size
        self._threshold = threshold
        self._doc_hashes: Set[str] = set()
        self._para_hashes: Dict[str, int] = defaultdict(int)
        self._sent_hashes: Dict[str, int] = defaultdict(int)

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _minhash_signature(self, text: str, num_hashes: int = 64) -> List[int]:
        """Compute a minhash signature for near-duplicate detection."""
        words = text.split()
        if len(words) < self._window_size:
            return []

        shingles = set()
        for i in range(len(words) - self._window_size + 1):
            shingle = " ".join(words[i:i + self._window_size])
            shingle_hash = hashlib.md5(shingle.encode()).digest()
            shingle_int = int.from_bytes(shingle_hash[:4], 'big')
            shingles.add(shingle_int)

        if not shingles:
            return []

        # Generate random hash functions
        signature = []
        for seed in range(num_hashes):
            min_hash = min((s * (seed + 1)) % (2**32 - 1) for s in shingles)
            signature.append(min_hash)
        return signature

    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from minhash signatures."""
        if not sig1 or not sig2:
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / max(len(sig1), len(sig2))

    def is_exact_duplicate(self, text: str, level: str = "document") -> bool:
        """Check if text is an exact duplicate at the given level."""
        h = self._sha256(text)
        if level == "document" or level == "paragraph":
            if h in self._doc_hashes:
                return True
            self._doc_hashes.add(h)
        elif level == "sentence":
            if h in self._sent_hashes:
                return True
            self._sent_hashes[h] += 1
        return False

    def is_near_duplicate(self, text: str, existing_signatures: List[List[int]]) -> Tuple[bool, float]:
        """Check if text is a near-duplicate of existing documents."""
        sig = self._minhash_signature(text)
        if not sig:
            return False, 0.0
        for existing_sig in existing_signatures:
            sim = self._jaccard_similarity(sig, existing_sig)
            if sim > self._threshold:
                return True, sim
        return False, 0.0

    def deduplicate_documents(self, documents: List[str]) -> List[str]:
        """Remove exact duplicate documents."""
        seen: Set[str] = set()
        unique = []
        for doc in documents:
            h = self._sha256(doc)
            if h not in seen:
                seen.add(h)
                unique.append(doc)
        logger.info(f"Deduplication: {len(documents)} -> {len(unique)} documents")
        return unique

    def deduplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences from text."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen: Set[str] = set()
        unique = []
        for sent in sentences:
            normalized = sent.strip().lower()
            h = self._sha256(normalized)
            if h not in seen and len(normalized) > 3:
                seen.add(h)
                unique.append(sent)
        return " ".join(unique)

    def reset(self) -> None:
        """Clear all stored hashes."""
        self._doc_hashes.clear()
        self._para_hashes.clear()
        self._sent_hashes.clear()
