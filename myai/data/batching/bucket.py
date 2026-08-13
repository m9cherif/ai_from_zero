"""Dynamic bucketing batching: groups sequences by length to minimize padding."""

import random
from typing import List, Optional, Dict, Any, Tuple
import torch
from ...core.types import Tensor
from ...core.logging import logger


class BucketBatchBuilder:
    """Groups sequences by length similarity for efficient batching.

    Sequences are sorted into length buckets, then each bucket is
    processed independently to minimize padding overhead.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        bucket_boundaries: Optional[List[int]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        max_length: Optional[int] = None,
    ):
        self._pad_token_id = pad_token_id
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._max_length = max_length

        # Default bucket boundaries (exponential)
        if bucket_boundaries is None:
            self._boundaries = [16, 32, 64, 128, 256, 384, 512, 768, 1024]
        else:
            self._boundaries = sorted(bucket_boundaries)

    def _assign_bucket(self, length: int) -> int:
        """Assign a sequence length to a bucket index."""
        for i, boundary in enumerate(self._boundaries):
            if length <= boundary:
                return i
        return len(self._boundaries)  # Extra bucket for very long sequences

    def build_batches(self, sequences: List[List[int]]) -> List[Dict[str, Tensor]]:
        """Group sequences into length-bucketed batches."""
        # Assign buckets
        buckets: Dict[int, List[List[int]]] = {}
        for seq in sequences:
            if self._max_length:
                seq = seq[:self._max_length]
            bucket_idx = self._assign_bucket(len(seq))
            buckets.setdefault(bucket_idx, []).append(seq)

        if self._shuffle:
            for bucket in buckets.values():
                random.shuffle(bucket)

        batches = []
        for bucket_idx in sorted(buckets.keys()):
            bucket_seqs = buckets[bucket_idx]
            max_seq_len = self._boundaries[bucket_idx] if bucket_idx < len(self._boundaries) else (self._max_length or 2048)

            for i in range(0, len(bucket_seqs), self._batch_size):
                batch_seqs = bucket_seqs[i:i + self._batch_size]

                # Build batch with padding to bucket max length
                input_ids = []
                for seq in batch_seqs:
                    padded = seq + [self._pad_token_id] * (max_seq_len - len(seq))
                    input_ids.append(padded)

                labels = [
                    [-100 if tid == self._pad_token_id else tid for tid in seq]
                    for seq in input_ids
                ]

                batches.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor([
                        [1 if tid != self._pad_token_id else 0 for tid in seq]
                        for seq in input_ids
                    ], dtype=torch.long),
                })

        return batches
