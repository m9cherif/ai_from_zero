"""Batch construction from token sequences."""

from typing import List, Optional, Dict, Any
import torch
from ...core.types import Tensor, Device
from ...core.logging import logger


class BatchBuilder:
    """Builds batches from token sequences with padding."""

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        device: Optional[Device] = None,
    ):
        self._pad_token_id = pad_token_id
        self._max_length = max_length
        self._device = device

    def build_batch(self, sequences: List[List[int]], return_attention_mask: bool = True) -> Dict[str, Tensor]:
        """Build a batch dict with input_ids, labels, and attention_mask."""
        if not sequences:
            raise ValueError("Cannot build batch from empty sequences")

        # Truncate
        if self._max_length:
            sequences = [seq[:self._max_length] for seq in sequences]

        # Find max length in batch
        max_len = max(len(seq) for seq in sequences)

        # Pad sequences
        input_ids = []
        labels = []
        attention_masks = []

        for seq in sequences:
            padded = seq + [self._pad_token_id] * (max_len - len(seq))
            input_ids.append(padded)

            # Labels are same as input_ids (for causal LM), pad tokens ignored in loss
            label = [-100 if tid == self._pad_token_id else tid for tid in padded]
            labels.append(label)

            mask = [1 if tid != self._pad_token_id else 0 for tid in padded]
            attention_masks.append(mask)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self._device),
            "labels": torch.tensor(labels, dtype=torch.long, device=self._device),
        }
        if return_attention_mask:
            batch["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long, device=self._device)

        return batch
