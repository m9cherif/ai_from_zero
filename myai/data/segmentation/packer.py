"""Sequence packing: combining short sequences into efficient training sequences."""

from typing import List, Tuple, Optional
from ...core.logging import logger


class SequencePacker:
    """Packs multiple short sequences into full-length training sequences.

    This maximizes GPU utilization by minimizing padding waste.
    """

    def __init__(self, max_length: int = 512, pad_token_id: int = 0, eos_token_id: Optional[int] = None):
        self._max_length = max_length
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id

    def pack(self, sequences: List[List[int]]) -> List[List[int]]:
        """Pack short sequences into full-length sequences.

        Multiple short sequences are concatenated with EOS tokens
        between them to fill max_length.
        """
        packed = []
        current_buffer: List[int] = []

        for seq in sequences:
            seq_with_eos = list(seq)
            if self._eos_token_id is not None:
                seq_with_eos = seq + [self._eos_token_id]

            if len(current_buffer) + len(seq_with_eos) <= self._max_length:
                current_buffer.extend(seq_with_eos)
            else:
                if current_buffer:
                    # Pad current buffer to max_length
                    padded = current_buffer + [self._pad_token_id] * (self._max_length - len(current_buffer))
                    packed.append(padded[:self._max_length])
                # Start new buffer
                current_buffer = list(seq)
                if self._eos_token_id is not None:
                    current_buffer.append(self._eos_token_id)

        # Flush remaining buffer
        if current_buffer:
            if len(current_buffer) < self._max_length:
                padded = current_buffer + [self._pad_token_id] * (self._max_length - len(current_buffer))
                packed.append(padded[:self._max_length])
            else:
                packed.append(current_buffer[:self._max_length])

        return packed

    def pack_with_attention_mask(self, sequences: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """Pack sequences and generate attention masks.

        Returns (packed_sequences, attention_masks).
        Attention mask marks actual tokens (1) vs padding (0) and
        also separates different sequences packed together.
        """
        packed = self.pack(sequences)
        attention_masks = []
        for seq in packed:
            mask = [1 if tid != self._pad_token_id else 0 for tid in seq]
            attention_masks.append(mask)
        return packed, attention_masks
