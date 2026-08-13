"""Dataset indexing for fast random access without loading entire files."""

import json
import struct
import mmap
from pathlib import Path
from typing import List, Optional, Iterator, Tuple, Dict, Any
from ...core.logging import logger


class DatasetIndex:
    """Binary index for fast random access to tokenized samples."""

    def __init__(self, index_path: Optional[str] = None):
        self._offsets: List[int] = []
        self._lengths: List[int] = []
        self._metadata: Dict[str, Any] = {}
        self._index_path = index_path

        if index_path and Path(index_path).exists():
            self.load(index_path)

    def build_from_sequences(self, sequences: List[List[int]], index_path: str) -> None:
        """Build an index from a list of token sequences and write to disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        offset = 0
        offsets = []
        lengths = []
        data_path = index_path.replace(".idx", ".bin")

        with open(data_path, "wb") as f:
            for seq in sequences:
                offsets.append(offset)
                lengths.append(len(seq))
                data = struct.pack(f"<{len(seq)}i", *seq)
                f.write(data)
                offset += len(data)

        self._offsets = offsets
        self._lengths = lengths
        self._metadata = {"num_sequences": len(sequences), "data_path": data_path}

        # Write index file
        with open(index_path, "w") as f:
            json.dump({
                "offsets": offsets,
                "lengths": lengths,
                "metadata": self._metadata,
            }, f)

        logger.info(f"Built index with {len(sequences)} sequences at {index_path}")

    def load(self, index_path: str) -> None:
        """Load index from file."""
        with open(index_path) as f:
            data = json.load(f)
        self._offsets = data["offsets"]
        self._lengths = data["lengths"]
        self._metadata = data.get("metadata", {})
        self._index_path = index_path

    def get_sequence(self, index: int) -> List[int]:
        """Retrieve a sequence by index using memory-mapped file access."""
        if index < 0 or index >= len(self._offsets):
            raise IndexError(f"Index {index} out of range [0, {len(self._offsets)})")

        data_path = self._metadata.get("data_path")
        if not data_path or not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        offset = self._offsets[index]
        length = self._lengths[index]

        with open(data_path, "rb") as f:
            f.seek(offset)
            data = f.read(length * 4)  # 4 bytes per int32
            return list(struct.unpack(f"<{length}i", data))

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> List[int]:
        return self.get_sequence(idx)

    @property
    def num_sequences(self) -> int:
        return len(self._offsets)
