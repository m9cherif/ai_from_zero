"""Shard management for distributed or large-scale datasets."""

import os
import json
import math
import random
from pathlib import Path
from typing import List, Optional, Iterator, Any, Dict
from ...core.logging import logger


class ShardManager:
    """Manages dataset shards for distributed processing and large datasets."""

    def __init__(
        self,
        shard_dir: str,
        num_shards: int = 1,
        shard_size: int = 10000,
        shard_extension: str = ".jsonl",
    ):
        self._shard_dir = Path(shard_dir)
        self._num_shards = num_shards
        self._shard_size = shard_size
        self._shard_extension = shard_extension
        self._shard_metadata: Dict[str, Any] = {}

        self._shard_dir.mkdir(parents=True, exist_ok=True)

    def _shard_path(self, shard_index: int) -> Path:
        return self._shard_dir / f"shard_{shard_index:06d}{self._shard_extension}"

    def create_shards(self, data_iterator: Iterator[str], prefix: str = "shard") -> List[Path]:
        """Split data stream into shards and write to disk."""
        shard_paths = []
        current_shard = 0
        current_count = 0
        current_file = None

        for item in data_iterator:
            if current_file is None:
                path = self._shard_path(current_shard)
                current_file = open(path, "w", encoding="utf-8")
                shard_paths.append(path)
                current_count = 0

            current_file.write(json.dumps({"text": item}, ensure_ascii=False) + "\n")
            current_count += 1

            if current_count >= self._shard_size:
                current_file.close()
                current_file = None
                current_shard += 1

        if current_file:
            current_file.close()

        self._shard_metadata[prefix] = {
            "num_shards": len(shard_paths),
            "paths": [str(p) for p in shard_paths],
        }

        logger.info(f"Created {len(shard_paths)} shards in {self._shard_dir}")
        return shard_paths

    def get_shard_paths(self, shuffle: bool = False) -> List[Path]:
        """Get all shard file paths."""
        paths = sorted(self._shard_dir.glob(f"*{self._shard_extension}"))
        if shuffle:
            random.shuffle(paths)
        return paths

    def read_shard(self, shard_path: Path) -> Iterator[str]:
        """Read a single shard, yielding text records."""
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue

    def shard_count(self) -> int:
        return len(self.get_shard_paths())

    def save_metadata(self, path: Optional[str] = None) -> None:
        path = path or str(self._shard_dir / "shard_metadata.json")
        with open(path, "w") as f:
            json.dump(self._shard_metadata, f, indent=2)

    def load_metadata(self, path: Optional[str] = None) -> Dict[str, Any]:
        path = path or str(self._shard_dir / "shard_metadata.json")
        with open(path) as f:
            self._shard_metadata = json.load(f)
        return self._shard_metadata
