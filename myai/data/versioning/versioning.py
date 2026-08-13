"""Data versioning: tracks exactly which dataset snapshot was used for each training run."""

import json
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from ...core.logging import logger


class DataVersionTracker:
    """Tracks dataset versions for full training reproducibility.

    Stores file hashes, configuration snapshots, and timestamps
    in checkpoints so training can be fully reproduced.
    """

    def __init__(self, data_paths: Optional[List[str]] = None):
        self._data_paths = data_paths or []
        self._file_hashes: Dict[str, str] = {}
        self._snapshot_time: float = 0.0

    def _hash_file(self, path: str) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def snapshot(self, data_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a version snapshot of the current dataset state."""
        paths = data_paths or self._data_paths
        self._snapshot_time = time.time()

        for path in paths:
            p = Path(path)
            if p.exists():
                if p.is_file():
                    self._file_hashes[str(p.resolve())] = self._hash_file(str(p.resolve()))
                elif p.is_dir():
                    for f in sorted(p.rglob("*")):
                        if f.is_file():
                            self._file_hashes[str(f.resolve())] = self._hash_file(str(f.resolve()))

        snapshot = {
            "timestamp": self._snapshot_time,
            "data_paths": paths,
            "num_files": len(self._file_hashes),
            "total_hash": hashlib.sha256(
                json.dumps(self._file_hashes, sort_keys=True).encode()
            ).hexdigest(),
        }

        return snapshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_paths": self._data_paths,
            "file_hashes": self._file_hashes,
            "snapshot_time": self._snapshot_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataVersionTracker":
        tracker = cls(data_paths=data.get("data_paths", []))
        tracker._file_hashes = data.get("file_hashes", {})
        tracker._snapshot_time = data.get("snapshot_time", 0.0)
        return tracker
