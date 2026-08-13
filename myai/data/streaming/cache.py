"""Preprocessing cache to avoid recomputing expensive transformations."""

import json
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Any, Callable, Dict
from ...core.logging import logger


class PreprocessingCache:
    """Versioned cache for preprocessing results.

    Automatically invalidates cached data when preprocessing logic changes
    by incorporating a version hash into the cache key.
    """

    def __init__(self, cache_dir: str, version: str = "1.0"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._version = version

    def _compute_key(self, data_id: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Compute a cache key from data ID, version, and parameters."""
        key_parts = [self._version, data_id]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.cache"

    def get(self, data_id: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Retrieve cached result if available."""
        key = self._compute_key(data_id, params)
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                logger.warning(f"Corrupted cache file: {path}")
                path.unlink()
        return None

    def set(self, data_id: str, value: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Store a result in the cache."""
        key = self._compute_key(data_id, params)
        path = self._cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_or_compute(self, data_id: str, compute_fn: Callable[[], Any], params: Optional[Dict[str, Any]] = None) -> Any:
        """Get from cache or compute and cache."""
        cached = self.get(data_id, params)
        if cached is not None:
            return cached
        result = compute_fn()
        self.set(data_id, result, params)
        return result

    def invalidate(self, data_id: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if data_id:
            key = self._compute_key(data_id)
            path = self._cache_path(key)
            if path.exists():
                path.unlink()
        else:
            for f in self._cache_dir.glob("*.cache"):
                f.unlink()
            logger.info(f"Cleared all cache in {self._cache_dir}")

    def clear(self) -> None:
        self.invalidate()
