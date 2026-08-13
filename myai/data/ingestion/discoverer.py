"""Source discovery: traverses directories and finds data files."""

import os
from pathlib import Path
from typing import List, Optional, Set
from ...core.logging import logger


SUPPORTED_EXTENSIONS = {
    ".txt", ".json", ".jsonl", ".csv", ".tsv",
    ".html", ".htm", ".xml", ".md", ".rst",
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
    ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".log", ".sql", ".sh", ".bat", ".ps1",
    ".tex", ".bib",
}


class DataDiscoverer:
    """Discovers and filters data files in given paths."""

    def __init__(
        self,
        extensions: Optional[Set[str]] = None,
        exclude_dirs: Optional[Set[str]] = None,
        max_file_size_mb: float = 100.0,
    ):
        self._extensions = extensions or SUPPORTED_EXTENSIONS
        self._exclude_dirs = exclude_dirs or {"__pycache__", ".git", ".svn", "node_modules", ".venv", "venv"}
        self._max_file_size_mb = max_file_size_mb

    def discover(self, paths: List[str]) -> List[str]:
        """Discover all supported data files in the given paths."""
        discovered = []
        for path in paths:
            p = Path(path)
            if p.is_file():
                if self._is_supported(p):
                    discovered.append(str(p.resolve()))
            elif p.is_dir():
                self._discover_dir(p, discovered)
            else:
                logger.warning(f"Path does not exist: {path}")
        logger.info(f"Discovered {len(discovered)} files from {len(paths)} paths")
        return discovered

    def _discover_dir(self, directory: Path, results: List[str]) -> None:
        """Recursively discover files in a directory."""
        try:
            for entry in os.scandir(directory):
                if entry.is_dir():
                    if entry.name not in self._exclude_dirs:
                        self._discover_dir(Path(entry.path), results)
                elif entry.is_file():
                    p = Path(entry.path)
                    if self._is_supported(p):
                        results.append(str(p.resolve()))
        except PermissionError as e:
            logger.warning(f"Permission denied: {directory}", error=str(e))

    def _is_supported(self, file_path: Path) -> bool:
        """Check if a file is supported based on extension and size."""
        ext = file_path.suffix.lower()
        if ext not in self._extensions:
            return False
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self._max_file_size_mb:
                logger.debug(f"Skipping large file: {file_path} ({size_mb:.1f} MB)")
                return False
        except OSError:
            return False
        return True
