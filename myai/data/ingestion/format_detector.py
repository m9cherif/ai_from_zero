"""Format detection for different file types."""

from pathlib import Path
from typing import Optional


class FormatDetector:
    """Detects the format of a data file by extension and content inspection."""

    FORMATS = {
        ".txt": "text",
        ".json": "json",
        ".jsonl": "jsonl",
        ".csv": "csv",
        ".tsv": "tsv",
        ".html": "html",
        ".htm": "html",
        ".xml": "xml",
        ".md": "markdown",
        ".rst": "rst",
    }

    def detect(self, file_path: str) -> str:
        """Detect the format of a file."""
        p = Path(file_path)
        ext = p.suffix.lower()
        fmt = self.FORMATS.get(ext, "text")

        # Content-based detection for ambiguous cases
        if fmt == "jsonl":
            return "jsonl"
        if fmt == "json":
            # Check if it might be JSONL
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline()
                if first_line.strip().startswith("{"):
                    f.seek(0)
                    second_line = f.readline()
                    if second_line.strip().startswith("{"):
                        return "jsonl"
            except Exception:
                pass

        return fmt
