"""Safe reading of various text formats."""

import csv
import json
from pathlib import Path
from typing import Iterator, Optional, List, Any, Dict
from ...core.errors import DataError
from ...core.logging import logger
from .encoding import EncodingDetector


class TextReader:
    """Reads text from files in various formats."""

    def __init__(self, encoding_detector: Optional[EncodingDetector] = None):
        self._encoding_detector = encoding_detector or EncodingDetector()

    def read_lines(self, file_path: str) -> Iterator[str]:
        """Read a file line by line, handling encoding."""
        try:
            enc = self._encoding_detector.detect(file_path)
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                for line in f:
                    yield line
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise DataError(f"Cannot read file {file_path}: {e}") from e

    def read_text(self, file_path: str) -> Optional[str]:
        """Read an entire text file."""
        try:
            return self._encoding_detector.detect_and_read(file_path)
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return None

    def read_json(self, file_path: str) -> Optional[Any]:
        """Read a JSON file."""
        try:
            text = self.read_text(file_path)
            if text:
                return json.loads(text)
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {file_path}: {e}")
            return None

    def read_jsonl(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """Read a JSONL file, yielding parsed JSON objects."""
        for line in self.read_lines(file_path):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line in {file_path}")

    def read_csv(self, file_path: str, delimiter: str = ",") -> Iterator[Dict[str, str]]:
        """Read a CSV/TSV file, yielding dictionaries."""
        text = self.read_text(file_path)
        if text is None:
            return
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        for row in reader:
            yield row

    def read_html(self, file_path: str) -> Optional[str]:
        """Extract text from an HTML file (simple tag stripping)."""
        import re
        text = self.read_text(file_path)
        if text is None:
            return None
        text = re.sub(r'<head>.*?</head>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def read_as_text(self, file_path: str, format: Optional[str] = None) -> Iterator[str]:
        """Read a file and yield text segments based on format."""
        if format is None:
            ext = Path(file_path).suffix.lower()
            format = {
                ".json": "json",
                ".jsonl": "jsonl",
                ".csv": "csv",
                ".tsv": "tsv",
                ".html": "html",
                ".htm": "html",
            }.get(ext, "text")

        try:
            if format == "text":
                yield self.read_text(file_path) or ""
            elif format == "json":
                data = self.read_json(file_path)
                if data:
                    if isinstance(data, dict):
                        for v in data.values():
                            if isinstance(v, str):
                                yield v
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                yield item
                            elif isinstance(item, dict):
                                for v in item.values():
                                    if isinstance(v, str):
                                        yield v
            elif format == "jsonl":
                for obj in self.read_jsonl(file_path):
                    for v in obj.values():
                        if isinstance(v, str):
                            yield v
            elif format in ("csv", "tsv"):
                delim = "\t" if format == "tsv" else ","
                for row in self.read_csv(file_path, delimiter=delim):
                    for v in row.values():
                        if v:
                            yield v
            elif format == "html":
                text = self.read_html(file_path)
                if text:
                    yield text
            else:
                yield self.read_text(file_path) or ""
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
