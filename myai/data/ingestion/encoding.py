"""Encoding detection for reading text files with unknown encodings."""

import codecs
from typing import Optional


# Common encodings to try in order
ENCODING_PRIORITY = [
    "utf-8",
    "utf-16-le",
    "utf-16-be",
    "latin-1",
    "cp1252",
    "iso-8859-1",
    "shift-jis",
    "euc-jp",
    "euc-kr",
    "gbk",
    "gb2312",
    "big5",
    "cp437",
]


class EncodingDetector:
    """Detects text encoding by trying common encodings."""

    @staticmethod
    def detect(file_path: str, sample_size: int = 8192) -> str:
        """Detect encoding by trying to decode a sample."""
        with open(file_path, "rb") as f:
            raw = f.read(sample_size)

        # Check for BOM
        if raw.startswith(codecs.BOM_UTF8):
            return "utf-8-sig"
        if raw.startswith(codecs.BOM_UTF16_LE):
            return "utf-16-le"
        if raw.startswith(codecs.BOM_UTF16_BE):
            return "utf-16-be"

        # Try each encoding
        for enc in ENCODING_PRIORITY:
            try:
                raw.decode(enc)
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue

        return "utf-8"  # fallback

    @staticmethod
    def detect_and_read(file_path: str) -> str:
        """Detect encoding and read the full file."""
        enc = EncodingDetector.detect(file_path)
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except Exception:
            # Ultimate fallback
            with open(file_path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
