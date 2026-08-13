"""Logging system with timestamp, severity, and contextual information."""

import sys
import os
import time
import traceback
from enum import Enum
from typing import IO, Optional


class LogLevel(Enum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    FATAL = 5

    def __ge__(self, other) -> bool:
        if isinstance(other, LogLevel):
            return self.value >= other.value
        return NotImplemented


_LEVEL_NAMES = {
    LogLevel.TRACE: "TRACE",
    LogLevel.DEBUG: "DEBUG",
    LogLevel.INFO: "INFO",
    LogLevel.WARNING: "WARNING",
    LogLevel.ERROR: "ERROR",
    LogLevel.FATAL: "FATAL",
}


class Logger:
    """Structured logger with severity levels and context."""

    def __init__(
        self,
        name: str = "myai",
        min_level: LogLevel = LogLevel.INFO,
        output: Optional[IO[str]] = None,
        log_file: Optional[str] = None,
    ):
        self._name = name
        self._min_level = min_level
        self._output = output or sys.stdout
        self._log_file: Optional[IO[str]] = None
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            self._log_file = open(log_file, "a", encoding="utf-8")

    def _format(self, level: LogLevel, message: str, context: Optional[dict] = None) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        level_str = _LEVEL_NAMES.get(level, "UNKNOWN")
        ctx_str = ""
        if context:
            ctx_str = " | " + " ".join(f"{k}={v}" for k, v in context.items())
        return f"[{timestamp}] [{level_str}] [{self._name}] {message}{ctx_str}"

    def _write(self, level: LogLevel, message: str, context: Optional[dict] = None) -> None:
        if level.value < self._min_level.value:
            return
        line = self._format(level, message, context) + "\n"
        self._output.write(line)
        self._output.flush()
        if self._log_file:
            self._log_file.write(line)
            self._log_file.flush()

    def trace(self, message: str, **context) -> None:
        self._write(LogLevel.TRACE, message, context)

    def debug(self, message: str, **context) -> None:
        self._write(LogLevel.DEBUG, message, context)

    def info(self, message: str, **context) -> None:
        self._write(LogLevel.INFO, message, context)

    def warning(self, message: str, **context) -> None:
        self._write(LogLevel.WARNING, message, context)

    def error(self, message: str, **context) -> None:
        self._write(LogLevel.ERROR, message, context)

    def fatal(self, message: str, **context) -> None:
        self._write(LogLevel.FATAL, message, context)

    def exception(self, message: str, exc: Optional[BaseException] = None, **context) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else traceback.format_exc()
        self._write(LogLevel.ERROR, f"{message}\n{tb}", context)

    def set_level(self, level: LogLevel) -> None:
        self._min_level = level

    def close(self) -> None:
        if self._log_file:
            self._log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


logger = Logger()
