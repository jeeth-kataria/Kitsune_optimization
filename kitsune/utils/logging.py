"""
Logging utilities for Kitsune

Provides structured logging with support for both console and file output.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Module-level logger
_logger: Optional[logging.Logger] = None
_configured: bool = False


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log output for terminal display."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str = None, use_colors: bool = True):
        super().__init__(fmt or "[%(levelname)s] %(name)s: %(message)s")
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors and record.levelno in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelno]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


def configure_logging(
    level: LogLevel | str = LogLevel.WARNING,
    log_file: Optional[Path | str] = None,
    use_colors: bool = True,
    format_string: str = None,
) -> logging.Logger:
    """
    Configure Kitsune logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_colors: Whether to use colored output in terminal
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    global _logger, _configured

    # Convert string to LogLevel if needed
    if isinstance(level, str):
        level = LogLevel[level.upper()]

    # Get or create logger
    logger = logging.getLogger("kitsune")
    logger.setLevel(level.value)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level.value)

    if format_string:
        console_handler.setFormatter(logging.Formatter(format_string))
    else:
        console_handler.setFormatter(ColoredFormatter(use_colors=use_colors))

    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level.value)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _logger = logger
    _configured = True

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a Kitsune logger instance.

    Args:
        name: Optional sub-logger name (e.g., "scheduler", "memory")

    Returns:
        Logger instance
    """
    global _logger, _configured

    if not _configured:
        configure_logging()

    if name:
        return logging.getLogger(f"kitsune.{name}")
    return logging.getLogger("kitsune")


# Convenience functions
def debug(msg: str, *args, **kwargs) -> None:
    """Log debug message."""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log info message."""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log warning message."""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log error message."""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log critical message."""
    get_logger().critical(msg, *args, **kwargs)
