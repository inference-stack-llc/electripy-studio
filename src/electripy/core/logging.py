"""Logging utilities for ElectriPy."""

import logging
import sys


def setup_logging(level: str = "INFO", format_type: str = "json") -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json' or 'text')
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if format_type == "json":
        fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    else:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Name for the logger, typically __name__

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
