"""Tests for core.logging module."""

import logging

from electripy.core.logging import get_logger, setup_logging


def test_setup_logging_default() -> None:
    """Test setup_logging with defaults."""
    setup_logging()
    logger = logging.getLogger()
    assert logger.level == logging.INFO


def test_setup_logging_debug() -> None:
    """Test setup_logging with DEBUG level."""
    setup_logging(level="DEBUG")
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG


def test_get_logger() -> None:
    """Test get_logger creates logger with correct name."""
    logger = get_logger("test_module")
    assert logger.name == "test_module"
    assert isinstance(logger, logging.Logger)
