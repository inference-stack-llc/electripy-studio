"""Tests for core.errors module."""

import pytest

from electripy.core.errors import (
    ConfigError,
    ElectriPyError,
    RateLimitError,
    RetryError,
    ValidationError,
)


def test_electripy_error() -> None:
    """Test base ElectriPyError."""
    with pytest.raises(ElectriPyError):
        raise ElectriPyError("Test error")


def test_config_error() -> None:
    """Test ConfigError inheritance."""
    with pytest.raises(ElectriPyError):
        raise ConfigError("Config error")


def test_validation_error() -> None:
    """Test ValidationError inheritance."""
    with pytest.raises(ElectriPyError):
        raise ValidationError("Validation error")


def test_retry_error() -> None:
    """Test RetryError inheritance."""
    with pytest.raises(ElectriPyError):
        raise RetryError("Retry error")


def test_rate_limit_error() -> None:
    """Test RateLimitError inheritance."""
    with pytest.raises(ElectriPyError):
        raise RateLimitError("Rate limit error")
