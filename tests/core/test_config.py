"""Tests for core.config module."""

import os
from pathlib import Path

from electripy.core.config import Config


def test_config_defaults() -> None:
    """Test Config with default values."""
    config = Config()
    assert config.log_level == "INFO"
    assert config.log_format == "json"
    assert config.config_dir == Path.home() / ".electripy"


def test_config_from_env(monkeypatch: object) -> None:
    """Test Config.from_env() with environment variables."""
    monkeypatch.setattr(os, "getenv", lambda k, d: {
        "ELECTRIPY_LOG_LEVEL": "DEBUG",
        "ELECTRIPY_LOG_FORMAT": "text",
        "ELECTRIPY_CONFIG_DIR": "/tmp/config",
    }.get(k, d))
    
    config = Config.from_env()
    assert config.log_level == "DEBUG"
    assert config.log_format == "text"
    assert config.config_dir == Path("/tmp/config")


def test_config_to_dict() -> None:
    """Test Config.to_dict() serialization."""
    config = Config(log_level="WARNING", log_format="text")
    data = config.to_dict()
    
    assert data["log_level"] == "WARNING"
    assert data["log_format"] == "text"
    assert "config_dir" in data
