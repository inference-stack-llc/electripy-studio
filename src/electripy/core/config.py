"""Configuration management for ElectriPy."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Config:
    """Application configuration with environment variable support."""

    log_level: str = field(default="INFO")
    log_format: str = field(default="json")
    config_dir: Path = field(default_factory=lambda: Path.home() / ".electripy")

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.

        Returns:
            Config instance populated from environment variables.
        """
        log_level = os.getenv("ELECTRIPY_LOG_LEVEL", "INFO")
        log_format = os.getenv("ELECTRIPY_LOG_FORMAT", "json")
        config_dir = Path(os.getenv("ELECTRIPY_CONFIG_DIR", str(Path.home() / ".electripy")))

        return cls(
            log_level=log_level,
            log_format=log_format,
            config_dir=config_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config.
        """
        return {
            "log_level": self.log_level,
            "log_format": self.log_format,
            "config_dir": str(self.config_dir),
        }
