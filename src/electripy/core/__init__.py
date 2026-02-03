"""Core module: Config, Logging, Errors, Typing utilities."""

from electripy.core.config import Config
from electripy.core.errors import ConfigError, ElectriPyError, ValidationError
from electripy.core.logging import get_logger, setup_logging
from electripy.core.typing import JSONDict, JSONValue

__all__ = [
    "Config",
    "ElectriPyError",
    "ConfigError",
    "ValidationError",
    "get_logger",
    "setup_logging",
    "JSONDict",
    "JSONValue",
]
