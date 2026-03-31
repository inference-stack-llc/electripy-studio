"""ElectriPy - The Python substrate for observable agent engineering."""

__version__ = "0.1.0"

from electripy.core.config import Config
from electripy.core.errors import ElectriPyError
from electripy.core.logging import get_logger, setup_logging

from . import ai

__all__ = [
    "Config",
    "ElectriPyError",
    "get_logger",
    "setup_logging",
    "ai",
    "__version__",
]
