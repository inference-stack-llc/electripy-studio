"""Custom exceptions for ElectriPy."""


class ElectriPyError(Exception):
    """Base exception for all ElectriPy errors."""

    pass


class ConfigError(ElectriPyError):
    """Raised when there's a configuration error."""

    pass


class ValidationError(ElectriPyError):
    """Raised when validation fails."""

    pass


class RetryError(ElectriPyError):
    """Raised when retry attempts are exhausted."""

    pass


class RateLimitError(ElectriPyError):
    """Raised when rate limit is exceeded."""

    pass
