"""Domain models and exceptions for the HTTP resilience client.

This module defines the core data structures and error types used by the
`http_resilience_client` component. It is intentionally free of any
third-party HTTP client dependencies.

The main guarantees are:
- Response data is normalized into a small, repository-owned model.
- All public errors are expressed as ElectriPy domain exceptions.
- Configuration for retries and circuit breaking is explicit and typed.

Example:
    from electripy.concurrency.http_resilience.domain import (
        ResponseData,
        RetryPolicy,
        CircuitBreakerConfig,
    )

    policy = RetryPolicy(max_attempts=3)
    breaker_config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=30.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, MutableMapping, Tuple, TypeAlias

from electripy.core.errors import ElectriPyError


Headers: TypeAlias = Mapping[str, str]
MutableHeaders: TypeAlias = MutableMapping[str, str]
QueryParams: TypeAlias = Mapping[str, str]
BodyType: TypeAlias = bytes | str | Mapping[str, object] | None


class HttpMethod(str, Enum):
    """HTTP methods supported by the resilience client.

    This enum is used for idempotency decisions and logging. It is
    intentionally small but can be extended in a backwards-compatible way.

    Example:
        method = HttpMethod.GET
    """

    GET = "GET"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass(slots=True)
class ResponseData:
    """Normalized HTTP response data.

    This model captures only the fields needed by callers and higher-level
    orchestration. The underlying HTTP client's response object is never
    exposed outside adapters.

    Attributes:
        status_code: Numeric HTTP status code.
        headers: Response headers as a case-insensitive mapping (normalized to
            a plain dict of lower-cased keys by adapters).
        body: Raw response body bytes.
        url: Final URL of the response, including redirects.
        elapsed_s: Optional elapsed time in seconds, if available.

    Example:
        response = ResponseData(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b"{}",
            url="https://api.example.com/resource",
            elapsed_s=0.123,
        )
    """

    status_code: int
    headers: Headers
    body: bytes
    url: str
    elapsed_s: float | None = None

    @property
    def is_success(self) -> bool:
        """Return True if the response status code is in the 2xx range.

        Returns:
            bool: True for 2xx status codes, False otherwise.

        Example:
            assert ResponseData(200, {}, b"", "").is_success is True
        """

        return 200 <= self.status_code < 300


class HttpClientError(ElectriPyError):
    """Base exception for HTTP client failures.

    All public errors raised by the resilience client derive from this base
    class. Callers can catch this type to handle all HTTP-related failures in
    a single place.

    Example:
        try:
            client.request("GET", "https://api.example.com")
        except HttpClientError as exc:
            handle_error(exc)
    """


class CircuitOpenError(HttpClientError):
    """Raised when the circuit breaker is open and calls are denied.

    Example:
        try:
            client.request("GET", "https://api.example.com")
        except CircuitOpenError:
            # Fail fast or route to a fallback
            ...
    """


class RetryExhaustedError(HttpClientError):
    """Raised when all retry attempts have been exhausted.

    Attributes:
        last_response: Optional last HTTP response, if the failure was due to
            non-success status codes rather than transport-level errors.

    Example:
        try:
            client.request("GET", "https://api.example.com")
        except RetryExhaustedError as exc:
            log_failure(exc.last_response)
    """

    def __init__(self, message: str, last_response: ResponseData | None = None) -> None:
        super().__init__(message)
        self.last_response = last_response


class TransientHttpError(HttpClientError):
    """Error type used for transient, retryable HTTP failures.

    Adapters should map low-level network and timeout errors into this type so
    the retry policy can treat them specially. Callers typically do not
    construct this directly.

    Example:
        raise TransientHttpError("Temporary network failure")
    """


@dataclass(slots=True)
class RetryPolicy:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts, including the initial
            request. Must be >= 1.
        base_delay_s: Base delay in seconds used for the first backoff
            interval.
        max_delay_s: Maximum delay in seconds between attempts.
        jitter_ratio: Jitter ratio in the range [0.0, 1.0]. A value of 0.0
            disables jitter. A value of 1.0 allows delays in the range
            [0.0, 2 * delay].
        retryable_status_codes: HTTP status codes that are eligible for
            retries (e.g., 500s, 429).
        retryable_exceptions: Exception types that should be considered
            retryable. By default, this includes :class:`TransientHttpError`.
        honor_retry_after_header: Whether to honor ``Retry-After`` headers
            (in seconds) when present on responses.
        allow_retry_non_idempotent: Whether non-idempotent methods (e.g.,
            POST) may be retried by default. It is safer to keep this False
            and opt in per-call when using idempotency keys.

    Example:
        policy = RetryPolicy(
            max_attempts=3,
            base_delay_s=0.1,
            max_delay_s=5.0,
        )
    """

    max_attempts: int = 3
    base_delay_s: float = 0.1
    max_delay_s: float = 10.0
    jitter_ratio: float = 0.2
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 425, 429, 500, 502, 503, 504})
    )
    retryable_exceptions: Tuple[type[Exception], ...] = field(
        default_factory=lambda: (TransientHttpError,)
    )
    honor_retry_after_header: bool = True
    allow_retry_non_idempotent: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration values are invalid.
        """

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_s < 0:
            raise ValueError("base_delay_s must be >= 0")
        if self.max_delay_s <= 0:
            raise ValueError("max_delay_s must be > 0")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio must be between 0.0 and 1.0")


@dataclass(slots=True)
class CircuitBreakerConfig:
    """Configuration for the circuit breaker.

    Attributes:
        failure_threshold: Number of consecutive failures required to open the
            circuit.
        recovery_timeout_s: Time in seconds the circuit remains open before
            transitioning to HALF_OPEN.
        half_open_max_calls: Maximum number of trial calls allowed while in
            HALF_OPEN state.
        half_open_success_threshold: Number of successful trial calls in
            HALF_OPEN required to close the circuit.

    Example:
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_s=30.0,
            half_open_max_calls=3,
            half_open_success_threshold=2,
        )
    """

    failure_threshold: int
    recovery_timeout_s: float
    half_open_max_calls: int = 1
    half_open_success_threshold: int = 1

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration values are invalid.
        """

        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be > 0")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")
        if self.half_open_success_threshold < 1:
            raise ValueError("half_open_success_threshold must be >= 1")


__all__ = [
    "Headers",
    "MutableHeaders",
    "QueryParams",
    "BodyType",
    "HttpMethod",
    "ResponseData",
    "HttpClientError",
    "CircuitOpenError",
    "RetryExhaustedError",
    "TransientHttpError",
    "RetryPolicy",
    "CircuitBreakerConfig",
]
