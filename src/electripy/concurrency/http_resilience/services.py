"""Service layer for resilient HTTP clients.

This module orchestrates retries, exponential backoff with jitter, and
circuit breaking on top of the abstract HTTP ports.

Key design points:
- Only depends on Protocol ports and domain models.
- Implements both synchronous and asynchronous resilient clients.
- Enforces explicit timeouts and bounded retries.

Example:
    from electripy.concurrency.http_resilience import (
        CircuitBreaker,
        CircuitBreakerConfig,
        HttpxSyncAdapter,
        ResilientSyncHttpClient,
        RetryPolicy,
    )

    adapter = HttpxSyncAdapter(base_url="https://api.example.com")
    policy = RetryPolicy(max_attempts=3)
    breaker = CircuitBreaker(CircuitBreakerConfig(5, 30.0))
    client = ResilientSyncHttpClient(adapter, policy, breaker)
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from electripy.core.logging import get_logger

from .circuit_breaker import CircuitBreaker
from .domain import (
    BodyType,
    HttpClientError,
    HttpMethod,
    QueryParams,
    ResponseData,
    RetryExhaustedError,
    RetryPolicy,
)
from .ports import AsyncHttpPort, SyncHttpPort

logger = get_logger(__name__)


def _is_idempotent_method(method: str) -> bool:
    """Return True if an HTTP method is considered idempotent.

    Args:
        method: HTTP method string.

    Returns:
        bool: True for methods treated as safe to retry by default.
    """

    try:
        method_enum = HttpMethod(method.upper())
    except ValueError:
        return False
    return method_enum in {HttpMethod.GET, HttpMethod.HEAD, HttpMethod.OPTIONS}


def _calculate_delay_s(
    *,
    policy: RetryPolicy,
    attempt_index: int,
    last_response: ResponseData | None,
) -> float:
    """Calculate the delay before the next retry attempt.

    The delay is computed using exponential backoff with optional jitter. If
    ``Retry-After`` is present and ``honor_retry_after_header`` is enabled, it
    is used as the base delay (in seconds) and capped by ``max_delay_s``.

    Args:
        policy: Retry policy configuration.
        attempt_index: Zero-based attempt index (0 for the first retry after
            the initial attempt).
        last_response: Optional last HTTP response.

    Returns:
        float: Delay in seconds.
    """

    base_delay = policy.base_delay_s * (2**attempt_index)
    delay = min(base_delay, policy.max_delay_s)

    if policy.honor_retry_after_header and last_response is not None:
        retry_after_raw = last_response.headers.get("retry-after")
        if retry_after_raw is not None:
            try:
                retry_after_s = float(retry_after_raw)
                delay = min(retry_after_s, policy.max_delay_s)
            except ValueError:
                logger.debug("Invalid Retry-After header value", extra={"value": retry_after_raw})

    result = delay

    if policy.jitter_ratio > 0.0:
        low = max(0.0, delay * (1.0 - policy.jitter_ratio))
        high = delay * (1.0 + policy.jitter_ratio)
        result = float(random.uniform(low, high))

    return result  # type: ignore[no-any-return]


@dataclass(slots=True)
class ResilientSyncHttpClient:
    """Resilient synchronous HTTP client.

    This client composes a synchronous HTTP port, retry policy, and circuit
    breaker to provide a hardened outbound HTTP client.

    Args:
        port: Synchronous HTTP port implementation.
        retry_policy: Configuration for retries and backoff.
        circuit_breaker: Circuit breaker instance.
        default_timeout_s: Default per-request timeout in seconds.
        sleep_fn: Optional sleep function, primarily for testing.

    Example:
        client = ResilientSyncHttpClient(port, retry_policy, circuit_breaker)
        response = client.request("GET", "https://api.example.com/resource")
    """

    port: SyncHttpPort
    retry_policy: RetryPolicy
    circuit_breaker: CircuitBreaker
    default_timeout_s: float = 5.0
    sleep_fn: Callable[[float], None] = time.sleep

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: QueryParams | None = None,
        json: object | None = None,
        data: BodyType | None = None,
        timeout_s: float | None = None,
        allow_retry_for_non_idempotent: bool | None = None,
    ) -> ResponseData:
        """Perform a resilient synchronous HTTP request.

        This method enforces idempotency rules: non-idempotent methods such as
        POST are not retried unless explicitly opted-in via
        ``allow_retry_for_non_idempotent`` or the retry policy.

        Args:
            method: HTTP method string.
            url: Absolute or relative URL.
            headers: Optional request headers.
            params: Optional query parameters.
            json: Optional JSON-serializable payload.
            data: Optional raw body or form data.
            timeout_s: Optional per-request timeout in seconds.
            allow_retry_for_non_idempotent: Override for retrying
                non-idempotent methods on this particular call.

        Returns:
            ResponseData: Normalized response data.

        Raises:
            CircuitOpenError: If the circuit breaker denies the call.
            RetryExhaustedError: If all retry attempts fail.
            HttpClientError: For other HTTP client errors.
        """

        is_idempotent = _is_idempotent_method(method)
        allow_retry_non_idempotent = (
            self.retry_policy.allow_retry_non_idempotent
            if allow_retry_for_non_idempotent is None
            else allow_retry_for_non_idempotent
        )

        max_attempts = self.retry_policy.max_attempts
        timeout = timeout_s or self.default_timeout_s
        last_response: ResponseData | None = None

        for attempt in range(max_attempts):
            try:
                self.circuit_breaker.before_call()
            except HttpClientError:
                # CircuitOpenError is a subclass; just propagate.
                raise

            try:
                response = self.port.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    data=data,
                    timeout_s=timeout,
                )
            except HttpClientError as exc:
                logger.warning(
                    "HTTP client error during request",
                    extra={"method": method, "url": url, "attempt": attempt + 1},
                )
                self.circuit_breaker.after_call(False)

                is_retryable_exc = isinstance(exc, self.retry_policy.retryable_exceptions)
                is_last_attempt = attempt >= max_attempts - 1

                if not is_retryable_exc or is_last_attempt:
                    raise RetryExhaustedError(
                        "All retry attempts failed", last_response=None
                    ) from exc

                delay_s = _calculate_delay_s(
                    policy=self.retry_policy,
                    attempt_index=attempt,
                    last_response=None,
                )
                self.sleep_fn(delay_s)
                continue

            last_response = response

            if response.is_success:
                self.circuit_breaker.after_call(True)
                return response

            is_retryable_status = response.status_code in self.retry_policy.retryable_status_codes
            is_last_attempt = attempt >= max_attempts - 1

            if (
                not is_retryable_status
                or (not is_idempotent and not allow_retry_non_idempotent)
                or is_last_attempt
            ):
                self.circuit_breaker.after_call(False)
                raise RetryExhaustedError(
                    f"All retry attempts failed with status {response.status_code}",
                    last_response=response,
                )

            logger.info(
                "Retrying HTTP request due to response status",
                extra={
                    "method": method,
                    "url": url,
                    "status_code": response.status_code,
                    "attempt": attempt + 1,
                },
            )
            self.circuit_breaker.after_call(False)

            delay_s = _calculate_delay_s(
                policy=self.retry_policy,
                attempt_index=attempt,
                last_response=response,
            )
            self.sleep_fn(delay_s)

        # Defensive fallback; loop should either return or raise.
        raise RetryExhaustedError("All retry attempts failed", last_response=last_response)


@dataclass(slots=True)
class ResilientAsyncHttpClient:
    """Resilient asynchronous HTTP client.

    This client mirrors :class:`ResilientSyncHttpClient` but for async
    operations.

    Args:
        port: Asynchronous HTTP port implementation.
        retry_policy: Configuration for retries and backoff.
        circuit_breaker: Circuit breaker instance.
        default_timeout_s: Default per-request timeout in seconds.
        sleep_fn: Optional awaitable sleep function, primarily for testing.

    Example:
        client = ResilientAsyncHttpClient(port, retry_policy, circuit_breaker)
        response = await client.request("GET", "https://api.example.com/resource")
    """

    port: AsyncHttpPort
    retry_policy: RetryPolicy
    circuit_breaker: CircuitBreaker
    default_timeout_s: float = 5.0
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: QueryParams | None = None,
        json: object | None = None,
        data: BodyType | None = None,
        timeout_s: float | None = None,
        allow_retry_for_non_idempotent: bool | None = None,
    ) -> ResponseData:
        """Perform a resilient asynchronous HTTP request.

        See :meth:`ResilientSyncHttpClient.request` for parameter and behavior
        details.
        """

        is_idempotent = _is_idempotent_method(method)
        allow_retry_non_idempotent = (
            self.retry_policy.allow_retry_non_idempotent
            if allow_retry_for_non_idempotent is None
            else allow_retry_for_non_idempotent
        )

        max_attempts = self.retry_policy.max_attempts
        timeout = timeout_s or self.default_timeout_s
        last_response: ResponseData | None = None

        for attempt in range(max_attempts):
            try:
                self.circuit_breaker.before_call()
            except HttpClientError:
                raise

            try:
                response = await self.port.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    data=data,
                    timeout_s=timeout,
                )
            except HttpClientError as exc:
                logger.warning(
                    "HTTP client error during async request",
                    extra={"method": method, "url": url, "attempt": attempt + 1},
                )
                self.circuit_breaker.after_call(False)

                is_retryable_exc = isinstance(exc, self.retry_policy.retryable_exceptions)
                is_last_attempt = attempt >= max_attempts - 1

                if not is_retryable_exc or is_last_attempt:
                    raise RetryExhaustedError(
                        "All retry attempts failed", last_response=None
                    ) from exc

                delay_s = _calculate_delay_s(
                    policy=self.retry_policy,
                    attempt_index=attempt,
                    last_response=None,
                )
                await self.sleep_fn(delay_s)
                continue

            last_response = response

            if response.is_success:
                self.circuit_breaker.after_call(True)
                return response

            is_retryable_status = response.status_code in self.retry_policy.retryable_status_codes
            is_last_attempt = attempt >= max_attempts - 1

            if (
                not is_retryable_status
                or (not is_idempotent and not allow_retry_non_idempotent)
                or is_last_attempt
            ):
                self.circuit_breaker.after_call(False)
                raise RetryExhaustedError(
                    f"All retry attempts failed with status {response.status_code}",
                    last_response=response,
                )

            logger.info(
                "Retrying async HTTP request due to response status",
                extra={
                    "method": method,
                    "url": url,
                    "status_code": response.status_code,
                    "attempt": attempt + 1,
                },
            )
            self.circuit_breaker.after_call(False)

            delay_s = _calculate_delay_s(
                policy=self.retry_policy,
                attempt_index=attempt,
                last_response=response,
            )
            await self.sleep_fn(delay_s)

        raise RetryExhaustedError("All retry attempts failed", last_response=last_response)


__all__ = ["ResilientSyncHttpClient", "ResilientAsyncHttpClient"]
