"""Tests for the HTTP resilience client.

These tests use fake implementations of the HTTP ports to avoid real network
access while exercising retry and circuit breaker behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from electripy.concurrency.http_resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    HttpClientError,
    ResilientAsyncHttpClient,
    ResilientSyncHttpClient,
    ResponseData,
    RetryExhaustedError,
    RetryPolicy,
)
from electripy.concurrency.http_resilience.domain import Headers, QueryParams, TransientHttpError
from electripy.concurrency.http_resilience.ports import AsyncHttpPort, SyncHttpPort


@dataclass
class FakeResponsePlan:
    status_code: int
    raise_error: bool = False


class FakeSyncPort(SyncHttpPort):
    """Fake synchronous port for testing.

    It returns pre-configured responses or raises errors to simulate
    downstream behavior.
    """

    def __init__(self, plans: list[FakeResponsePlan]) -> None:
        self._plans = plans
        self.calls: int = 0

    def request(  # type: ignore[override]
        self,
        method: str,
        url: str,
        *,
        headers: Headers | None = None,
        params: QueryParams | None = None,
        json: object | None = None,
        data: bytes | str | None = None,
        timeout_s: float | None = None,
    ) -> ResponseData:
        plan = self._plans[min(self.calls, len(self._plans) - 1)]
        self.calls += 1

        if plan.raise_error:
            raise TransientHttpError("simulated transient error")

        return ResponseData(
            status_code=plan.status_code,
            headers={},
            body=b"{}",
            url=url,
            elapsed_s=0.01,
        )


class FakeAsyncPort(AsyncHttpPort):
    """Fake asynchronous port for testing."""

    def __init__(self, plans: list[FakeResponsePlan]) -> None:
        self._plans = plans
        self.calls: int = 0

    async def request(  # type: ignore[override]
        self,
        method: str,
        url: str,
        *,
        headers: Headers | None = None,
        params: QueryParams | None = None,
        json: object | None = None,
        data: bytes | str | None = None,
        timeout_s: float | None = None,
    ) -> ResponseData:
        plan = self._plans[min(self.calls, len(self._plans) - 1)]
        self.calls += 1

        if plan.raise_error:
            raise TransientHttpError("simulated transient error")

        return ResponseData(
            status_code=plan.status_code,
            headers={},
            body=b"{}",
            url=url,
            elapsed_s=0.01,
        )


def test_sync_client_success_no_retry() -> None:
    plans = [FakeResponsePlan(status_code=200)]
    port = FakeSyncPort(plans)
    policy = RetryPolicy(max_attempts=3, jitter_ratio=0.0)
    breaker = CircuitBreaker(CircuitBreakerConfig(3, 10.0))

    client = ResilientSyncHttpClient(port, policy, breaker, sleep_fn=lambda _: None)

    response = client.request("GET", "https://example.com")

    assert response.status_code == 200
    assert port.calls == 1


def test_sync_client_retries_on_transient_error() -> None:
    plans = [FakeResponsePlan(status_code=500, raise_error=True), FakeResponsePlan(status_code=200)]
    port = FakeSyncPort(plans)
    policy = RetryPolicy(max_attempts=2, jitter_ratio=0.0)
    breaker = CircuitBreaker(CircuitBreakerConfig(3, 10.0))

    client = ResilientSyncHttpClient(port, policy, breaker, sleep_fn=lambda _: None)

    response = client.request("GET", "https://example.com")

    assert response.status_code == 200
    assert port.calls == 2


def test_sync_client_does_not_retry_non_idempotent_by_default() -> None:
    plans = [FakeResponsePlan(status_code=500), FakeResponsePlan(status_code=200)]
    port = FakeSyncPort(plans)
    policy = RetryPolicy(max_attempts=2, jitter_ratio=0.0)
    breaker = CircuitBreaker(CircuitBreakerConfig(3, 10.0))

    client = ResilientSyncHttpClient(port, policy, breaker, sleep_fn=lambda _: None)

    with pytest.raises(RetryExhaustedError):
        client.request("POST", "https://example.com")

    assert port.calls == 1


def test_sync_client_can_retry_non_idempotent_with_opt_in() -> None:
    plans = [FakeResponsePlan(status_code=500), FakeResponsePlan(status_code=200)]
    port = FakeSyncPort(plans)
    policy = RetryPolicy(max_attempts=2, jitter_ratio=0.0)
    breaker = CircuitBreaker(CircuitBreakerConfig(3, 10.0))

    client = ResilientSyncHttpClient(port, policy, breaker, sleep_fn=lambda _: None)

    response = client.request(
        "POST",
        "https://example.com",
        headers={"Idempotency-Key": "test-key"},
        allow_retry_for_non_idempotent=True,
    )

    assert response.status_code == 200
    assert port.calls == 2


def test_circuit_breaker_opens_after_failures() -> None:
    plans = [FakeResponsePlan(status_code=500), FakeResponsePlan(status_code=500)]
    port = FakeSyncPort(plans)
    policy = RetryPolicy(max_attempts=1, jitter_ratio=0.0)

    # Configure breaker to open after 1 failure and not recover during test
    breaker = CircuitBreaker(CircuitBreakerConfig(1, recovery_timeout_s=1000.0))

    client = ResilientSyncHttpClient(port, policy, breaker, sleep_fn=lambda _: None)

    with pytest.raises(RetryExhaustedError):
        client.request("GET", "https://example.com")

    # Second call should fail fast due to open circuit
    with pytest.raises(HttpClientError):
        client.request("GET", "https://example.com")


@pytest.mark.asyncio
async def test_async_client_success_no_retry() -> None:
    plans = [FakeResponsePlan(status_code=200)]
    port = FakeAsyncPort(plans)
    policy = RetryPolicy(max_attempts=3, jitter_ratio=0.0)
    breaker = CircuitBreaker(CircuitBreakerConfig(3, 10.0))

    client = ResilientAsyncHttpClient(port, policy, breaker)

    response = await client.request("GET", "https://example.com")

    assert response.status_code == 200
    assert port.calls == 1


@pytest.mark.asyncio
async def test_async_client_retries_on_transient_error() -> None:
    plans = [FakeResponsePlan(status_code=500, raise_error=True), FakeResponsePlan(status_code=200)]
    port = FakeAsyncPort(plans)
    policy = RetryPolicy(max_attempts=2, jitter_ratio=0.0)
    breaker = CircuitBreaker(CircuitBreakerConfig(3, 10.0))

    client = ResilientAsyncHttpClient(port, policy, breaker)

    response = await client.request("GET", "https://example.com")

    assert response.status_code == 200
    assert port.calls == 2
