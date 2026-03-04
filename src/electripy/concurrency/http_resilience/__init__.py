"""Public API for the HTTP resilience client component.

This package exposes the main domain models, configuration objects, circuit
breaker, adapters, and resilient clients.

Example:
    from electripy.concurrency.http_resilience import (
        CircuitBreaker,
        CircuitBreakerConfig,
        HttpxAsyncAdapter,
        HttpxSyncAdapter,
        ResilientAsyncHttpClient,
        ResilientSyncHttpClient,
        ResponseData,
        RetryPolicy,
    )

    adapter = HttpxSyncAdapter(base_url="https://api.example.com")
    policy = RetryPolicy(max_attempts=3)
    breaker = CircuitBreaker(CircuitBreakerConfig(5, 30.0))
    client = ResilientSyncHttpClient(adapter, policy, breaker)
"""

from __future__ import annotations

from .adapters import HttpxAsyncAdapter, HttpxSyncAdapter
from .circuit_breaker import CircuitBreaker
from .domain import (
    BodyType,
    CircuitBreakerConfig,
    CircuitOpenError,
    HttpClientError,
    HttpMethod,
    QueryParams,
    ResponseData,
    RetryExhaustedError,
    RetryPolicy,
)
from .services import ResilientAsyncHttpClient, ResilientSyncHttpClient

__all__ = [
    "BodyType",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitOpenError",
    "HttpClientError",
    "HttpMethod",
    "QueryParams",
    "ResponseData",
    "RetryExhaustedError",
    "RetryPolicy",
    "HttpxAsyncAdapter",
    "HttpxSyncAdapter",
    "ResilientAsyncHttpClient",
    "ResilientSyncHttpClient",
]
