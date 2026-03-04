# HTTP Resilience Client

A production-minded HTTP client that combines retries, exponential backoff
with jitter, and a circuit breaker to safely call downstream services.

## What it is

- A small, focused component for **outbound** HTTP calls.
- Built around **Ports & Adapters** so the underlying HTTP library is
  swappable (default: httpx).
- Provides both **synchronous** and **asynchronous** resilient clients.
- Normalizes responses into a repository-owned `ResponseData` model.
- Raises repository-owned exceptions instead of leaking third-party errors.

## When to use

Use this component when:

- Calling third-party APIs or internal services over HTTP.
- You need predictable latency bounds and want to avoid unbounded retries.
- You want to fail fast during sustained outages via a circuit breaker.
- You want a clean seam where the HTTP library can be swapped.

## When not to use

This component is not a full HTTP framework. Do not use it for:

- Building HTTP servers.
- Streaming/download-heavy use cases that require advanced httpx features.
- Very latency-sensitive code paths where even modest backoff is too costly.

## Design decisions

- **Ports & Adapters:**
  - `SyncHttpPort` and `AsyncHttpPort` define the minimal surface the
    orchestration layer depends on.
  - `HttpxSyncAdapter` and `HttpxAsyncAdapter` implement those ports using
    httpx.
- **Domain models:**
  - `ResponseData` is a small, stable response model owned by this
    repository.
  - `RetryPolicy` and `CircuitBreakerConfig` configure behavior explicitly.
- **Domain exceptions:**
  - `HttpClientError` is the base type for all failures.
  - `CircuitOpenError` is raised when the breaker denies calls.
  - `RetryExhaustedError` is raised when all attempts fail.

## Failure modes & guarantees

- **Retries:**
  - Configured via `RetryPolicy`.
  - Only retryable status codes (e.g., 500, 503) and exceptions trigger
    retries.
  - Retries are bounded by `max_attempts`.
- **Backoff:**
  - Exponential backoff starting from `base_delay_s`.
  - Jitter is applied via `jitter_ratio` to reduce thundering herd effects.
  - Optional honoring of `Retry-After` (seconds) when present.
- **Circuit breaker:**
  - `CircuitBreaker` has three states: CLOSED, OPEN, HALF_OPEN.
  - After `failure_threshold` consecutive failures, the circuit moves to OPEN.
  - In OPEN, calls fail fast with `CircuitOpenError`.
  - After `recovery_timeout_s`, the circuit moves to HALF_OPEN to probe
    recovery with up to `half_open_max_calls`.
  - If `half_open_success_threshold` calls succeed, the circuit closes.

## How retries, backoff, and jitter work

1. The client performs an initial attempt.
2. If a retryable error occurs (status code or exception), it calculates a
   delay using exponential backoff and optional jitter.
3. The delay is bounded by `max_delay_s` and may be overridden by
   `Retry-After` (seconds) when enabled.
4. This repeats up to `max_attempts`.

## Idempotency & safety

- By default, only **idempotent** methods (`GET`, `HEAD`, `OPTIONS`) are
  retried.
- Non-idempotent methods (e.g., `POST`, `PATCH`, `DELETE`) are **not**
  retried unless explicitly opted in per-call or via policy.
- For `POST` requests that are safe to retry, use an
  `Idempotency-Key` header and explicitly enable retries.

## Basic usage example (sync)

```python
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

response = client.request("GET", "/health")
print(response.status_code)
```

## Advanced usage example (async + POST with idempotency)

```python
import asyncio

from electripy.concurrency.http_resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    HttpxAsyncAdapter,
    ResilientAsyncHttpClient,
    RetryPolicy,
)


async def main() -> None:
    adapter = HttpxAsyncAdapter(base_url="https://api.example.com")
    policy = RetryPolicy(max_attempts=5)
    breaker = CircuitBreaker(CircuitBreakerConfig(3, 15.0))
    client = ResilientAsyncHttpClient(adapter, policy, breaker)

    headers = {"Idempotency-Key": "your-generated-key"}
    response = await client.request(
        "POST",
        "/orders",
        headers=headers,
        json={"amount": 100},
        allow_retry_for_non_idempotent=True,
    )
    print(response.status_code)


asyncio.run(main())
```

## Swap guide: using a different HTTP library

To replace httpx with another library (e.g., `requests`):

1. Implement `SyncHttpPort` or `AsyncHttpPort` using the new library.
2. Map the library's response object into `ResponseData`.
3. Map transport-level errors into domain exceptions such as
   `TransientHttpError`.

Example (sync skeleton):

```python
from dataclasses import dataclass

import requests

from electripy.concurrency.http_resilience.domain import ResponseData, TransientHttpError
from electripy.concurrency.http_resilience.ports import SyncHttpPort


@dataclass
class RequestsSyncAdapter(SyncHttpPort):
    base_url: str | None = None

    def request(self, method: str, url: str, **kwargs: object) -> ResponseData:  # type: ignore[override]
        full_url = (self.base_url or "") + url
        try:
            resp = requests.request(method=method, url=full_url, **kwargs)
        except requests.Timeout as exc:
            raise TransientHttpError("Timeout") from exc
        # Map other exceptions as needed

        return ResponseData(
            status_code=resp.status_code,
            headers={k.lower(): v for k, v in resp.headers.items()},
            body=resp.content,
            url=resp.url,
            elapsed_s=resp.elapsed.total_seconds(),
        )
```

Once the adapter implements `SyncHttpPort`/`AsyncHttpPort`, you can plug it
into `ResilientSyncHttpClient` or `ResilientAsyncHttpClient` without any
changes to the service layer.
