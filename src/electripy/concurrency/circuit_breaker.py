"""Circuit Breaker — protect against cascading failures from flaky providers.

Purpose:
  - Track consecutive failures and trip open when a threshold is reached.
  - While open, calls fail fast without hitting the downstream service.
  - After a cooldown, allow a single probe call (half-open state).

Guarantees:
  - Thread-safe — safe for concurrent use.
  - Decorator-friendly — wrap any callable or method.
  - No dependencies beyond the standard library.

Usage::

    from electripy.concurrency.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

    @cb
    def call_llm(prompt: str) -> str:
        return expensive_api_call(prompt)

    # Or without decorator:
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    try:
        result = cb.call(lambda: provider.complete(request))
    except CircuitOpenError:
        result = cached_fallback(request)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
]

T = TypeVar("T")


class CircuitState(StrEnum):
    """Observable state of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    failures: int
    recovery_timeout: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Circuit open after {self.failures} consecutive failures. "
            f"Retry after {self.recovery_timeout}s."
        )


class CircuitBreaker:
    """Thread-safe circuit breaker with closed → open → half-open states.

    Args:
        failure_threshold: Consecutive failures before tripping open.
        recovery_timeout: Seconds to wait before allowing a probe call.
        success_threshold: Consecutive successes in half-open state
            before closing the circuit (default 1).
    """

    __slots__ = (
        "_failure_threshold",
        "_recovery_timeout",
        "_success_threshold",
        "_failure_count",
        "_success_count",
        "_state",
        "_opened_at",
        "_lock",
    )

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._failure_count = 0
        self._success_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    # -- Public API ---------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition on read)."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def call(self, fn: Callable[[], T]) -> T:
        """Execute *fn* through the circuit breaker.

        Raises:
            CircuitOpenError: If the circuit is open and the recovery
                timeout has not elapsed.
        """
        self._before_call()
        try:
            result = fn()
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def reset(self) -> None:
        """Force the circuit back to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0

    # -- Decorator support --------------------------------------------------

    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Use as a decorator: ``@circuit_breaker``."""

        def wrapper(*args: object, **kwargs: object) -> T:
            return self.call(lambda: fn(*args, **kwargs))

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    # -- Internal state machine ---------------------------------------------

    def _before_call(self) -> None:
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state is CircuitState.OPEN:
                raise CircuitOpenError(
                    failures=self._failure_count,
                    recovery_timeout=self._recovery_timeout,
                )

    def _on_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            else:
                self._failure_count = 0

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state is CircuitState.HALF_OPEN:
                # Half-open probe failed — re-open.
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._success_count = 0
            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    def _maybe_transition_to_half_open(self) -> None:
        """Transition from OPEN → HALF_OPEN if the timeout has elapsed.

        Must be called while holding ``_lock``.
        """
        if (
            self._state is CircuitState.OPEN
            and (time.monotonic() - self._opened_at) >= self._recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
