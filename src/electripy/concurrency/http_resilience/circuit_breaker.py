"""Circuit breaker implementation for the HTTP resilience client.

The circuit breaker protects downstream services and callers from sustained
failures by failing fast once a configurable failure threshold is reached.

States:
- CLOSED: All calls flow through. Consecutive failures are counted.
- OPEN: Calls are rejected immediately until a recovery timeout elapses.
- HALF_OPEN: A limited number of trial calls are allowed to probe recovery.

Example:
    from electripy.concurrency.http_resilience.circuit_breaker import (
        CircuitBreaker,
    )
    from electripy.concurrency.http_resilience.domain import CircuitBreakerConfig

    breaker = CircuitBreaker(CircuitBreakerConfig(5, 30.0))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic

from electripy.core.logging import get_logger

from .domain import CircuitBreakerConfig, CircuitOpenError

logger = get_logger(__name__)


class CircuitState(StrEnum):
    """Circuit breaker states.

    Example:
        state = CircuitState.CLOSED
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(slots=True)
class CircuitBreaker:
    """Stateful circuit breaker.

    This class is intentionally generic and independent of any HTTP client. It
    uses a monotonic clock to track time spent in the OPEN state and to
    transition into HALF_OPEN when the recovery timeout elapses.

    Args:
        config: Circuit breaker configuration.
        time_fn: Optional time provider, primarily for testing. Defaults to
            :func:`time.monotonic`.

    Example:
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=30.0)
        breaker = CircuitBreaker(config)
    """

    config: CircuitBreakerConfig
    time_fn: Callable[[], float] = monotonic
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _last_state_change: float = field(default_factory=monotonic, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _half_open_successes: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state.

        Returns:
            CircuitState: The current state.

        Example:
            assert breaker.state is CircuitState.CLOSED
        """

        self._update_state_if_needed()
        return self._state

    def before_call(self) -> None:
        """Check whether a call is allowed and update internal state.

        This method should be invoked immediately before a protected call.

        Raises:
            CircuitOpenError: If the circuit is OPEN and calls are not yet
                allowed.
        """

        self._update_state_if_needed()
        now = self.time_fn()

        if self._state is CircuitState.OPEN:
            logger.debug("Circuit breaker OPEN: rejecting call")
            raise CircuitOpenError("Circuit is open; calls are temporarily blocked")

        if self._state is CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                logger.debug("Circuit breaker HALF_OPEN: max trial calls reached")
                raise CircuitOpenError(
                    "Circuit is half-open; max trial calls reached; calls are blocked"
                )
            self._half_open_calls += 1
            logger.debug("Circuit breaker HALF_OPEN: allowing trial call", extra={"time": now})

    def after_call(self, success: bool) -> None:
        """Record the outcome of a call and transition state as needed.

        Args:
            success: True if the call succeeded (e.g., HTTP 2xx), False
                otherwise.
        """

        self._update_state_if_needed()
        now = self.time_fn()

        if success:
            if self._state is CircuitState.CLOSED:
                self._consecutive_failures = 0
            elif self._state is CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.half_open_success_threshold:
                    logger.info("Circuit breaker: transitioning to CLOSED after successes")
                    self._transition_to(CircuitState.CLOSED, now)
        else:
            if self._state is CircuitState.CLOSED:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.failure_threshold:
                    logger.warning("Circuit breaker: transitioning to OPEN due to failures")
                    self._transition_to(CircuitState.OPEN, now)
            elif self._state is CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker: failure in HALF_OPEN, transitioning to OPEN")
                self._transition_to(CircuitState.OPEN, now)

    def _update_state_if_needed(self) -> None:
        """Update the state based on elapsed time while OPEN.

        This method transitions the circuit from OPEN to HALF_OPEN when the
        configured recovery timeout has elapsed.
        """

        now = self.time_fn()

        if self._state is CircuitState.OPEN:
            if now - self._last_state_change >= self.config.recovery_timeout_s:
                logger.info("Circuit breaker: transitioning from OPEN to HALF_OPEN")
                self._transition_to(CircuitState.HALF_OPEN, now)

    def _transition_to(self, new_state: CircuitState, now: float) -> None:
        """Transition to a new state and reset relevant counters.

        Args:
            new_state: Target circuit state.
            now: Current monotonic time.
        """

        self._state = new_state
        self._last_state_change = now
        self._consecutive_failures = 0
        self._half_open_calls = 0
        self._half_open_successes = 0


__all__ = ["CircuitBreaker", "CircuitState"]
