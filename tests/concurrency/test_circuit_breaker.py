"""Tests for CircuitBreaker."""

from __future__ import annotations

import time

import pytest

from electripy.concurrency.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        assert cb.state is CircuitState.CLOSED

    def test_success_keeps_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state is CircuitState.OPEN
        assert cb.failure_count == 3

    def test_open_rejects_calls(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: "should not run")

        assert exc_info.value.failures == 2

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state is CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state is CircuitState.HALF_OPEN

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        time.sleep(0.15)

        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        time.sleep(0.15)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("still broken")))

        assert cb.state is CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

        # 2 failures, then a success, then 2 more failures — should NOT trip.
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        cb.call(lambda: "ok")
        assert cb.failure_count == 0

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state is CircuitState.CLOSED  # Still below threshold

    def test_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state is CircuitState.OPEN
        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_decorator_usage(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

        call_count = 0

        @cb
        def do_work() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        assert do_work() == "done"
        assert call_count == 1

    def test_decorator_preserves_name(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

        @cb
        def my_function() -> str:
            """My docstring."""
            return "ok"

        assert my_function.__name__ == "my_function"

    def test_success_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, success_threshold=2)

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        time.sleep(0.15)

        # First success in half-open → still half-open.
        cb.call(lambda: "ok1")
        assert cb.state is CircuitState.HALF_OPEN

        # Second success → closes.
        cb.call(lambda: "ok2")
        assert cb.state is CircuitState.CLOSED

    def test_circuit_open_error_fields(self) -> None:
        err = CircuitOpenError(failures=5, recovery_timeout=30.0)
        assert err.failures == 5
        assert err.recovery_timeout == 30.0
