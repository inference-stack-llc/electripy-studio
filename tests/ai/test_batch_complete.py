"""Tests for batch_complete."""

from __future__ import annotations

import threading
from typing import Any

from electripy.ai.batch_complete import batch_complete
from electripy.ai.llm_gateway.domain import LlmMessage, LlmRequest, LlmResponse, LlmRole


def _req(content: str = "hi") -> LlmRequest:
    return LlmRequest(
        model="test",
        messages=[LlmMessage(role=LlmRole.USER, content=content)],
    )


class _EchoPort:
    """Returns the user message as the response text."""

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        return LlmResponse(text=request.messages[0].content, model=request.model)


class _FailOnPort:
    """Fails for specific indices."""

    def __init__(self, fail_indices: set[int]) -> None:
        self._fail = fail_indices
        self._call_count = 0
        self._lock = threading.Lock()

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        with self._lock:
            self._call_count += 1
        # Use the message content as a proxy for the index
        msg_idx = int(request.messages[0].content)
        if msg_idx in self._fail:
            raise RuntimeError(f"fail-{msg_idx}")
        return LlmResponse(text=f"ok-{msg_idx}", model=request.model)


class TestBatchComplete:
    def test_basic_batch(self) -> None:
        reqs = [_req(f"msg-{i}") for i in range(5)]
        results = batch_complete(port=_EchoPort(), requests=reqs, max_concurrency=3)

        assert len(results) == 5
        for i, r in enumerate(results):
            assert isinstance(r, LlmResponse)
            assert r.text == f"msg-{i}"

    def test_empty_requests(self) -> None:
        results = batch_complete(port=_EchoPort(), requests=[])
        assert results == []

    def test_preserves_order(self) -> None:
        reqs = [_req(f"item-{i}") for i in range(10)]
        results = batch_complete(port=_EchoPort(), requests=reqs, max_concurrency=2)

        texts = [r.text for r in results if isinstance(r, LlmResponse)]
        assert texts == [f"item-{i}" for i in range(10)]

    def test_failed_requests_return_exceptions(self) -> None:
        reqs = [_req(str(i)) for i in range(5)]
        port = _FailOnPort(fail_indices={1, 3})
        results = batch_complete(port=port, requests=reqs, max_concurrency=5)

        assert isinstance(results[0], LlmResponse)
        assert isinstance(results[1], Exception)
        assert isinstance(results[2], LlmResponse)
        assert isinstance(results[3], Exception)
        assert isinstance(results[4], LlmResponse)

    def test_progress_callback(self) -> None:
        reqs = [_req(f"msg-{i}") for i in range(3)]
        progress: list[tuple[int, int]] = []

        batch_complete(
            port=_EchoPort(),
            requests=reqs,
            max_concurrency=1,
            on_progress=lambda done, total: progress.append((done, total)),
        )

        assert len(progress) == 3
        assert all(total == 3 for _, total in progress)
        assert sorted(done for done, _ in progress) == [1, 2, 3]

    def test_timeout_forwarded(self) -> None:
        received: list[Any] = []

        class _Capture:
            def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
                received.append(timeout)
                return LlmResponse(text="ok", model="test")

        batch_complete(
            port=_Capture(),
            requests=[_req()],
            max_concurrency=1,
            timeout=7.5,
        )
        assert received == [7.5]

    def test_concurrency_capped(self) -> None:
        """Verify max_concurrency limits parallel execution."""
        peak = 0
        current = 0
        lock = threading.Lock()

        class _SlowPort:
            def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
                nonlocal peak, current
                import time

                with lock:
                    current += 1
                    peak = max(peak, current)
                time.sleep(0.05)
                with lock:
                    current -= 1
                return LlmResponse(text="ok", model="test")

        batch_complete(
            port=_SlowPort(),
            requests=[_req() for _ in range(10)],
            max_concurrency=3,
        )
        assert peak <= 3
