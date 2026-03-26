"""Tests for FallbackChainPort."""

from __future__ import annotations

import pytest

from electripy.ai.fallback_chain import FallbackChainPort
from electripy.ai.llm_gateway.domain import LlmMessage, LlmRequest, LlmResponse, LlmRole


def _req(content: str = "hi") -> LlmRequest:
    return LlmRequest(
        model="test",
        messages=[LlmMessage(role=LlmRole.USER, content=content)],
    )


class _FakePort:
    def __init__(self, text: str) -> None:
        self._text = text
        self.called = False

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        self.called = True
        return LlmResponse(text=self._text, model=request.model)


class _FailingPort:
    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("provider down")
        self.called = False

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        self.called = True
        raise self._exc


class TestFallbackChainPort:
    def test_first_provider_succeeds(self) -> None:
        p1 = _FakePort("from-p1")
        p2 = _FakePort("from-p2")
        chain = FallbackChainPort(providers=[p1, p2])

        resp = chain.complete(_req())

        assert resp.text == "from-p1"
        assert resp.metadata["_fallback_provider_index"] == 0
        assert p1.called
        assert not p2.called

    def test_falls_through_on_failure(self) -> None:
        p1 = _FailingPort()
        p2 = _FakePort("from-p2")
        chain = FallbackChainPort(providers=[p1, p2])

        resp = chain.complete(_req())

        assert resp.text == "from-p2"
        assert resp.metadata["_fallback_provider_index"] == 1
        assert p1.called
        assert p2.called

    def test_all_fail_raises_last_exception(self) -> None:
        p1 = _FailingPort(ValueError("first"))
        p2 = _FailingPort(RuntimeError("second"))
        chain = FallbackChainPort(providers=[p1, p2])

        with pytest.raises(RuntimeError, match="second"):
            chain.complete(_req())

    def test_three_providers_middle_succeeds(self) -> None:
        p1 = _FailingPort()
        p2 = _FakePort("from-p2")
        p3 = _FakePort("from-p3")
        chain = FallbackChainPort(providers=[p1, p2, p3])

        resp = chain.complete(_req())

        assert resp.text == "from-p2"
        assert resp.metadata["_fallback_provider_index"] == 1
        assert not p3.called

    def test_empty_providers_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            FallbackChainPort(providers=[])

    def test_timeout_forwarded(self) -> None:
        received_timeout: list[float | None] = []

        class _TimeoutCapture:
            def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
                received_timeout.append(timeout)
                return LlmResponse(text="ok", model="test")

        chain = FallbackChainPort(providers=[_TimeoutCapture()])
        chain.complete(_req(), timeout=5.0)
        assert received_timeout == [5.0]
