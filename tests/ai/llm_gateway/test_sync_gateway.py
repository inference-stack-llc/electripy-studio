"""Tests for the synchronous LLM gateway client."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from electripy.ai.llm_gateway import (
    LlmGatewaySettings,
    LlmGatewaySyncClient,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    RetryPolicy,
)
from electripy.ai.llm_gateway.errors import (
    PromptRejectedError,
    RateLimitedError,
    RetryExhaustedError,
    TokenBudgetExceededError,
    TransientLlmError,
)
from electripy.ai.llm_gateway.ports import GuardResult, PromptGuardPort, RedactorPort, SyncLlmPort


@dataclass
class FakeSyncPort(SyncLlmPort):
    """Simple fake port returning a constant response."""

    text: str = "ok"
    calls: int = 0

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        self.calls += 1
        return LlmResponse(text=self.text, model=request.model)


def test_sync_plain_text_success() -> None:
    port = FakeSyncPort(text="hello")
    client = LlmGatewaySyncClient(port=port)
    request = LlmRequest(model="gpt-test", messages=[LlmMessage.user("Hi")])

    response = client.complete(request)

    assert response.text == "hello"
    assert port.calls == 1


class FlakyRateLimitedPort(SyncLlmPort):
    """Port that fails with RateLimitedError a configurable number of times."""

    def __init__(self, fail_times: int, final_text: str) -> None:
        self._fail_times = fail_times
        self._final_text = final_text
        self.calls = 0

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RateLimitedError("rate limited", status_code=429, retry_after_seconds=0.0)
        return LlmResponse(text=self._final_text, model=request.model)


def test_retry_on_rate_limit_success() -> None:
    port = FlakyRateLimitedPort(fail_times=1, final_text="ok")
    settings = LlmGatewaySettings(
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            total_timeout_seconds=5.0,
        ),
    )
    client = LlmGatewaySyncClient(port=port, settings=settings)
    request = LlmRequest(model="gpt-test", messages=[LlmMessage.user("Hi")])

    response = client.complete(request)

    assert response.text == "ok"
    assert port.calls == 2


def test_retry_exhausted_error() -> None:
    class AlwaysFailPort(SyncLlmPort):
        def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
            raise TransientLlmError(message="temporary", status_code=503)

    port = AlwaysFailPort()
    settings = LlmGatewaySettings(
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            total_timeout_seconds=1.0,
        ),
    )
    client = LlmGatewaySyncClient(port=port, settings=settings)
    request = LlmRequest(model="gpt-test", messages=[LlmMessage.user("Hi")])

    with pytest.raises(RetryExhaustedError):
        client.complete(request)


def test_token_budget_exceeded_raises() -> None:
    port = FakeSyncPort()
    settings = LlmGatewaySettings(default_max_input_chars=5)
    client = LlmGatewaySyncClient(port=port, settings=settings)
    request = LlmRequest(model="gpt-test", messages=[LlmMessage.user("too long message")])

    with pytest.raises(TokenBudgetExceededError):
        client.complete(request)


class DenyAllGuard(PromptGuardPort):
    def assess(self, messages: Sequence[LlmMessage]) -> GuardResult:
        return GuardResult(allowed=False, score=0.0, reasons=("blocked",))


def test_prompt_guard_rejects() -> None:
    port = FakeSyncPort()
    settings = LlmGatewaySettings(prompt_guard=DenyAllGuard())
    client = LlmGatewaySyncClient(port=port, settings=settings)
    request = LlmRequest(model="gpt-test", messages=[LlmMessage.user("Hi")])

    with pytest.raises(PromptRejectedError):
        client.complete(request)


class RecordingRedactor(RedactorPort):
    def __init__(self) -> None:
        self.last_text: str | None = None

    def redact(self, text: str) -> str:
        # Record only the first text passed for assertions.
        if self.last_text is None:
            self.last_text = text
        return "[redacted]"


def test_safe_logging_uses_redactor() -> None:
    port = FakeSyncPort(text="secret response")
    redactor = RecordingRedactor()
    settings = LlmGatewaySettings(enable_safe_logging=True, redactor=redactor)
    client = LlmGatewaySyncClient(port=port, settings=settings)
    request = LlmRequest(
        model="gpt-test",
        messages=[LlmMessage.user("secret prompt with email user@example.com")],
    )

    response = client.complete(request)

    assert response.text == "secret response"
    assert redactor.last_text is not None
    assert "user@example.com" in redactor.last_text
