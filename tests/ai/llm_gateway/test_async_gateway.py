from __future__ import annotations

"""Tests for the asynchronous LLM gateway client."""

import pytest

from electripy.ai.llm_gateway import (
    LlmGatewayAsyncClient,
    LlmGatewaySettings,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    RetryPolicy,
)
from electripy.ai.llm_gateway.errors import RateLimitedError, RetryExhaustedError, TransientLlmError
from electripy.ai.llm_gateway.ports import AsyncLlmPort


class FakeAsyncPort(AsyncLlmPort):
    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls = 0

    async def complete(
        self,
        request: LlmRequest,
        *,
        timeout: float | None = None,
    ) -> LlmResponse:
        self.calls += 1
        return LlmResponse(text=self.text, model=request.model)


@pytest.mark.asyncio
async def test_async_plain_text_success() -> None:
    port = FakeAsyncPort(text="hello")
    client = LlmGatewayAsyncClient(port=port)
    request = LlmRequest(model="gpt-test", messages=[LlmMessage.user("Hi")])

    response = await client.complete(request)

    assert response.text == "hello"
    assert port.calls == 1


class FlakyAsyncRateLimitedPort(AsyncLlmPort):
    def __init__(self, fail_times: int, final_text: str) -> None:
        self._fail_times = fail_times
        self._final_text = final_text
        self.calls = 0

    async def complete(
        self,
        request: LlmRequest,
        *,
        timeout: float | None = None,
    ) -> LlmResponse:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RateLimitedError("rate limited", status_code=429, retry_after_seconds=0.0)
        return LlmResponse(text=self._final_text, model=request.model)


@pytest.mark.asyncio
async def test_async_retry_on_rate_limit_success() -> None:
    port = FlakyAsyncRateLimitedPort(fail_times=1, final_text="ok")
    settings = LlmGatewaySettings(
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            total_timeout_seconds=5.0,
        ),
    )
    client = LlmGatewayAsyncClient(port=port, settings=settings)
    request = LlmRequest(model="gpt-test", messages=[LlmMessage.user("Hi")])

    response = await client.complete(request)

    assert response.text == "ok"
    assert port.calls == 2


class AlwaysFailAsyncPort(AsyncLlmPort):
    async def complete(
        self,
        request: LlmRequest,
        *,
        timeout: float | None = None,
    ) -> LlmResponse:
        raise TransientLlmError(message="temporary", status_code=503)


@pytest.mark.asyncio
async def test_async_retry_exhausted_error() -> None:
    port = AlwaysFailAsyncPort()
    settings = LlmGatewaySettings(
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            total_timeout_seconds=1.0,
        ),
    )
    client = LlmGatewayAsyncClient(port=port, settings=settings)
    request = LlmRequest(model="gpt-test", messages=[LlmMessage.user("Hi")])

    with pytest.raises(RetryExhaustedError):
        await client.complete(request)
