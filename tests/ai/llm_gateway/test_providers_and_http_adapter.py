"""Tests for provider factory helpers and HTTP JSON adapters.

These tests use simple fake HTTP clients and responses to avoid real
network calls while exercising the adapter logic and provider factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from electripy.ai.llm_gateway import (
    LlmGatewaySettings,
    LlmGatewaySyncClient,
    LlmMessage,
    LlmRequest,
    build_llm_sync_client,
)
from electripy.ai.llm_gateway.adapters import HttpJsonChatAsyncAdapter, HttpJsonChatSyncAdapter
from electripy.ai.llm_gateway.errors import RateLimitedError


@dataclass
class FakeResponse:
    status_code: int
    body: dict[str, Any]
    headers: dict[str, str] | None = None

    def json(self) -> dict[str, Any]:  # pragma: no cover - trivial
        return self.body


class FakeHttpClient:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float | None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self._response


@pytest.mark.parametrize("status", [200, 201])
def test_http_json_sync_adapter_success(status: int) -> None:
    response_data = {
        "id": "req_123",
        "model": "test-model",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 10},
    }
    fake_response = FakeResponse(status_code=status, body=response_data, headers={})
    client = FakeHttpClient(fake_response)

    adapter = HttpJsonChatSyncAdapter(
        base_url="https://example.com",
        path="/v1/chat/completions",
        api_key="test",
        client=client,  # type: ignore[arg-type]
    )

    request = LlmRequest(
        model="test-model",
        messages=[LlmMessage.user("Hi")],
    )
    response = adapter.complete(request)

    assert response.text == "hello"
    assert response.model == "test-model"
    assert response.usage_total_tokens == 10
    assert client.calls, "Expected HTTP client to be called"


def test_http_json_sync_adapter_rate_limited() -> None:
    fake_response = FakeResponse(status_code=429, body={"error": "rate limited"}, headers={"Retry-After": "1"})
    client = FakeHttpClient(fake_response)

    adapter = HttpJsonChatSyncAdapter(
        base_url="https://example.com",
        path="/v1/chat/completions",
        api_key="test",
        client=client,  # type: ignore[arg-type]
    )

    request = LlmRequest(
        model="test-model",
        messages=[LlmMessage.user("Hi")],
    )

    with pytest.raises(RateLimitedError):
        adapter.complete(request)


@pytest.mark.asyncio
async def test_http_json_async_adapter_success() -> None:
    response_data = {
        "id": "req_456",
        "model": "test-model",
        "choices": [{"message": {"content": "hello-async"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 20},
    }

    class FakeAsyncResponse:
        def __init__(self, data: dict[str, Any]) -> None:
            self.status_code = 200
            self._data = data
            self.headers: dict[str, str] = {}

        def json(self) -> dict[str, Any]:  # pragma: no cover - trivial
            return self._data

    class FakeAsyncClient:
        def __init__(self, response: FakeAsyncResponse) -> None:
            self._response = response
            self.calls: list[dict[str, Any]] = []

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float | None):
            self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return self._response

    fake_response = FakeAsyncResponse(response_data)
    client = FakeAsyncClient(fake_response)

    adapter = HttpJsonChatAsyncAdapter(
        base_url="https://example.com",
        api_key="test",
        client=client,  # type: ignore[arg-type]
    )

    request = LlmRequest(
        model="test-model",
        messages=[LlmMessage.user("Hi")],
    )

    response = await adapter.complete(request)

    assert response.text == "hello-async"
    assert response.model == "test-model"
    assert response.usage_total_tokens == 20
    assert client.calls, "Expected async HTTP client to be called"


def test_build_llm_sync_client_http_json() -> None:
    response_data = {
        "id": "req_789",
        "model": "test-model",
        "choices": [{"message": {"content": "via-factory"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 30},
    }
    fake_response = FakeResponse(status_code=200, body=response_data, headers={})
    client = FakeHttpClient(fake_response)

    settings = LlmGatewaySettings()
    gateway = build_llm_sync_client(
        "http-json",
        settings=settings,
        base_url="https://example.com",
        path="/v1/chat/completions",
        api_key="test",
        client=client,  # type: ignore[arg-type]
    )

    request = LlmRequest(
        model="test-model",
        messages=[LlmMessage.user("Hi")],
    )

    response = gateway.complete(request)

    assert isinstance(gateway, LlmGatewaySyncClient)
    assert response.text == "via-factory"


def test_build_llm_sync_client_unknown_provider() -> None:
    with pytest.raises(ValueError):
        build_llm_sync_client("unknown-provider")
