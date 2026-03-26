"""Tests for OpenAI, Anthropic, and Ollama provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from electripy.ai.llm_gateway.domain import LlmMessage, LlmRequest, LlmRole
from electripy.ai.llm_gateway.errors import RateLimitedError, TransientLlmError
from electripy.ai.llm_gateway.provider_adapters import (
    AnthropicSyncAdapter,
    OllamaSyncAdapter,
    OpenAiSyncAdapter,
)

# ---------------------------------------------------------------------------
# Fake OpenAI SDK objects
# ---------------------------------------------------------------------------


@dataclass
class _FakeOpenAiUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class _FakeOpenAiMessage:
    content: str = "Hello from GPT"
    role: str = "assistant"


@dataclass
class _FakeOpenAiChoice:
    message: _FakeOpenAiMessage = None  # type: ignore[assignment]
    finish_reason: str = "stop"
    index: int = 0

    def __post_init__(self) -> None:
        if self.message is None:
            self.message = _FakeOpenAiMessage()


@dataclass
class _FakeOpenAiResponse:
    id: str = "chatcmpl-abc123"
    choices: list = None  # type: ignore[assignment]
    model: str = "gpt-4o-mini"
    usage: _FakeOpenAiUsage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.choices is None:
            self.choices = [_FakeOpenAiChoice()]
        if self.usage is None:
            self.usage = _FakeOpenAiUsage()


class _FakeOpenAiCompletions:
    def __init__(self, response: _FakeOpenAiResponse | None = None) -> None:
        self._response = response or _FakeOpenAiResponse()
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _FakeOpenAiResponse:
        self.last_kwargs = kwargs
        return self._response


class _FakeOpenAiChat:
    def __init__(self, response: _FakeOpenAiResponse | None = None) -> None:
        self.completions = _FakeOpenAiCompletions(response)


class _FakeOpenAiClient:
    def __init__(self, response: _FakeOpenAiResponse | None = None) -> None:
        self.chat = _FakeOpenAiChat(response)


class _FakeOpenAiRateLimitError(Exception):
    status_code = 429


class _FakeOpenAiServerError(Exception):
    status_code = 500


class _FakeOpenAiRateLimitCompletions:
    def create(self, **kwargs: Any) -> None:
        raise _FakeOpenAiRateLimitError()


class _FakeOpenAiServerCompletions:
    def create(self, **kwargs: Any) -> None:
        raise _FakeOpenAiServerError()


# ---------------------------------------------------------------------------
# OpenAI adapter tests
# ---------------------------------------------------------------------------


class TestOpenAiSyncAdapterFromProviderAdapters:
    """Verify OpenAiSyncAdapter is accessible from provider_adapters."""

    def test_complete_basic(self) -> None:
        client = _FakeOpenAiClient()
        adapter = OpenAiSyncAdapter(client=client)

        resp = adapter.complete(_make_request("hi", model="gpt-4o-mini"))

        assert resp.text == "Hello from GPT"
        assert resp.model == "gpt-4o-mini"
        assert resp.usage_total_tokens == 30
        assert resp.finish_reason == "stop"
        assert resp.request_id == "chatcmpl-abc123"

    def test_messages_forwarded(self) -> None:
        client = _FakeOpenAiClient()
        adapter = OpenAiSyncAdapter(client=client)

        adapter.complete(_make_request("what is 2+2?", model="gpt-4o-mini"))

        kwargs = client.chat.completions.last_kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "what is 2+2?"}]
        assert kwargs["model"] == "gpt-4o-mini"

    def test_system_message_included(self) -> None:
        client = _FakeOpenAiClient()
        adapter = OpenAiSyncAdapter(client=client)

        adapter.complete(_make_request("hi", model="gpt-4o-mini", system="Be concise"))

        kwargs = client.chat.completions.last_kwargs
        roles = [m["role"] for m in kwargs["messages"]]
        assert "system" in roles

    def test_usage_extraction(self) -> None:
        usage = _FakeOpenAiUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        response = _FakeOpenAiResponse(usage=usage)
        client = _FakeOpenAiClient(response)
        adapter = OpenAiSyncAdapter(client=client)

        resp = adapter.complete(_make_request(model="gpt-4o-mini"))
        assert resp.usage_total_tokens == 150

    def test_empty_content_returns_empty_text(self) -> None:
        msg = _FakeOpenAiMessage(content="")
        choice = _FakeOpenAiChoice(message=msg)
        response = _FakeOpenAiResponse(choices=[choice])
        client = _FakeOpenAiClient(response)
        adapter = OpenAiSyncAdapter(client=client)

        resp = adapter.complete(_make_request(model="gpt-4o-mini"))
        assert resp.text == ""


# ---------------------------------------------------------------------------
# Fake Anthropic SDK objects
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 20


@dataclass
class _FakeTextBlock:
    text: str = "Hello from Claude"
    type: str = "text"


@dataclass
class _FakeAnthropicResponse:
    id: str = "msg_123"
    content: list = None  # type: ignore[assignment]
    model: str = "claude-3-sonnet"
    stop_reason: str = "end_turn"
    usage: _FakeUsage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.content is None:
            self.content = [_FakeTextBlock()]
        if self.usage is None:
            self.usage = _FakeUsage()


class _FakeAnthropicMessages:
    def __init__(self, response: _FakeAnthropicResponse | None = None) -> None:
        self._response = response or _FakeAnthropicResponse()
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: _FakeAnthropicResponse | None = None) -> None:
        self.messages = _FakeAnthropicMessages(response)


class _FakeAnthropicRateLimitError(Exception):
    status_code = 429


class _FakeAnthropicServerError(Exception):
    status_code = 500


class _FakeAnthropicRateLimitMessages:
    def create(self, **kwargs: Any) -> None:
        raise _FakeAnthropicRateLimitError()


class _FakeAnthropicServerMessages:
    def create(self, **kwargs: Any) -> None:
        raise _FakeAnthropicServerError()


# ---------------------------------------------------------------------------
# Anthropic adapter tests
# ---------------------------------------------------------------------------


def _make_request(
    content: str = "hello",
    model: str = "claude-3-sonnet",
    system: str | None = None,
) -> LlmRequest:
    messages: list[LlmMessage] = []
    if system:
        messages.append(LlmMessage(role=LlmRole.SYSTEM, content=system))
    messages.append(LlmMessage(role=LlmRole.USER, content=content))
    return LlmRequest(model=model, messages=messages)


class TestAnthropicSyncAdapter:
    def test_complete_basic(self) -> None:
        client = _FakeAnthropicClient()
        adapter = AnthropicSyncAdapter(client=client)

        resp = adapter.complete(_make_request("hi"))

        assert resp.text == "Hello from Claude"
        assert resp.model == "claude-3-sonnet"
        assert resp.usage_total_tokens == 30
        assert resp.finish_reason == "end_turn"
        assert resp.request_id == "msg_123"

    def test_system_message_extracted(self) -> None:
        client = _FakeAnthropicClient()
        adapter = AnthropicSyncAdapter(client=client)

        adapter.complete(_make_request("hi", system="Be helpful"))

        kwargs = client.messages.last_kwargs
        assert kwargs["system"] == "Be helpful"
        # user message should not include system
        assert all(m["role"] != "system" for m in kwargs["messages"])

    def test_messages_forwarded(self) -> None:
        client = _FakeAnthropicClient()
        adapter = AnthropicSyncAdapter(client=client)

        adapter.complete(_make_request("what is 2+2?"))

        kwargs = client.messages.last_kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "what is 2+2?"}]
        assert kwargs["model"] == "claude-3-sonnet"

    def test_rate_limit_maps_to_domain_error(self) -> None:
        client = _FakeAnthropicClient()
        client.messages = _FakeAnthropicRateLimitMessages()
        adapter = AnthropicSyncAdapter(client=client)

        with pytest.raises(RateLimitedError):
            adapter.complete(_make_request())

    def test_server_error_maps_to_transient(self) -> None:
        client = _FakeAnthropicClient()
        client.messages = _FakeAnthropicServerMessages()
        adapter = AnthropicSyncAdapter(client=client)

        with pytest.raises(TransientLlmError):
            adapter.complete(_make_request())

    def test_usage_extraction(self) -> None:
        usage = _FakeUsage(input_tokens=100, output_tokens=50)
        response = _FakeAnthropicResponse(usage=usage)
        client = _FakeAnthropicClient(response)
        adapter = AnthropicSyncAdapter(client=client)

        resp = adapter.complete(_make_request())
        assert resp.usage_total_tokens == 150

    def test_empty_content_returns_empty_text(self) -> None:
        response = _FakeAnthropicResponse(content=[])
        client = _FakeAnthropicClient(response)
        adapter = AnthropicSyncAdapter(client=client)

        resp = adapter.complete(_make_request())
        assert resp.text == ""


# ---------------------------------------------------------------------------
# Fake Ollama HTTP responses
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=request,
                response=response,
            )


class _FakeHttpClient:
    def __init__(self, response: _FakeHttpResponse | None = None) -> None:
        self._response = response or _FakeHttpResponse(
            {
                "model": "llama3",
                "message": {"role": "assistant", "content": "Hello from Ollama"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 15,
                "eval_count": 25,
            }
        )
        self.last_url: str = ""
        self.last_json: dict = {}

    def post(self, url: str, *, json: dict, timeout: Any = None) -> _FakeHttpResponse:
        self.last_url = url
        self.last_json = json
        return self._response


# ---------------------------------------------------------------------------
# Ollama adapter tests
# ---------------------------------------------------------------------------


class TestOllamaSyncAdapter:
    def test_complete_basic(self) -> None:
        client = _FakeHttpClient()
        adapter = OllamaSyncAdapter(client=client)

        resp = adapter.complete(_make_request("hi", model="llama3"))

        assert resp.text == "Hello from Ollama"
        assert resp.model == "llama3"
        assert resp.usage_total_tokens == 40
        assert resp.finish_reason == "stop"

    def test_request_format(self) -> None:
        client = _FakeHttpClient()
        adapter = OllamaSyncAdapter(base_url="http://myhost:11434", client=client)

        adapter.complete(_make_request("test", model="llama3"))

        assert client.last_url == "http://myhost:11434/api/chat"
        assert client.last_json["model"] == "llama3"
        assert client.last_json["stream"] is False
        assert client.last_json["messages"] == [{"role": "user", "content": "test"}]

    def test_temperature_forwarded(self) -> None:
        client = _FakeHttpClient()
        adapter = OllamaSyncAdapter(client=client)

        req = LlmRequest(
            model="llama3",
            messages=[LlmMessage(role=LlmRole.USER, content="hi")],
            temperature=0.7,
        )
        adapter.complete(req)

        assert client.last_json["options"]["temperature"] == 0.7

    def test_max_tokens_forwarded(self) -> None:
        client = _FakeHttpClient()
        adapter = OllamaSyncAdapter(client=client)

        req = LlmRequest(
            model="llama3",
            messages=[LlmMessage(role=LlmRole.USER, content="hi")],
            max_output_tokens=100,
        )
        adapter.complete(req)

        assert client.last_json["options"]["num_predict"] == 100

    def test_missing_usage_returns_none(self) -> None:
        response = _FakeHttpResponse(
            {
                "model": "llama3",
                "message": {"role": "assistant", "content": "hi"},
            }
        )
        client = _FakeHttpClient(response)
        adapter = OllamaSyncAdapter(client=client)

        resp = adapter.complete(_make_request(model="llama3"))
        assert resp.usage_total_tokens is None

    def test_rate_limit_maps_to_domain_error(self) -> None:
        response = _FakeHttpResponse({}, status_code=429)
        client = _FakeHttpClient(response)
        adapter = OllamaSyncAdapter(client=client)

        with pytest.raises(RateLimitedError):
            adapter.complete(_make_request(model="llama3"))

    def test_server_error_maps_to_transient(self) -> None:
        response = _FakeHttpResponse({}, status_code=500)
        client = _FakeHttpClient(response)
        adapter = OllamaSyncAdapter(client=client)

        with pytest.raises(TransientLlmError):
            adapter.complete(_make_request(model="llama3"))
