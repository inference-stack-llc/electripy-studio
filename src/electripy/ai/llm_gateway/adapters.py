"""Adapters for the LLM Gateway.

Purpose:
    - Implement ports using concrete third-party libraries (OpenAI by default).
    - Provide HTTP+JSON adapters that work with OpenAI-like chat APIs using
        :mod:`httpx`, suitable for providers such as OpenRouter, Copilot, or
        custom gateways.
    - Provide example adapters for prompt guard and redaction.

Guarantees:
    - Adapters conform to the port Protocols.
    - Provider-specific exceptions are mapped to domain exceptions.

Usage:
    Basic example (OpenAI)::

        from electripy.ai.llm_gateway import OpenAiSyncAdapter

        adapter = OpenAiSyncAdapter()

    HTTP JSON example (OpenRouter-like)::

        from electripy.ai.llm_gateway import HttpJsonChatSyncAdapter

        adapter = HttpJsonChatSyncAdapter(
            base_url="https://api.openrouter.ai",
            path="/v1/chat/completions",
            api_key="sk-...",
        )
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from .domain import LlmMessage, LlmRequest, LlmResponse
from .errors import RateLimitedError, TransientLlmError
from .ports import AsyncLlmPort, GuardResult, PromptGuardPort, RedactorPort, SyncLlmPort


def _map_openai_exception(exc: Exception) -> None:
    """Map an OpenAI exception to a domain error and raise it.

    This helper inspects common attributes of OpenAI errors to derive
    status_code and retry_after_seconds hints without leaking the original type.
    """

    status_code: int | None = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    headers: Mapping[str, Any] | None = None
    if response is not None:
        status_code = getattr(response, "status_code", status_code)
        headers = getattr(response, "headers", None)

    retry_after_seconds: float | None = None
    if headers:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if isinstance(retry_after, str):
            try:
                retry_after_seconds = float(retry_after)
            except ValueError:  # pragma: no cover - defensive
                retry_after_seconds = None

    if status_code == 429:
        raise RateLimitedError(
            message="Provider rate limit encountered",
            status_code=status_code,
            retry_after_seconds=retry_after_seconds,
        ) from None

    if status_code is not None and status_code >= 500:
        raise TransientLlmError(
            message="Transient provider error", status_code=status_code
        ) from None

    # Fallback: treat as non-retryable provider error.
    raise TransientLlmError(message=str(exc), status_code=status_code) from None


class OpenAiSyncAdapter(SyncLlmPort):
    """Synchronous OpenAI adapter implementing SyncLlmPort.

    Notes:
      - Requires the official OpenAI Python SDK (``openai>=1.0``).
      - Only the minimal ``chat.completions.create`` surface is used.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:  # pragma: no cover - import-time dependency
            raise ImportError(
                "OpenAiSyncAdapter requires the `openai` package. "
                "Install with `pip install openai`.",
            ) from exc

        self._client = openai_module.OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        messages = [
            {"role": message.role.value, "content": message.content} for message in request.messages
        ]
        messages_param: Any = messages
        response: Any
        try:
            response = self._client.chat.completions.create(
                model=request.model,
                messages=messages_param,
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            _map_openai_exception(exc)
            raise AssertionError("_map_openai_exception must raise") from exc

        choice = response.choices[0]
        text: str = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        total_tokens: int | None = getattr(usage, "total_tokens", None)
        request_id: str | None = getattr(response, "id", None)
        model_used: str | None = getattr(response, "model", None)

        return LlmResponse(
            text=text,
            raw_json=None,
            usage_total_tokens=total_tokens,
            finish_reason=finish_reason,
            request_id=request_id,
            model=model_used,
        )


class OpenAiAsyncAdapter(AsyncLlmPort):
    """Asynchronous OpenAI adapter implementing AsyncLlmPort."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:  # pragma: no cover - import-time dependency
            raise ImportError(
                "OpenAiAsyncAdapter requires the `openai` package. "
                "Install with `pip install openai`.",
            ) from exc

        self._client = openai_module.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    async def complete(
        self,
        request: LlmRequest,
        *,
        timeout: float | None = None,
    ) -> LlmResponse:
        messages = [
            {"role": message.role.value, "content": message.content} for message in request.messages
        ]
        messages_param: Any = messages
        response: Any
        try:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=messages_param,
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            _map_openai_exception(exc)
            raise AssertionError("_map_openai_exception must raise") from exc

        choice = response.choices[0]
        text: str = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        total_tokens: int | None = getattr(usage, "total_tokens", None)
        request_id: str | None = getattr(response, "id", None)
        model_used: str | None = getattr(response, "model", None)

        return LlmResponse(
            text=text,
            raw_json=None,
            usage_total_tokens=total_tokens,
            finish_reason=finish_reason,
            request_id=request_id,
            model=model_used,
        )


class HttpJsonChatSyncAdapter(SyncLlmPort):
    """Generic HTTP+JSON chat adapter using :mod:`httpx`.

    This adapter targets OpenAI-style chat completion APIs and is suitable
    for providers that expose a compatible surface, such as OpenRouter or
    custom gateways. For providers with different payload/response schemas
    (for example Claude messages APIs), subclass this adapter and override
    :meth:`_build_payload` and :meth:`_parse_response_json`.
    """

    def __init__(
        self,
        *,
        base_url: str,
        path: str = "/v1/chat/completions",
        api_key: str | None = None,
        auth_header_name: str = "Authorization",
        client: httpx.Client | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + path
        self._headers: dict[str, str] = {}
        if api_key is not None:
            self._headers[auth_header_name] = f"Bearer {api_key}"
        self._client = client or httpx.Client(timeout=30.0)

    def _build_payload(self, request: LlmRequest) -> dict[str, Any]:
        return {
            "model": request.model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }

    def _parse_response_json(self, data: Mapping[str, Any]) -> LlmResponse:
        choices = data.get("choices") or []
        if not choices:
            raise TransientLlmError("Missing 'choices' in response", status_code=None)
        first = choices[0]
        message = first.get("message") or {}
        text = str(message.get("content") or "")
        finish_reason = first.get("finish_reason")
        usage = data.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        request_id = data.get("id")
        model_used = data.get("model")

        return LlmResponse(
            text=text,
            raw_json=data,
            usage_total_tokens=total_tokens,
            finish_reason=finish_reason,
            request_id=request_id,
            model=model_used,
        )

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        payload = self._build_payload(request)
        try:
            response = self._client.post(
                self._url,
                headers=self._headers,
                json=payload,
                timeout=timeout,
            )
        except httpx.RequestError as exc:  # network / DNS / connection issues
            raise TransientLlmError(message=str(exc), status_code=None) from None

        status = response.status_code
        retry_after_header = response.headers.get("Retry-After") or response.headers.get(
            "retry-after",
        )
        retry_after_seconds: float | None = None
        if isinstance(retry_after_header, str):
            try:
                retry_after_seconds = float(retry_after_header)
            except ValueError:  # pragma: no cover - defensive
                retry_after_seconds = None

        if status == 429:
            raise RateLimitedError(
                message="Provider rate limit encountered",
                status_code=status,
                retry_after_seconds=retry_after_seconds,
            ) from None

        if status >= 500:
            raise TransientLlmError("Transient provider error", status_code=status) from None

        if status >= 400:
            raise TransientLlmError(
                f"HTTP error from provider: {status}",
                status_code=status,
            ) from None

        data = response.json()
        return self._parse_response_json(data)


class HttpJsonChatAsyncAdapter(AsyncLlmPort):
    """Asynchronous variant of :class:`HttpJsonChatSyncAdapter`."""

    def __init__(
        self,
        *,
        base_url: str,
        path: str = "/v1/chat/completions",
        api_key: str | None = None,
        auth_header_name: str = "Authorization",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + path
        self._headers: dict[str, str] = {}
        if api_key is not None:
            self._headers[auth_header_name] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(timeout=30.0)

    def _build_payload(self, request: LlmRequest) -> dict[str, Any]:
        return {
            "model": request.model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }

    def _parse_response_json(self, data: Mapping[str, Any]) -> LlmResponse:
        choices = data.get("choices") or []
        if not choices:
            raise TransientLlmError("Missing 'choices' in response", status_code=None)
        first = choices[0]
        message = first.get("message") or {}
        text = str(message.get("content") or "")
        finish_reason = first.get("finish_reason")
        usage = data.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        request_id = data.get("id")
        model_used = data.get("model")

        return LlmResponse(
            text=text,
            raw_json=data,
            usage_total_tokens=total_tokens,
            finish_reason=finish_reason,
            request_id=request_id,
            model=model_used,
        )

    async def complete(
        self,
        request: LlmRequest,
        *,
        timeout: float | None = None,
    ) -> LlmResponse:
        payload = self._build_payload(request)
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                json=payload,
                timeout=timeout,
            )
        except httpx.RequestError as exc:  # network / DNS / connection issues
            raise TransientLlmError(message=str(exc), status_code=None) from None

        status = response.status_code
        retry_after_header = response.headers.get("Retry-After") or response.headers.get(
            "retry-after",
        )
        retry_after_seconds: float | None = None
        if isinstance(retry_after_header, str):
            try:
                retry_after_seconds = float(retry_after_header)
            except ValueError:  # pragma: no cover - defensive
                retry_after_seconds = None

        if status == 429:
            raise RateLimitedError(
                message="Provider rate limit encountered",
                status_code=status,
                retry_after_seconds=retry_after_seconds,
            ) from None

        if status >= 500:
            raise TransientLlmError("Transient provider error", status_code=status) from None

        if status >= 400:
            raise TransientLlmError(
                f"HTTP error from provider: {status}",
                status_code=status,
            ) from None

        data = response.json()
        return self._parse_response_json(data)


class SimpleRedactor(RedactorPort):
    """Simple regex-based redactor example.

    This is intentionally minimal and intended as an example only.
    """

    EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    DIGIT_PATTERN = re.compile(r"\d{4,}")

    def redact(self, text: str) -> str:
        redacted = self.EMAIL_PATTERN.sub("[redacted-email]", text)
        redacted = self.DIGIT_PATTERN.sub("[redacted-number]", redacted)
        return redacted


class HeuristicPromptGuard(PromptGuardPort):
    """Heuristic prompt guard that detects simple injection patterns.

    It looks for suspicious substrings such as "ignore previous", "system prompt",
    "exfiltrate", etc., in a case-insensitive manner.
    """

    SUSPICIOUS_PATTERNS: tuple[str, ...] = (
        "ignore previous",
        "ignore all previous instructions",
        "system prompt",
        "exfiltrate",
        "leak",
        "jailbreak",
    )

    def assess(self, messages: Sequence[LlmMessage]) -> GuardResult:
        lower_text = " ".join(message.content.lower() for message in messages)
        reasons: list[str] = [
            f"Found suspicious pattern: {pattern!r}"
            for pattern in self.SUSPICIOUS_PATTERNS
            if pattern in lower_text
        ]
        allowed = not reasons
        score = 1.0 if allowed else 0.0
        return GuardResult(allowed=allowed, score=score, reasons=tuple(reasons))
