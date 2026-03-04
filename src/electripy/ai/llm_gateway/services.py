"""Gateway services for LLM calls.

Purpose:
  - Provide high-level sync/async clients with retries, token budgeting,
    and structured output validation.

Guarantees:
  - Depend only on ports and domain models, not on provider SDKs.
  - Enforce token budgets and safety hooks when configured.

Usage:
  Basic example (sync)::

    client = LlmGatewaySyncClient(port=OpenAiSyncAdapter())
    response = client.complete(
        LlmRequest(
            model="gpt-4o-mini",
            messages=[LlmMessage.user("Say hi")],
        )
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .config import LlmGatewaySettings
from .domain import LlmMessage, LlmRequest, LlmResponse, StructuredOutputSpec
from .errors import (
    LlmGatewayError,
    PromptRejectedError,
    RateLimitedError,
    RetryExhaustedError,
    StructuredOutputError,
    TokenBudgetExceededError,
    TransientLlmError,
)
from .ports import AsyncLlmPort, GuardResult, PromptGuardPort, RedactorPort, SyncLlmPort

logger = logging.getLogger(__name__)


def _count_input_chars(messages: Sequence[LlmMessage]) -> int:
    return sum(len(message.content) for message in messages)


def _enforce_token_budget(request: LlmRequest, settings: LlmGatewaySettings) -> None:
    max_input_chars = request.max_input_chars or settings.default_max_input_chars
    current_chars = _count_input_chars(request.messages)
    if current_chars > max_input_chars:
        raise TokenBudgetExceededError(
            input_chars=current_chars,
            max_input_chars=max_input_chars,
        )


def _maybe_guard(
    request: LlmRequest,
    prompt_guard: PromptGuardPort | None,
) -> tuple[LlmRequest, GuardResult | None]:
    if prompt_guard is None:
        return request, None
    result = prompt_guard.assess(request.messages)
    if not result.allowed:
        raise PromptRejectedError(reasons=result.reasons)
    return request, result


def _maybe_log_safe(
    *,
    settings: LlmGatewaySettings,
    request: LlmRequest,
    response: LlmResponse,
) -> None:
    """Log safe, redacted summaries if enabled.

    Notes:
      - Prompts and responses are passed through the configured redactor.
      - Logging is opt-in via settings.enable_safe_logging.
    """

    if not settings.enable_safe_logging:
        return
    redactor: RedactorPort | None = settings.redactor
    if redactor is None:
        return

    prompt_text = " ".join(message.content for message in request.messages)
    redacted_prompt = redactor.redact(prompt_text)
    redacted_response = redactor.redact(response.text)

    logger.info(
        "LLM call completed: model=%s, prompt_preview=%s, response_preview=%s",
        response.model or request.model,
        redacted_prompt[:200],
        redacted_response[:200],
    )


def _validate_structured_output(data: Any, spec: StructuredOutputSpec) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, Mapping):
        errors.append("Top-level JSON value must be an object.")
        return errors
    for key, expected_type in spec.field_types.items():
        if key not in data:
            errors.append(f"Missing required field: {key}")
            continue
        value = data[key]
        if not isinstance(value, expected_type):
            errors.append(
                f"Field {key!r} expected type {expected_type.__name__}, "
                f"got {type(value).__name__}",
            )
    return errors


def _build_structured_system_message(
    base_spec: StructuredOutputSpec,
    *,
    repair: bool,
    previous_output: str | None,
) -> str:
    schema_text = base_spec.describe_for_prompt()
    if not repair:
        return (
            "You are a JSON API.\n"
            "Return ONLY a single JSON object, with no commentary.\n"
            f"{schema_text}\n"
            "Do not include any explanatory text. Output must be valid JSON."
        )

    previous = previous_output or ""
    return (
        "You are a JSON repair API.\n"
        "You will be given a previous invalid JSON output. "
        "Return ONLY a corrected JSON object that matches the schema.\n"
        f"{schema_text}\n"
        "Previous invalid JSON:\n"
        f"{previous}\n"
        "Do not include any explanatory text. Output must be valid JSON."
    )


class LlmGatewaySyncClient:
    """Synchronous LLM gateway client.

    This client adds:
      - Token budgeting.
      - Retry and rate-limit handling.
      - Optional prompt guard and redaction hooks.
      - Structured JSON output mode with one repair attempt.
    """

    def __init__(
        self,
        *,
        port: SyncLlmPort,
        settings: LlmGatewaySettings | None = None,
    ) -> None:
        self._port = port
        self._settings = settings or LlmGatewaySettings()

    @property
    def settings(self) -> LlmGatewaySettings:
        """Return the settings used by this gateway instance."""

        return self._settings

    def complete(
        self,
        request: LlmRequest,
        *,
        structured_spec: StructuredOutputSpec | None = None,
        timeout: float | None = None,
    ) -> LlmResponse:
        """Perform a completion with optional structured output mode.

        Args:
          request: Normalized LLM request.
          structured_spec: StructuredOutputSpec for strict JSON mode; if None,
            plain-text mode is used.
          timeout: Optional per-call timeout in seconds.

        Returns:
          LlmResponse with normalized content and metadata.
        """

        _enforce_token_budget(request, self._settings)
        guarded_request, guard_result = _maybe_guard(request, self._settings.prompt_guard)

        if structured_spec is None:
            response = self._call_with_retry(guarded_request, timeout=timeout)
        else:
            response = self._complete_structured(
                guarded_request,
                structured_spec=structured_spec,
                timeout=timeout,
            )

        if guard_result is not None:
            response.metadata["prompt_guard"] = {
                "allowed": guard_result.allowed,
                "score": guard_result.score,
                "reasons": list(guard_result.reasons),
            }
        _maybe_log_safe(settings=self._settings, request=request, response=response)
        return response

    def _call_with_retry(self, request: LlmRequest, *, timeout: float | None) -> LlmResponse:
        policy = self._settings.retry_policy
        attempts = 0
        delay = policy.initial_backoff_seconds
        start = time.monotonic()
        last_error_message = "unknown error"

        while attempts < policy.max_attempts:
            attempts += 1
            try:
                return self._port.complete(request, timeout=timeout)
            except RateLimitedError as exc:
                last_error_message = str(exc)
                now = time.monotonic()
                if now - start >= policy.total_timeout_seconds:
                    break
                sleep_for = exc.retry_after_seconds or delay
                sleep_for = min(sleep_for, policy.max_backoff_seconds)
                if now + sleep_for - start > policy.total_timeout_seconds:
                    break
                time.sleep(sleep_for)
                delay = min(delay * 2.0, policy.max_backoff_seconds)
                continue
            except TransientLlmError as exc:
                last_error_message = str(exc)
                now = time.monotonic()
                if now - start >= policy.total_timeout_seconds:
                    break
                sleep_for = min(delay, policy.max_backoff_seconds)
                if now + sleep_for - start > policy.total_timeout_seconds:
                    break
                time.sleep(sleep_for)
                delay = min(delay * 2.0, policy.max_backoff_seconds)
                continue
            except LlmGatewayError:
                raise

        raise RetryExhaustedError(attempts=attempts, last_error_message=last_error_message)

    def _complete_structured(
        self,
        base_request: LlmRequest,
        *,
        structured_spec: StructuredOutputSpec,
        timeout: float | None,
    ) -> LlmResponse:
        attempts = 0
        last_raw_output: str | None = None
        last_validation_errors: list[str] = []

        max_repairs = max(1, self._settings.structured_output_max_repair_attempts)

        while attempts <= max_repairs:
            repair = attempts > 0
            system_message = LlmMessage.system(
                _build_structured_system_message(
                    base_spec=structured_spec,
                    repair=repair,
                    previous_output=last_raw_output,
                ),
            )
            request = base_request.clone_with_messages(
                [system_message, *base_request.messages],
            )
            response = self._call_with_retry(request, timeout=timeout)
            last_raw_output = response.text

            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError as exc:  # pragma: no cover - error path only
                last_validation_errors = [f"JSON parse error: {exc}"]
            else:
                last_validation_errors = _validate_structured_output(parsed, structured_spec)
                if not last_validation_errors:
                    response.raw_json = parsed
                    response.metadata["structured_output"] = {
                        "schema_name": structured_spec.name,
                        "attempts": attempts + 1,
                        "valid": True,
                    }
                    return response

            attempts += 1
            if attempts > max_repairs:
                break

        truncated_output = (last_raw_output or "")[:500]
        raise StructuredOutputError(
            details="Structured output validation failed after repair attempt.",
            last_raw_output=truncated_output,
            validation_errors=tuple(last_validation_errors),
        )


class LlmGatewayAsyncClient:
    """Asynchronous LLM gateway client."""

    def __init__(
        self,
        *,
        port: AsyncLlmPort,
        settings: LlmGatewaySettings | None = None,
    ) -> None:
        self._port = port
        self._settings = settings or LlmGatewaySettings()

    @property
    def settings(self) -> LlmGatewaySettings:
        """Return the settings used by this gateway instance."""

        return self._settings

    async def complete(
        self,
        request: LlmRequest,
        *,
        structured_spec: StructuredOutputSpec | None = None,
        timeout: float | None = None,
    ) -> LlmResponse:
        """Perform an asynchronous completion with optional structured mode."""

        _enforce_token_budget(request, self._settings)
        guarded_request, guard_result = _maybe_guard(request, self._settings.prompt_guard)

        if structured_spec is None:
            response = await self._call_with_retry_async(guarded_request, timeout=timeout)
        else:
            response = await self._complete_structured_async(
                guarded_request,
                structured_spec=structured_spec,
                timeout=timeout,
            )

        if guard_result is not None:
            response.metadata["prompt_guard"] = {
                "allowed": guard_result.allowed,
                "score": guard_result.score,
                "reasons": list(guard_result.reasons),
            }
        _maybe_log_safe(settings=self._settings, request=request, response=response)
        return response

    async def _call_with_retry_async(
        self,
        request: LlmRequest,
        *,
        timeout: float | None,
    ) -> LlmResponse:
        policy = self._settings.retry_policy
        attempts = 0
        delay = policy.initial_backoff_seconds
        start = time.monotonic()
        last_error_message = "unknown error"

        while attempts < policy.max_attempts:
            attempts += 1
            try:
                return await self._port.complete(request, timeout=timeout)
            except RateLimitedError as exc:
                last_error_message = str(exc)
                now = time.monotonic()
                if now - start >= policy.total_timeout_seconds:
                    break
                sleep_for = exc.retry_after_seconds or delay
                sleep_for = min(sleep_for, policy.max_backoff_seconds)
                if now + sleep_for - start > policy.total_timeout_seconds:
                    break
                await asyncio.sleep(sleep_for)
                delay = min(delay * 2.0, policy.max_backoff_seconds)
                continue
            except TransientLlmError as exc:
                last_error_message = str(exc)
                now = time.monotonic()
                if now - start >= policy.total_timeout_seconds:
                    break
                sleep_for = min(delay, policy.max_backoff_seconds)
                if now + sleep_for - start > policy.total_timeout_seconds:
                    break
                await asyncio.sleep(sleep_for)
                delay = min(delay * 2.0, policy.max_backoff_seconds)
                continue
            except LlmGatewayError:
                raise

        raise RetryExhaustedError(attempts=attempts, last_error_message=last_error_message)

    async def _complete_structured_async(
        self,
        base_request: LlmRequest,
        *,
        structured_spec: StructuredOutputSpec,
        timeout: float | None,
    ) -> LlmResponse:
        attempts = 0
        last_raw_output: str | None = None
        last_validation_errors: list[str] = []

        max_repairs = max(1, self._settings.structured_output_max_repair_attempts)

        while attempts <= max_repairs:
            repair = attempts > 0
            system_message = LlmMessage.system(
                _build_structured_system_message(
                    base_spec=structured_spec,
                    repair=repair,
                    previous_output=last_raw_output,
                ),
            )
            request = base_request.clone_with_messages(
                [system_message, *base_request.messages],
            )
            response = await self._call_with_retry_async(request, timeout=timeout)
            last_raw_output = response.text

            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError as exc:  # pragma: no cover - error path only
                last_validation_errors = [f"JSON parse error: {exc}"]
            else:
                last_validation_errors = _validate_structured_output(parsed, structured_spec)
                if not last_validation_errors:
                    response.raw_json = parsed
                    response.metadata["structured_output"] = {
                        "schema_name": structured_spec.name,
                        "attempts": attempts + 1,
                        "valid": True,
                    }
                    return response

            attempts += 1
            if attempts > max_repairs:
                break

        truncated_output = (last_raw_output or "")[:500]
        raise StructuredOutputError(
            details="Structured output validation failed after repair attempt.",
            last_raw_output=truncated_output,
            validation_errors=tuple(last_validation_errors),
        )
