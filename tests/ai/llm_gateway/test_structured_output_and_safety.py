"""Tests for structured output and safety hooks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from electripy.ai.llm_gateway import (
    LlmGatewaySettings,
    LlmGatewaySyncClient,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    StructuredOutputSpec,
)
from electripy.ai.llm_gateway.errors import StructuredOutputError
from electripy.ai.llm_gateway.ports import GuardResult, PromptGuardPort, SyncLlmPort


@dataclass
class StructuredFakePort(SyncLlmPort):
    """Port that returns invalid JSON once, then valid JSON."""

    first_invalid: str
    second_valid: dict[str, object]
    calls: int = 0

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(text=self.first_invalid, model=request.model)
        return LlmResponse(text=json.dumps(self.second_valid), model=request.model)


def test_structured_output_repairs_once() -> None:
    spec = StructuredOutputSpec(
        name="TestSchema",
        field_types={"name": str, "age": int},
    )
    port = StructuredFakePort(
        first_invalid="not json",
        second_valid={"name": "Alice", "age": 30},
    )
    client = LlmGatewaySyncClient(port=port)
    request = LlmRequest(
        model="gpt-test",
        messages=[LlmMessage.user("Return a user profile.")],
    )

    response = client.complete(request, structured_spec=spec)

    assert port.calls == 2
    assert response.raw_json is not None
    assert response.raw_json["name"] == "Alice"
    assert response.metadata["structured_output"]["valid"] is True


@dataclass
class AlwaysInvalidPort(SyncLlmPort):
    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        return LlmResponse(text="not json", model=request.model)


def test_structured_output_error_after_repair_attempt() -> None:
    spec = StructuredOutputSpec(
        name="TestSchema",
        field_types={"name": str},
    )
    port = AlwaysInvalidPort()
    settings = LlmGatewaySettings(structured_output_max_repair_attempts=1)
    client = LlmGatewaySyncClient(port=port, settings=settings)
    request = LlmRequest(
        model="gpt-test",
        messages=[LlmMessage.user("Return a user profile.")],
    )

    with pytest.raises(StructuredOutputError):
        client.complete(request, structured_spec=spec)


class AllowAllGuard(PromptGuardPort):
    def assess(self, messages: Sequence[LlmMessage]) -> GuardResult:
        return GuardResult(allowed=True, score=1.0, reasons=())


def test_prompt_guard_metadata_populated() -> None:
    port = StructuredFakePort(
        first_invalid="not json",
        second_valid={"name": "Alice", "age": 30},
    )
    settings = LlmGatewaySettings(prompt_guard=AllowAllGuard())
    client = LlmGatewaySyncClient(port=port, settings=settings)
    request = LlmRequest(
        model="gpt-test",
        messages=[LlmMessage.user("Return a user profile.")],
    )
    spec = StructuredOutputSpec(
        name="TestSchema",
        field_types={"name": str, "age": int},
    )

    response = client.complete(request, structured_spec=spec)

    guard_meta = response.metadata.get("prompt_guard")
    assert guard_meta is not None
    assert guard_meta["allowed"] is True
