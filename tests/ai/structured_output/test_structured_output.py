"""Tests for the Structured Output Engine."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from electripy.ai.llm_gateway.domain import LlmRequest, LlmResponse, LlmRole
from electripy.ai.structured_output import (
    ExtractionAttempt,
    ExtractionResult,
    PydanticSchemaRenderer,
    StructuredOutputExtractor,
)
from electripy.ai.structured_output.errors import (
    ExtractionExhaustedError,
    SchemaGenerationError,
)

# ---------------------------------------------------------------------------
# Fake Pydantic-like model for tests (no real Pydantic dependency needed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Sentiment:
    label: str
    score: float

    @classmethod
    def model_validate(cls, obj: dict) -> _Sentiment:
        if "label" not in obj or "score" not in obj:
            raise ValueError("missing required fields")
        return cls(label=str(obj["label"]), score=float(obj["score"]))

    @classmethod
    def model_json_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["label", "score"],
        }


@dataclass(frozen=True, slots=True)
class _BadModel:
    """Model without model_json_schema — used to test SchemaGenerationError."""

    value: str


# ---------------------------------------------------------------------------
# Fake LLM port
# ---------------------------------------------------------------------------


class _FakeLlmPort:
    """Deterministic LLM port returning pre-configured responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.requests: list[LlmRequest] = []

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        self.requests.append(request)
        text = self._responses[self._call_index]
        self._call_index += 1
        return LlmResponse(text=text, model=request.model)


# ---------------------------------------------------------------------------
# SchemaRenderer tests
# ---------------------------------------------------------------------------


class TestPydanticSchemaRenderer:
    def test_render_valid_model(self) -> None:
        renderer = PydanticSchemaRenderer()
        result = renderer.render(_Sentiment)
        parsed = json.loads(result)
        assert parsed["type"] == "object"
        assert "label" in parsed["properties"]
        assert "score" in parsed["properties"]

    def test_render_model_without_schema_raises(self) -> None:
        renderer = PydanticSchemaRenderer()
        with pytest.raises(SchemaGenerationError, match="does not expose"):
            renderer.render(_BadModel)


# ---------------------------------------------------------------------------
# Extractor — success on first attempt
# ---------------------------------------------------------------------------


class TestStructuredOutputExtractorSuccess:
    def test_extract_first_attempt(self) -> None:
        port = _FakeLlmPort(['{"label": "positive", "score": 0.95}'])
        extractor = StructuredOutputExtractor(llm_port=port)

        result = extractor.extract(
            prompt="Classify: 'I love Python'",
            output_model=_Sentiment,
            model="test-model",
        )

        assert result.parsed.label == "positive"
        assert result.parsed.score == 0.95
        assert result.total_attempts == 1
        assert len(result.attempts) == 1
        assert result.attempts[0].success is True
        assert result.model == "test-model"

    def test_extract_with_markdown_fences(self) -> None:
        port = _FakeLlmPort(['```json\n{"label": "negative", "score": 0.2}\n```'])
        extractor = StructuredOutputExtractor(llm_port=port)

        result = extractor.extract(
            prompt="Classify sentiment",
            output_model=_Sentiment,
        )

        assert result.parsed.label == "negative"
        assert result.parsed.score == 0.2

    def test_extract_with_surrounding_text(self) -> None:
        port = _FakeLlmPort(
            ['Here is the result: {"label": "neutral", "score": 0.5} hope this helps!']
        )
        extractor = StructuredOutputExtractor(llm_port=port)

        result = extractor.extract(
            prompt="Classify",
            output_model=_Sentiment,
        )

        assert result.parsed.label == "neutral"


# ---------------------------------------------------------------------------
# Extractor — retry on failure
# ---------------------------------------------------------------------------


class TestStructuredOutputExtractorRetry:
    def test_retry_succeeds_on_second_attempt(self) -> None:
        port = _FakeLlmPort(
            [
                "I think the sentiment is positive.",  # Bad: no JSON
                '{"label": "positive", "score": 0.8}',  # Good
            ]
        )
        extractor = StructuredOutputExtractor(llm_port=port, max_retries=3)

        result = extractor.extract(
            prompt="Classify",
            output_model=_Sentiment,
        )

        assert result.total_attempts == 2
        assert result.attempts[0].success is False
        assert result.attempts[0].error is not None
        assert result.attempts[1].success is True
        assert result.parsed.label == "positive"

    def test_temperature_decays_on_retry(self) -> None:
        port = _FakeLlmPort(
            [
                "bad response",
                "still bad",
                '{"label": "ok", "score": 0.5}',
            ]
        )
        extractor = StructuredOutputExtractor(
            llm_port=port,
            max_retries=3,
            initial_temperature=0.8,
            temperature_decay=0.5,
        )

        result = extractor.extract(
            prompt="Classify",
            output_model=_Sentiment,
        )

        assert result.attempts[0].temperature == 0.8
        assert result.attempts[1].temperature == pytest.approx(0.4)
        assert result.attempts[2].temperature == pytest.approx(0.2)

    def test_all_attempts_exhausted_raises(self) -> None:
        port = _FakeLlmPort(
            [
                "not json",
                "still not json",
                "nope",
            ]
        )
        extractor = StructuredOutputExtractor(llm_port=port, max_retries=3)

        with pytest.raises(ExtractionExhaustedError) as exc_info:
            extractor.extract(
                prompt="Classify",
                output_model=_Sentiment,
            )

        assert exc_info.value.attempts == 3
        assert "failed to extract _Sentiment" in exc_info.value.message

    def test_validation_error_triggers_retry(self) -> None:
        port = _FakeLlmPort(
            [
                '{"wrong_field": "oops"}',  # Valid JSON, but fails validation
                '{"label": "positive", "score": 0.9}',
            ]
        )
        extractor = StructuredOutputExtractor(llm_port=port, max_retries=2)

        result = extractor.extract(
            prompt="Classify",
            output_model=_Sentiment,
        )

        assert result.total_attempts == 2
        assert result.parsed.label == "positive"


# ---------------------------------------------------------------------------
# Extractor — system prompt injection
# ---------------------------------------------------------------------------


class TestStructuredOutputExtractorPrompt:
    def test_system_prompt_contains_schema(self) -> None:
        port = _FakeLlmPort(['{"label": "ok", "score": 0.5}'])
        extractor = StructuredOutputExtractor(llm_port=port)

        extractor.extract(
            prompt="Classify",
            output_model=_Sentiment,
        )

        req = port.requests[0]
        system_msg = req.messages[0]
        assert system_msg.role == LlmRole.SYSTEM
        assert '"label"' in system_msg.content
        assert '"score"' in system_msg.content

    def test_user_prompt_passed_through(self) -> None:
        port = _FakeLlmPort(['{"label": "ok", "score": 0.5}'])
        extractor = StructuredOutputExtractor(llm_port=port)

        extractor.extract(
            prompt="Classify: 'hello world'",
            output_model=_Sentiment,
        )

        req = port.requests[0]
        user_msg = req.messages[1]
        assert user_msg.role == LlmRole.USER
        assert "hello world" in user_msg.content


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def test_total_attempts_computed(self) -> None:
        attempts = (
            ExtractionAttempt(attempt_number=1, raw_text="bad", temperature=0.2, error="err"),
            ExtractionAttempt(attempt_number=2, raw_text="good", temperature=0.1, success=True),
        )
        result = ExtractionResult(
            parsed="ok",
            attempts=attempts,
            model="m",
            schema_prompt="s",
        )
        assert result.total_attempts == 2

    def test_extraction_attempt_defaults(self) -> None:
        attempt = ExtractionAttempt(attempt_number=1, raw_text="t", temperature=0.5)
        assert attempt.error is None
        assert attempt.success is False
