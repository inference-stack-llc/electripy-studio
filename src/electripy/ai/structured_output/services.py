"""Services for the Structured Output Engine."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from electripy.ai.llm_gateway.domain import LlmMessage, LlmRequest, LlmRole
from electripy.ai.llm_gateway.ports import SyncLlmPort
from electripy.ai.response_robustness.services import extract_json_object

from .domain import ExtractionAttempt, ExtractionResult, ExtractionStrategy
from .errors import ExtractionExhaustedError, SchemaGenerationError

T = TypeVar("T")

_DEFAULT_MAX_RETRIES = 3
_TEMPERATURE_DECAY = 0.5
_SYSTEM_PROMPT = (
    "You are a structured-data extraction assistant. "
    "You MUST respond with a single valid JSON object matching the schema below. "
    "Do NOT include markdown fences, explanations, or any text outside the JSON object.\n\n"
    "Schema:\n{schema}"
)


class PydanticSchemaRenderer:
    """Renders a Pydantic model class into a prompt-friendly JSON schema string.

    This renderer calls ``model_json_schema()`` and formats the result as
    indented JSON suitable for injection into an LLM system prompt.
    """

    __slots__ = ()

    def render(self, model: type) -> str:
        """Return a JSON-schema string for *model*.

        Args:
          model: A Pydantic ``BaseModel`` subclass (or any class exposing
            ``model_json_schema()``).

        Returns:
          Indented JSON string.

        Raises:
          SchemaGenerationError: If the model does not expose a schema method.
        """
        schema_fn = getattr(model, "model_json_schema", None)
        if schema_fn is None:
            raise SchemaGenerationError(
                f"{model.__name__} does not expose model_json_schema(); "
                "pass a Pydantic BaseModel subclass"
            )
        try:
            schema = schema_fn()
        except Exception as exc:
            raise SchemaGenerationError(
                f"failed to generate schema for {model.__name__}: {exc}"
            ) from exc
        return json.dumps(schema, indent=2)


class StructuredOutputExtractor:
    """Extract typed objects from LLM output with auto-retry.

    Composes the LLM gateway, response-robustness JSON extractor, and a
    schema renderer into a single extraction pipeline.  On each failed parse,
    the temperature is decayed and the LLM is re-prompted until *max_retries*
    is exhausted.

    Args:
      llm_port: A synchronous LLM port for completions.
      schema_renderer: Renderer turning a model class into a schema string.
        Defaults to :class:`PydanticSchemaRenderer`.
      max_retries: Maximum number of extraction attempts (default 3).
      initial_temperature: Starting temperature for the first attempt.
      temperature_decay: Multiplicative decay applied after each failure.
      strategy: Extraction strategy (currently ``json_mode`` and
        ``prompt_coerce`` are supported; both inject the schema into the
        system prompt).
    """

    __slots__ = (
        "_llm_port",
        "_renderer",
        "_max_retries",
        "_initial_temperature",
        "_temperature_decay",
        "_strategy",
    )

    def __init__(
        self,
        *,
        llm_port: SyncLlmPort,
        schema_renderer: PydanticSchemaRenderer | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        initial_temperature: float = 0.2,
        temperature_decay: float = _TEMPERATURE_DECAY,
        strategy: ExtractionStrategy = ExtractionStrategy.PROMPT_COERCE,
    ) -> None:
        self._llm_port = llm_port
        self._renderer = schema_renderer or PydanticSchemaRenderer()
        self._max_retries = max_retries
        self._initial_temperature = initial_temperature
        self._temperature_decay = temperature_decay
        self._strategy = strategy

    def extract(
        self,
        *,
        prompt: str,
        output_model: type[T],
        model: str = "gpt-4o-mini",
        max_output_tokens: int | None = None,
    ) -> ExtractionResult[T]:
        """Run the extraction pipeline.

        Args:
          prompt: The user prompt describing what to extract.
          output_model: A Pydantic ``BaseModel`` subclass to validate against.
          model: LLM model identifier.
          max_output_tokens: Optional token limit for the LLM response.

        Returns:
          An :class:`ExtractionResult` containing the validated object and
          attempt history.

        Raises:
          ExtractionExhaustedError: If all attempts fail.
          SchemaGenerationError: If the output model schema cannot be rendered.
        """
        schema_prompt = self._renderer.render(output_model)
        system_content = _SYSTEM_PROMPT.format(schema=schema_prompt)

        attempts: list[ExtractionAttempt] = []
        temperature = self._initial_temperature
        last_error = ""

        for attempt_num in range(1, self._max_retries + 1):
            request = LlmRequest(
                model=model,
                messages=[
                    LlmMessage(role=LlmRole.SYSTEM, content=system_content),
                    LlmMessage(role=LlmRole.USER, content=prompt),
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            response = self._llm_port.complete(request)
            raw_text = response.text

            try:
                parsed = _parse_and_validate(raw_text, output_model)
            except Exception as exc:  # noqa: BLE001 — catch validation errors
                last_error = str(exc)
                attempts.append(
                    ExtractionAttempt(
                        attempt_number=attempt_num,
                        raw_text=raw_text,
                        temperature=temperature,
                        error=last_error,
                        success=False,
                    )
                )
                temperature *= self._temperature_decay
                continue

            attempts.append(
                ExtractionAttempt(
                    attempt_number=attempt_num,
                    raw_text=raw_text,
                    temperature=temperature,
                    success=True,
                )
            )
            return ExtractionResult(
                parsed=parsed,
                attempts=tuple(attempts),
                model=model,
                schema_prompt=schema_prompt,
            )

        raise ExtractionExhaustedError(
            message=f"failed to extract {output_model.__name__}",
            attempts=len(attempts),
            last_error=last_error,
        )


def _parse_and_validate(raw_text: str, output_model: type[T]) -> T:
    """Parse raw LLM text into a validated instance of *output_model*."""
    json_str = extract_json_object(raw_text)
    data: dict[str, Any] = json.loads(json_str)
    validate_fn = getattr(output_model, "model_validate", None)
    if validate_fn is not None:
        return validate_fn(data)  # type: ignore[no-any-return]
    # Fallback: try direct construction from dict
    return output_model(**data)
