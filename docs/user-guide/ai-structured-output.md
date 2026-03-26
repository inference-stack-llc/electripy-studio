# Structured Output Engine

The **Structured Output Engine** extracts typed, validated data from raw
LLM responses.  Hand it a Pydantic model and a prompt; the engine asks
the LLM, parses the JSON, validates it against the model, and retries
with decreasing temperature when parsing or validation fails.

## When to use it

- You need **strongly-typed** objects from an LLM, not raw text.
- You want automatic retry with temperature decay for convergence.
- You already use the LLM Gateway and want structured output on top.

## Core concepts

- **Domain models**:
    - `ExtractionResult[T]` — generic frozen result carrying the parsed
      object, raw LLM text, attempt count, and strategy used.
    - `ExtractionAttempt` — details of one parse/validate round.
    - `ExtractionStrategy` — enum: `json_prompt`, `tool_call`.
- **Ports**:
    - `OutputModelPort` — protocol describing a Pydantic-like model class.
    - `SchemaRendererPort` — protocol for turning a model into a JSON-schema
      string for the system prompt.
- **Services**:
    - `StructuredOutputExtractor` — the main entrypoint.
    - `PydanticSchemaRenderer` — default renderer using Pydantic v2
      `model_json_schema()`.
- **Errors**:
    - `ExtractionExhaustedError` — raised when all retry attempts fail.
    - `ExtractionError` / `SchemaGenerationError` — specific failure modes.

## Basic example

```python
from pydantic import BaseModel

from electripy.ai.llm_gateway import LlmMessage, LlmRequest, build_llm_sync_client
from electripy.ai.structured_output import (
    ExtractionResult,
    StructuredOutputExtractor,
)


class Sentiment(BaseModel):
    label: str
    score: float


client = build_llm_sync_client("openai")
extractor = StructuredOutputExtractor(llm_port=client)

result: ExtractionResult[Sentiment] = extractor.extract(
    prompt="Classify sentiment: 'I love Python'",
    output_model=Sentiment,
    model="gpt-4o-mini",
)

print(result.parsed.label)   # "positive"
print(result.parsed.score)   # 0.95
print(result.attempts)       # 1 (first-try success)
```

## Retry with temperature decay

When the LLM returns invalid JSON or data that fails Pydantic
validation, the extractor automatically retries with a lower
temperature:

```python
result = extractor.extract(
    prompt="Extract meeting action items as JSON.",
    output_model=ActionItems,
    model="gpt-4o-mini",
    max_attempts=3,        # default: 3
    initial_temperature=0.7,
)
```

Each retry decreases temperature linearly towards 0, nudging the model
toward more deterministic output.  If all attempts fail, an
`ExtractionExhaustedError` is raised with details of every attempt.

## Custom schema renderer

The default `PydanticSchemaRenderer` embeds the JSON schema into the
system prompt.  You can swap it for any implementation of
`SchemaRendererPort`:

```python
from electripy.ai.structured_output import SchemaRendererPort


class MyRenderer(SchemaRendererPort):
    def render(self, model: type) -> str:
        return "Return only valid JSON matching: {\"name\": str, \"age\": int}"


extractor = StructuredOutputExtractor(
    llm_port=client,
    schema_renderer=MyRenderer(),
)
```

## Integration with other components

The Structured Output Engine composes naturally with other ElectriPy
components:

- **LLM Caching** — wrap the LLM port with `CachedLlmPort` to cache
  structured extraction results.
- **Replay Tape** — wrap with `RecordingLlmPort` to capture extraction
  calls for offline test replay.
- **Eval Assertions** — validate the `.parsed` result with
  `matches_json_schema()` or `passes_predicate()` in CI tests.
