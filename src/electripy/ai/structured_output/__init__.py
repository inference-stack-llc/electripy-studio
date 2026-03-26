"""Structured Output Engine — validates LLM responses against Pydantic models.

Purpose:
  - Parse raw LLM text into typed Pydantic models with automatic retry.
  - Bridge the LLM gateway, response-robustness, and retry subsystems into
    a single composable extraction pipeline.

Guarantees:
  - No provider-specific types; works with any ``SyncLlmPort`` / ``AsyncLlmPort``.
  - Zero hard dependency on Pydantic at import time — lazy-loaded on first use.
  - Deterministic: retries use decreasing temperature for convergence.

Usage::

    from electripy.ai.structured_output import StructuredOutputExtractor, ExtractionResult
    from pydantic import BaseModel

    class Sentiment(BaseModel):
        label: str
        score: float

    extractor = StructuredOutputExtractor(llm_port=my_port)
    result: ExtractionResult[Sentiment] = extractor.extract(
        prompt="Classify sentiment: 'I love Python'",
        output_model=Sentiment,
        model="gpt-4o-mini",
    )
    assert result.parsed.label in ("positive", "negative", "neutral")
"""

from __future__ import annotations

from .domain import (
    ExtractionAttempt,
    ExtractionResult,
    ExtractionStrategy,
)
from .errors import (
    ExtractionError,
    ExtractionExhaustedError,
    SchemaGenerationError,
    StructuredOutputEngineError,
)
from .ports import (
    OutputModelPort,
    SchemaRendererPort,
)
from .services import (
    PydanticSchemaRenderer,
    StructuredOutputExtractor,
)

__all__ = [
    # Domain models
    "ExtractionAttempt",
    "ExtractionResult",
    "ExtractionStrategy",
    # Ports
    "OutputModelPort",
    "SchemaRendererPort",
    # Services
    "StructuredOutputExtractor",
    "PydanticSchemaRenderer",
    # Errors
    "StructuredOutputEngineError",
    "ExtractionError",
    "ExtractionExhaustedError",
    "SchemaGenerationError",
]
