"""Domain models for the RAG evaluation runner.

Purpose:
  - Define dataset records, experiment configuration, and result models
    in a framework-agnostic way.

Guarantees:
  - No third-party client or database types appear in this module.
  - Data models are fully typed and suitable for static analysis.

Usage:
  Basic example::

    from electripy.ai.rag_eval_runner import CorpusRecord, QueryRecord

    corpus = CorpusRecord(id="doc-1", source_uri="memory://", text="Hello", metadata=None)
    query = QueryRecord(id="q1", query="hello", relevant_ids=["doc-1:0"], metadata=None)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from electripy.ai.rag.config import ChunkingConfig


@dataclass(slots=True)
class CorpusRecord:
    """Single corpus record consumed by the evaluation runner.

    Attributes:
        id: Stable document identifier.
        source_uri: Optional logical source location (path, URL, etc.).
        text: Raw document text.
        metadata: Optional metadata dictionary.
    """

    id: str
    text: str
    source_uri: str | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(slots=True)
class QueryRecord:
    """Single query record with ground-truth relevance.

    Notes:
        ``relevant_ids`` are interpreted as **chunk ids** produced by
        the chunker. This choice is documented so that callers can
        construct ground-truth files consistently.
    """

    id: str
    query: str
    relevant_ids: list[str]
    metadata: Mapping[str, object] | None = None


@dataclass(slots=True)
class ChunkingVariant:
    """Named chunking configuration to be used in experiments."""

    name: str
    config: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass(slots=True)
class EmbedderVariant:
    """Named embedder configuration for experiments.

    Attributes:
        name: Logical embedder name (for example, "fake", "openai").
        provider: Provider identifier; for now this mirrors ``name`` but
            is kept separate to allow future differentiation (for
            example, "openai", "local").
        params: Optional provider-specific configuration for adapter
            construction. This is treated as opaque configuration and
            must be JSON-serialisable to participate in experiment id
            hashing.
    """

    name: str
    provider: str
    params: Mapping[str, object] | None = None


@dataclass(slots=True)
class RunMetadata:
    """Metadata describing a full evaluation run."""

    name: str
    timestamp_iso: str
    git_sha: str | None = None


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for a matrix of RAG evaluation experiments.

    The effective experiment space is ``chunk_variants × embedder_variants
    × top_k_values``.

    Attributes:
        run_metadata: High-level run metadata for reporting.
        corpus_path: Path to the corpus JSONL file.
        queries_path: Path to the queries JSONL file.
        chunk_variants: List of chunking variants to test.
        embedder_variants: List of embedder variants to test.
        top_k_values: Distinct ``k`` values to evaluate.
        output_json_path: Optional path for a JSON report.
        output_csv_path: Optional path for a CSV report.
        include_per_query_breakdown: Whether to emit per-query metrics
            in the JSON report.
    """

    run_metadata: RunMetadata
    corpus_path: Path
    queries_path: Path
    chunk_variants: list[ChunkingVariant]
    embedder_variants: list[EmbedderVariant]
    top_k_values: list[int]
    output_json_path: Path | None = None
    output_csv_path: Path | None = None
    include_per_query_breakdown: bool = True

    def __post_init__(self) -> None:
        if not self.top_k_values:
            raise ValueError("top_k_values must not be empty")
        if any(k <= 0 for k in self.top_k_values):
            raise ValueError("top_k_values must all be positive")
        if not self.chunk_variants:
            raise ValueError("chunk_variants must not be empty")
        if not self.embedder_variants:
            raise ValueError("embedder_variants must not be empty")


@dataclass(slots=True)
class PerQueryMetrics:
    """Per-query evaluation metrics for a single experiment."""

    query_id: str
    query_text: str
    metrics: Mapping[str, float]


@dataclass(slots=True)
class ExperimentResult:
    """Aggregate and per-query results for a single experiment."""

    experiment_id: str
    chunker_name: str
    embedder_name: str
    top_k: int
    aggregate_metrics: Mapping[str, float]
    per_query: Sequence[PerQueryMetrics]


@dataclass(slots=True)
class RunResult:
    """Full run result containing all experiment results."""

    run_metadata: RunMetadata
    experiments: Sequence[ExperimentResult]


_METRIC_NAME_HIT_RATE: Final[str] = "hit_rate"
_METRIC_NAME_PRECISION: Final[str] = "precision"
_METRIC_NAME_RECALL: Final[str] = "recall"
_METRIC_NAME_MRR: Final[str] = "mrr"


def metric_key(name: str, k: int) -> str:
    """Return a canonical metric key of the form ``"name@k"``.

    Args:
        name: Base metric name (for example, ``"hit_rate"``).
        k: Cut-off rank.

    Returns:
        Combined metric key.
    """

    return f"{name}@{k}"
