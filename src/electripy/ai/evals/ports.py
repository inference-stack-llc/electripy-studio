"""Ports (Protocol interfaces) for the evaluation framework.

These runtime-checkable protocols define the pluggable boundaries:

- **DatasetLoaderPort** — loads evaluation datasets from any source.
- **ScorerPort** — scores a model output against an evaluation case.
- **ReportWriterPort** — writes evaluation summaries to a destination.
- **ArtifactStorePort** — persists evaluation artifacts.
- **ModelInvocationPort** — invokes a model for offline evaluation.

All concrete implementations live in
:mod:`electripy.ai.evals.adapters` and
:mod:`electripy.ai.evals.scorers`.

Example::

    from electripy.ai.evals.ports import ScorerPort

    class MySentimentScorer:
        @property
        def name(self) -> str:
            return "sentiment"

        def score(self, case, actual_output, **kwargs):
            ...
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .domain import (
    EvalArtifact,
    EvalCase,
    EvalDataset,
    EvalScore,
    EvalSummary,
)

__all__ = [
    "ArtifactStorePort",
    "DatasetLoaderPort",
    "ModelInvocationPort",
    "ReportWriterPort",
    "ScorerPort",
]


@runtime_checkable
class DatasetLoaderPort(Protocol):
    """Loads evaluation datasets from an external source.

    Implementations may read from JSONL files, databases, APIs, or
    in-memory fixtures.
    """

    def load(self, source: str) -> EvalDataset:
        """Load a dataset from the given source.

        Args:
            source: Path, URI, or identifier for the dataset.

        Returns:
            A loaded evaluation dataset.
        """
        ...


@runtime_checkable
class ScorerPort(Protocol):
    """Scores a model output against an evaluation case.

    Each scorer produces one or more :class:`EvalScore` instances.
    Scorers should be stateless and deterministic.
    """

    @property
    def name(self) -> str:
        """Human-readable scorer name used in score records."""
        ...

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        """Score the actual output for a given case.

        Args:
            case: The evaluation case with expectations.
            actual_output: The model's actual output.
            **kwargs: Additional context (e.g. retrieved_ids,
                tool_calls).

        Returns:
            A list of scores produced by this scorer.
        """
        ...


@runtime_checkable
class ReportWriterPort(Protocol):
    """Writes evaluation summaries to a destination.

    Implementations may write JSON, Markdown, CSV, or push to an API.
    """

    def write(self, summary: EvalSummary, destination: str) -> None:
        """Write the evaluation summary.

        Args:
            summary: The evaluation summary to write.
            destination: Path or identifier for the output location.
        """
        ...


@runtime_checkable
class ArtifactStorePort(Protocol):
    """Persists evaluation artifacts.

    Implementations may write to the filesystem, object storage, or
    a database.
    """

    def save(self, artifact: EvalArtifact, run_id: str) -> str:
        """Save an artifact and return its storage key or path.

        Args:
            artifact: The artifact to persist.
            run_id: The evaluation run identifier for namespacing.

        Returns:
            A key or path identifying the stored artifact.
        """
        ...


@runtime_checkable
class ModelInvocationPort(Protocol):
    """Invokes a model to produce an output for an evaluation case.

    Used for offline evaluations where the runner needs to call
    a model.  For evaluations where outputs are pre-computed, this
    port is not required.
    """

    def invoke(self, input_text: str, metadata: dict[str, Any] | None = None) -> str:
        """Invoke the model and return its output.

        Args:
            input_text: The prompt or query to send.
            metadata: Optional metadata for the invocation.

        Returns:
            The model's text output.
        """
        ...
