"""Enterprise evaluation framework for AI systems.

Purpose:
  - Run structured, dataset-driven evaluations against LLM and AI systems.
  - Score model outputs with pluggable scorers for text, retrieval, tool
    calls, JSON structure, and custom criteria.
  - Compare evaluation runs against baselines to detect regressions.
  - Generate machine-readable reports for CI gating and engineering review.

Guarantees:
  - Evaluation runs are deterministic from stable inputs and scorers.
  - All domain models are immutable where possible.
  - The framework is model-vendor agnostic.
  - Scorers compose and extend cleanly.

Usage:
  Basic example::

    from electripy.ai.evals import (
        EvalCase,
        EvalDataset,
        EvalRunner,
        ExactMatchScorer,
        GroundTruth,
    )

    dataset = EvalDataset(
        name="capitals",
        cases=[
            EvalCase(
                case_id="q1",
                input="What is the capital of France?",
                ground_truth=GroundTruth(reference_output="Paris"),
            ),
        ],
    )
    runner = EvalRunner(scorers=[ExactMatchScorer()])
    run = runner.run_dataset(dataset, outputs={"q1": "Paris"})
    print(run.summary.pass_rate)
"""

from __future__ import annotations

from .adapters import (
    CallbackModelInvocation,
    FileArtifactStore,
    JsonlDatasetLoader,
    JsonReportWriter,
    MarkdownReportWriter,
)
from .domain import (
    EvalArtifact,
    EvalCase,
    EvalDataset,
    EvalFailure,
    EvalMetric,
    EvalResult,
    EvalRun,
    EvalScore,
    EvalSummary,
    GroundTruth,
    RegressionComparison,
    RegressionDelta,
    RetrievalExpectation,
    ToolCallExpectation,
)
from .errors import (
    DatasetLoadError,
    EvalError,
    RegressionError,
    ScorerError,
)
from .ports import (
    ArtifactStorePort,
    DatasetLoaderPort,
    ModelInvocationPort,
    ReportWriterPort,
    ScorerPort,
)
from .scorers import (
    CompositeScorer,
    ContainsScorer,
    ExactMatchScorer,
    JsonStructureScorer,
    NormalizedTextScorer,
    RetrievalScorer,
    ThresholdScorer,
    ToolCallScorer,
)
from .services import (
    EvalRunner,
)

__all__ = [
    # Domain
    "EvalArtifact",
    "EvalCase",
    "EvalDataset",
    "EvalFailure",
    "EvalMetric",
    "EvalResult",
    "EvalRun",
    "EvalScore",
    "EvalSummary",
    "GroundTruth",
    "RegressionComparison",
    "RegressionDelta",
    "RetrievalExpectation",
    "ToolCallExpectation",
    # Errors
    "DatasetLoadError",
    "EvalError",
    "RegressionError",
    "ScorerError",
    # Ports
    "ArtifactStorePort",
    "DatasetLoaderPort",
    "ModelInvocationPort",
    "ReportWriterPort",
    "ScorerPort",
    # Adapters
    "CallbackModelInvocation",
    "FileArtifactStore",
    "JsonlDatasetLoader",
    "JsonReportWriter",
    "MarkdownReportWriter",
    # Scorers
    "CompositeScorer",
    "ContainsScorer",
    "ExactMatchScorer",
    "JsonStructureScorer",
    "NormalizedTextScorer",
    "RetrievalScorer",
    "ThresholdScorer",
    "ToolCallScorer",
    # Services
    "EvalRunner",
]
