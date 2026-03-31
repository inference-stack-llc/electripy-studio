"""Domain models for the evaluation framework.

All models are frozen dataclasses for immutability and hashability.
Mutable containers use ``tuple`` instead of ``list`` to preserve
immutability guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

# ── Ground truth & expectations ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Expected correct answer for an evaluation case.

    Attributes:
        reference_output: The canonical correct output.
        acceptable_alternatives: Additional outputs considered correct.
    """

    reference_output: str
    acceptable_alternatives: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCallExpectation:
    """Expected tool invocation for a case.

    Attributes:
        tool_name: Name of the tool that should be called.
        expected_args: Partial dict of expected argument key/values.
            Only the specified keys are checked; extra args are allowed
            unless *allow_extra_args* is ``False``.
        allow_extra_args: Whether the actual call may contain arguments
            not listed in *expected_args*.
    """

    tool_name: str
    expected_args: dict[str, Any] = field(default_factory=dict)
    allow_extra_args: bool = True


@dataclass(frozen=True, slots=True)
class RetrievalExpectation:
    """Expected retrieval results for a case.

    Attributes:
        expected_ids: Document/chunk IDs that should appear in results.
        k: The cut-off rank to evaluate at.
    """

    expected_ids: tuple[str, ...]
    k: int = 5


# ── Evaluation case & dataset ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvalCase:
    """A single evaluation test case.

    Attributes:
        case_id: Unique identifier for this case.
        input: The prompt or query to evaluate.
        ground_truth: Expected correct output, if applicable.
        expected_tool_calls: Expected tool invocations, if applicable.
        expected_retrieval: Expected retrieval results, if applicable.
        metadata: Arbitrary key/value metadata for grouping or filtering.
    """

    case_id: str
    input: str = ""
    ground_truth: GroundTruth | None = None
    expected_tool_calls: tuple[ToolCallExpectation, ...] = ()
    expected_retrieval: RetrievalExpectation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A named collection of evaluation cases.

    Attributes:
        name: Human-readable dataset name.
        cases: Ordered tuple of evaluation cases.
        metadata: Arbitrary dataset-level metadata.
    """

    name: str
    cases: tuple[EvalCase, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of cases in the dataset."""
        return len(self.cases)


# ── Scoring results ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvalMetric:
    """A single named metric with a value and optional pass/fail threshold.

    Attributes:
        name: Metric name (e.g. ``"exact_match"``, ``"hit_at_5"``).
        value: Numeric metric value.
        threshold: Optional minimum value for pass.  ``None`` means no
            threshold is applied.
    """

    name: str
    value: float
    threshold: float | None = None

    @property
    def passed(self) -> bool:
        """Whether the metric meets its threshold (or has no threshold)."""
        if self.threshold is None:
            return True
        return self.value >= self.threshold


@dataclass(frozen=True, slots=True)
class EvalScore:
    """Score for a specific case from a specific scorer.

    Attributes:
        case_id: The evaluation case that was scored.
        scorer_name: Name of the scorer that produced this score.
        metric: The metric result.
    """

    case_id: str
    scorer_name: str
    metric: EvalMetric


@dataclass(frozen=True, slots=True)
class EvalFailure:
    """Record of a failure for a specific case.

    Attributes:
        case_id: The evaluation case that failed.
        reason: Short description of the failure.
        details: Extended diagnostic information.
    """

    case_id: str
    reason: str
    details: str = ""


# ── Per-case result ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Full evaluation result for a single case.

    Attributes:
        case_id: The evaluation case identifier.
        actual_output: The model's actual output for this case.
        scores: All scores from all scorers for this case.
        passed: Whether all scored metrics meet their thresholds.
        failures: Any failures recorded for this case.
        metadata: Arbitrary result metadata.
    """

    case_id: str
    actual_output: str = ""
    scores: tuple[EvalScore, ...] = ()
    passed: bool = True
    failures: tuple[EvalFailure, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Artifacts & summaries ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvalArtifact:
    """An artifact produced by or for an evaluation run.

    Attributes:
        name: Artifact file name or key.
        format: MIME type or short format tag (e.g. ``"json"``, ``"md"``).
        content: Artifact content as a string.
    """

    name: str
    format: str = "text"
    content: str = ""


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Aggregate summary of an evaluation run.

    Attributes:
        dataset_name: Name of the evaluated dataset.
        total: Total number of cases.
        passed: Number of cases that passed.
        failed: Number of cases that failed.
        metrics: Aggregate metrics across all cases.
        results: Per-case results.
    """

    dataset_name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    metrics: tuple[EvalMetric, ...] = ()
    results: tuple[EvalResult, ...] = ()

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed (0.0–1.0)."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


@dataclass(frozen=True, slots=True)
class EvalRun:
    """A complete evaluation run with results and artifacts.

    Attributes:
        run_id: Unique run identifier.
        timestamp: UTC timestamp of the run.
        dataset_name: Name of the dataset that was evaluated.
        summary: Aggregate summary.
        artifacts: Artifacts produced during the run.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    dataset_name: str = ""
    summary: EvalSummary = field(
        default_factory=lambda: EvalSummary(dataset_name=""),
    )
    artifacts: tuple[EvalArtifact, ...] = ()


# ── Regression comparison ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RegressionDelta:
    """Change in a single metric between baseline and current run.

    Attributes:
        metric_name: Name of the metric.
        baseline_value: Value in the baseline run.
        current_value: Value in the current run.
        threshold: Minimum acceptable delta (negative means tolerance
            for regression).  A delta below this threshold is flagged.
        regressed: Whether this metric regressed beyond the threshold.
    """

    metric_name: str
    baseline_value: float
    current_value: float
    threshold: float = 0.0

    @property
    def delta(self) -> float:
        """Signed change: current − baseline."""
        return self.current_value - self.baseline_value

    @property
    def regressed(self) -> bool:
        """Whether the metric regressed beyond the threshold."""
        return self.delta < -abs(self.threshold)


@dataclass(frozen=True, slots=True)
class RegressionComparison:
    """Summary of regression analysis between two runs.

    Attributes:
        baseline_run_id: Run ID of the baseline.
        current_run_id: Run ID of the current run.
        deltas: Per-metric deltas.
    """

    baseline_run_id: str
    current_run_id: str
    deltas: tuple[RegressionDelta, ...] = ()

    @property
    def has_regressions(self) -> bool:
        """Whether any metric regressed beyond its threshold."""
        return any(d.regressed for d in self.deltas)

    @property
    def regressions(self) -> tuple[RegressionDelta, ...]:
        """Only the deltas that regressed."""
        return tuple(d for d in self.deltas if d.regressed)

    @property
    def improvements(self) -> tuple[RegressionDelta, ...]:
        """Only the deltas that improved."""
        return tuple(d for d in self.deltas if d.delta > 0)
