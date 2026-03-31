"""Evaluation runner and orchestration services.

The :class:`EvalRunner` is the primary orchestrator.  It takes a
dataset (or pre-computed outputs), applies scorers, aggregates
results, and optionally compares against a baseline for regression
detection.

Example::

    from electripy.ai.evals import EvalRunner, ExactMatchScorer, EvalDataset

    runner = EvalRunner(scorers=[ExactMatchScorer()])
    run = runner.run_dataset(dataset, outputs={"q1": "Paris"})
    print(run.summary.pass_rate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from electripy.core.logging import get_logger

from .domain import (
    EvalCase,
    EvalDataset,
    EvalFailure,
    EvalMetric,
    EvalResult,
    EvalRun,
    EvalScore,
    EvalSummary,
    RegressionComparison,
    RegressionDelta,
)
from .errors import RegressionError
from .ports import ArtifactStorePort, ModelInvocationPort

__all__ = [
    "EvalRunner",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class EvalRunner:
    """Orchestrates evaluation runs.

    The runner scores model outputs against evaluation cases using
    one or more scorers, then aggregates results into a summary.

    Attributes:
        scorers: Scorers to apply to each case.
        model: Optional model port for invoking a model during the run.
            If not provided, outputs must be supplied via the
            ``outputs`` parameter.
        report_writers: Optional report writers invoked after each run.
        artifact_store: Optional artifact store for persisting artifacts.
    """

    scorers: list[Any] = field(default_factory=list)  # list[ScorerPort]
    model: ModelInvocationPort | None = None
    report_writers: list[Any] = field(default_factory=list)  # list[ReportWriterPort]
    artifact_store: ArtifactStorePort | None = None

    # ── Primary API ──────────────────────────────────────────────────

    def run_dataset(
        self,
        dataset: EvalDataset,
        *,
        outputs: dict[str, str] | None = None,
        scorer_kwargs: dict[str, Any] | None = None,
    ) -> EvalRun:
        """Run evaluation across all cases in a dataset.

        For each case, the runner either uses a pre-computed output
        from *outputs* or invokes the model port.  Then all scorers
        are applied and results are aggregated.

        Args:
            dataset: The evaluation dataset.
            outputs: Mapping of case_id → actual model output.  If
                not provided for a case, the model port is invoked.
            scorer_kwargs: Extra keyword arguments passed to every
                scorer (e.g. ``retrieved_ids``, ``tool_calls``).

        Returns:
            A complete evaluation run with summary and artifacts.
        """
        outputs = outputs or {}
        scorer_kwargs = scorer_kwargs or {}
        results: list[EvalResult] = []

        for case in dataset.cases:
            actual = self._get_output(case, outputs)
            case_kwargs = dict(scorer_kwargs)
            # Allow per-case kwargs from outputs dict if it carries a dict
            result = self.score_case(case, actual, **case_kwargs)
            results.append(result)

        summary = self._build_summary(dataset.name, results)
        run = EvalRun(dataset_name=dataset.name, summary=summary)

        self._run_report_writers(summary)

        logger.info(
            "Eval run %s complete: %d/%d passed (%.1f%%)",
            run.run_id,
            summary.passed,
            summary.total,
            summary.pass_rate * 100,
        )
        return run

    def score_case(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> EvalResult:
        """Score a single case against all configured scorers.

        Args:
            case: The evaluation case.
            actual_output: The model's actual output.
            **kwargs: Additional context for scorers.

        Returns:
            The evaluation result for this case.
        """
        all_scores: list[EvalScore] = []
        failures: list[EvalFailure] = []

        for scorer in self.scorers:
            try:
                scores = scorer.score(case, actual_output, **kwargs)
                all_scores.extend(scores)
            except Exception as exc:
                failures.append(
                    EvalFailure(
                        case_id=case.case_id,
                        reason=f"Scorer '{scorer.name}' raised an error",
                        details=str(exc),
                    )
                )

        passed = all(s.metric.passed for s in all_scores) and not failures
        return EvalResult(
            case_id=case.case_id,
            actual_output=actual_output,
            scores=tuple(all_scores),
            passed=passed,
            failures=tuple(failures),
        )

    def compare_runs(
        self,
        baseline: EvalSummary,
        current: EvalSummary,
        *,
        thresholds: dict[str, float] | None = None,
        fail_on_regression: bool = False,
    ) -> RegressionComparison:
        """Compare current run metrics against a baseline.

        Matches metrics by name and computes deltas.  If a metric
        exists in the current run but not the baseline, it is skipped.

        Args:
            baseline: The baseline run summary.
            current: The current run summary.
            thresholds: Per-metric regression thresholds.  A delta
                below ``-threshold`` is flagged as a regression.
                Defaults to 0.0 (any decrease is a regression).
            fail_on_regression: If True, raise :class:`RegressionError`
                when any regression is detected.

        Returns:
            A regression comparison summary.

        Raises:
            RegressionError: If regressions are found and
                *fail_on_regression* is True.
        """
        thresholds = thresholds or {}
        baseline_metrics = {m.name: m.value for m in baseline.metrics}
        deltas: list[RegressionDelta] = []

        for metric in current.metrics:
            if metric.name not in baseline_metrics:
                continue
            threshold = thresholds.get(metric.name, 0.0)
            deltas.append(
                RegressionDelta(
                    metric_name=metric.name,
                    baseline_value=baseline_metrics[metric.name],
                    current_value=metric.value,
                    threshold=threshold,
                )
            )

        comparison = RegressionComparison(
            baseline_run_id="baseline",
            current_run_id="current",
            deltas=tuple(deltas),
        )

        if fail_on_regression and comparison.has_regressions:
            names = ", ".join(d.metric_name for d in comparison.regressions)
            raise RegressionError(f"Regression detected in metric(s): {names}")

        return comparison

    # ── Convenience methods ──────────────────────────────────────────

    def run_with_model(
        self,
        dataset: EvalDataset,
        **scorer_kwargs: Any,
    ) -> EvalRun:
        """Run evaluation by invoking the model port for each case.

        Equivalent to ``run_dataset(dataset, scorer_kwargs=...)``,
        explicitly requiring the model port.

        Args:
            dataset: The evaluation dataset.
            **scorer_kwargs: Extra kwargs for scorers.

        Returns:
            A complete evaluation run.

        Raises:
            EvalError: If no model port is configured.
        """
        return self.run_dataset(dataset, scorer_kwargs=scorer_kwargs)

    # ── Internal ─────────────────────────────────────────────────────

    def _get_output(
        self,
        case: EvalCase,
        outputs: dict[str, str],
    ) -> str:
        """Get the actual output for a case."""
        if case.case_id in outputs:
            return outputs[case.case_id]
        if self.model is not None:
            return self.model.invoke(case.input, case.metadata or None)
        return ""

    def _build_summary(
        self,
        dataset_name: str,
        results: list[EvalResult],
    ) -> EvalSummary:
        """Build an aggregate summary from per-case results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Aggregate metrics: average each unique metric name
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        metric_thresholds: dict[str, float | None] = {}

        for result in results:
            for score in result.scores:
                name = score.metric.name
                metric_sums[name] = metric_sums.get(name, 0.0) + score.metric.value
                metric_counts[name] = metric_counts.get(name, 0) + 1
                if name not in metric_thresholds:
                    metric_thresholds[name] = score.metric.threshold

        aggregate_metrics: list[EvalMetric] = []
        for name in sorted(metric_sums):
            avg = metric_sums[name] / metric_counts[name]
            aggregate_metrics.append(
                EvalMetric(
                    name=name,
                    value=avg,
                    threshold=metric_thresholds.get(name),
                )
            )

        return EvalSummary(
            dataset_name=dataset_name,
            total=total,
            passed=passed,
            failed=failed,
            metrics=tuple(aggregate_metrics),
            results=tuple(results),
        )

    def _run_report_writers(self, summary: EvalSummary) -> None:
        """Invoke all configured report writers."""
        for writer in self.report_writers:
            try:
                writer.write(summary, f"{summary.dataset_name}_report")
            except Exception:
                logger.exception("Report writer failed")
