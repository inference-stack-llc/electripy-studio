"""Tests for electripy.ai.evals.services."""

from __future__ import annotations

import pytest

from electripy.ai.evals.domain import (
    EvalCase,
    EvalDataset,
    EvalMetric,
    EvalSummary,
    GroundTruth,
    RetrievalExpectation,
    ToolCallExpectation,
)
from electripy.ai.evals.errors import RegressionError
from electripy.ai.evals.scorers import (
    CompositeScorer,
    ContainsScorer,
    ExactMatchScorer,
    NormalizedTextScorer,
    RetrievalScorer,
    ThresholdScorer,
    ToolCallScorer,
)
from electripy.ai.evals.services import EvalRunner


# ── Fixtures ─────────────────────────────────────────────────────────


def _dataset() -> EvalDataset:
    return EvalDataset(
        name="capitals",
        cases=(
            EvalCase(
                case_id="q1",
                input="Capital of France?",
                ground_truth=GroundTruth(reference_output="Paris"),
            ),
            EvalCase(
                case_id="q2",
                input="Capital of Germany?",
                ground_truth=GroundTruth(reference_output="Berlin"),
            ),
            EvalCase(
                case_id="q3",
                input="Capital of Japan?",
                ground_truth=GroundTruth(reference_output="Tokyo"),
            ),
        ),
    )


# ── Basic run ─────────────────────────────────────────────────────


class TestBasicRun:
    def test_all_correct(self) -> None:
        runner = EvalRunner(scorers=[ExactMatchScorer()])
        run = runner.run_dataset(
            _dataset(),
            outputs={"q1": "Paris", "q2": "Berlin", "q3": "Tokyo"},
        )
        assert run.summary.total == 3
        assert run.summary.passed == 3
        assert run.summary.pass_rate == 1.0

    def test_partial_correct(self) -> None:
        scorer = ThresholdScorer(
            inner=ExactMatchScorer(),
            thresholds={"exact_match": 1.0},
        )
        runner = EvalRunner(scorers=[scorer])
        run = runner.run_dataset(
            _dataset(),
            outputs={"q1": "Paris", "q2": "Munich", "q3": "Osaka"},
        )
        assert run.summary.total == 3
        assert run.summary.passed == 1
        assert run.summary.failed == 2

    def test_missing_outputs_empty_string(self) -> None:
        scorer = ThresholdScorer(
            inner=ExactMatchScorer(),
            thresholds={"exact_match": 1.0},
        )
        runner = EvalRunner(scorers=[scorer])
        run = runner.run_dataset(_dataset(), outputs={"q1": "Paris"})
        # q2 and q3 have no output and no model → empty string
        assert run.summary.passed == 1

    def test_run_has_id_and_timestamp(self) -> None:
        runner = EvalRunner(scorers=[ExactMatchScorer()])
        run = runner.run_dataset(
            _dataset(), outputs={"q1": "Paris", "q2": "Berlin", "q3": "Tokyo"},
        )
        assert len(run.run_id) == 12
        assert "T" in run.timestamp

    def test_dataset_name_propagated(self) -> None:
        runner = EvalRunner(scorers=[ExactMatchScorer()])
        run = runner.run_dataset(
            _dataset(), outputs={"q1": "Paris", "q2": "Berlin", "q3": "Tokyo"},
        )
        assert run.dataset_name == "capitals"
        assert run.summary.dataset_name == "capitals"


# ── Model invocation ─────────────────────────────────────────────


class TestModelInvocation:
    def test_uses_model_port(self) -> None:
        from electripy.ai.evals.adapters import CallbackModelInvocation

        model = CallbackModelInvocation(
            callback=lambda text, **kw: {
                "Capital of France?": "Paris",
                "Capital of Germany?": "Berlin",
                "Capital of Japan?": "Tokyo",
            }.get(text, "unknown"),
        )
        runner = EvalRunner(
            scorers=[ExactMatchScorer()],
            model=model,
        )
        run = runner.run_dataset(_dataset())
        assert run.summary.passed == 3

    def test_outputs_override_model(self) -> None:
        from electripy.ai.evals.adapters import CallbackModelInvocation

        model = CallbackModelInvocation(callback=lambda text, **kw: "WRONG")
        runner = EvalRunner(
            scorers=[ExactMatchScorer()],
            model=model,
        )
        run = runner.run_dataset(
            _dataset(),
            outputs={"q1": "Paris", "q2": "Berlin", "q3": "Tokyo"},
        )
        assert run.summary.passed == 3


# ── Score case ───────────────────────────────────────────────────


class TestScoreCase:
    def test_single_case(self) -> None:
        runner = EvalRunner(scorers=[ExactMatchScorer()])
        case = EvalCase(
            case_id="q1",
            ground_truth=GroundTruth(reference_output="Paris"),
        )
        result = runner.score_case(case, "Paris")
        assert result.passed is True
        assert result.scores[0].metric.value == 1.0

    def test_scorer_error_captured(self) -> None:
        class BadScorer:
            @property
            def name(self) -> str:
                return "bad"

            def score(self, case, actual, **kw):
                raise ValueError("oops")

        runner = EvalRunner(scorers=[BadScorer()])
        case = EvalCase(case_id="q1")
        result = runner.score_case(case, "anything")
        assert result.passed is False
        assert len(result.failures) == 1
        assert "oops" in result.failures[0].details


# ── Multiple scorers ─────────────────────────────────────────────


class TestMultipleScorers:
    def test_all_scores_collected(self) -> None:
        runner = EvalRunner(
            scorers=[ExactMatchScorer(), NormalizedTextScorer()],
        )
        case = EvalCase(
            case_id="q1",
            ground_truth=GroundTruth(reference_output="Paris"),
        )
        result = runner.score_case(case, "Paris")
        assert len(result.scores) == 2

    def test_composite_scorer(self) -> None:
        scorer = CompositeScorer(
            scorers=(ExactMatchScorer(), NormalizedTextScorer()),
        )
        runner = EvalRunner(scorers=[scorer])
        case = EvalCase(
            case_id="q1",
            ground_truth=GroundTruth(reference_output="Paris"),
        )
        result = runner.score_case(case, "Paris")
        assert len(result.scores) == 2


# ── Threshold pass/fail ──────────────────────────────────────────


class TestThresholdPassFail:
    def test_pass(self) -> None:
        scorer = ThresholdScorer(
            inner=ExactMatchScorer(),
            thresholds={"exact_match": 1.0},
        )
        runner = EvalRunner(scorers=[scorer])
        case = EvalCase(
            case_id="q1",
            ground_truth=GroundTruth(reference_output="Paris"),
        )
        result = runner.score_case(case, "Paris")
        assert result.passed is True

    def test_fail(self) -> None:
        scorer = ThresholdScorer(
            inner=ExactMatchScorer(),
            thresholds={"exact_match": 1.0},
        )
        runner = EvalRunner(scorers=[scorer])
        case = EvalCase(
            case_id="q1",
            ground_truth=GroundTruth(reference_output="Paris"),
        )
        result = runner.score_case(case, "London")
        assert result.passed is False


# ── Retrieval eval ───────────────────────────────────────────────


class TestRetrievalEval:
    def test_retrieval_scoring(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_retrieval=RetrievalExpectation(
                expected_ids=("doc-1", "doc-2"), k=3,
            ),
        )
        runner = EvalRunner(scorers=[RetrievalScorer()])
        result = runner.score_case(
            case, "",
            retrieved_ids=["doc-1", "doc-3", "doc-2"],
        )
        hit = next(
            s for s in result.scores if s.metric.name == "hit_at_k"
        )
        recall = next(
            s for s in result.scores if s.metric.name == "recall_at_k"
        )
        assert hit.metric.value == 1.0
        # expected_ids=("doc-1", "doc-2"), both in top-3 → recall = 2/2 = 1.0
        assert recall.metric.value == 1.0


# ── Tool-call eval ───────────────────────────────────────────────


class TestToolCallEval:
    def test_tool_call_scoring(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_tool_calls=(
                ToolCallExpectation(
                    tool_name="search",
                    expected_args={"query": "weather"},
                ),
            ),
        )
        runner = EvalRunner(scorers=[ToolCallScorer()])
        result = runner.score_case(
            case, "",
            tool_calls=[
                {"name": "search", "arguments": {"query": "weather"}},
            ],
        )
        assert result.passed is True
        name_score = next(
            s for s in result.scores if s.metric.name == "tool_name_match"
        )
        assert name_score.metric.value == 1.0


# ── Aggregate metrics ────────────────────────────────────────────


class TestAggregateMetrics:
    def test_averages_metrics(self) -> None:
        runner = EvalRunner(scorers=[ExactMatchScorer()])
        run = runner.run_dataset(
            _dataset(),
            outputs={"q1": "Paris", "q2": "Munich", "q3": "Tokyo"},
        )
        metrics = {m.name: m.value for m in run.summary.metrics}
        # 2/3 correct → avg exact_match = 2/3
        assert abs(metrics["exact_match"] - 2 / 3) < 0.001


# ── Regression comparison ────────────────────────────────────────


class TestRegressionComparison:
    def test_no_regression(self) -> None:
        runner = EvalRunner()
        baseline = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="accuracy", value=0.8),),
        )
        current = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="accuracy", value=0.85),),
        )
        comp = runner.compare_runs(baseline, current)
        assert comp.has_regressions is False
        assert len(comp.improvements) == 1

    def test_regression_detected(self) -> None:
        runner = EvalRunner()
        baseline = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="accuracy", value=0.9),),
        )
        current = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="accuracy", value=0.7),),
        )
        comp = runner.compare_runs(baseline, current)
        assert comp.has_regressions is True
        assert comp.regressions[0].metric_name == "accuracy"

    def test_within_threshold(self) -> None:
        runner = EvalRunner()
        baseline = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="accuracy", value=0.9),),
        )
        current = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="accuracy", value=0.88),),
        )
        comp = runner.compare_runs(
            baseline, current, thresholds={"accuracy": 0.05},
        )
        assert comp.has_regressions is False

    def test_fail_on_regression(self) -> None:
        runner = EvalRunner()
        baseline = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="accuracy", value=0.9),),
        )
        current = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="accuracy", value=0.7),),
        )
        with pytest.raises(RegressionError, match="accuracy"):
            runner.compare_runs(
                baseline, current, fail_on_regression=True,
            )

    def test_missing_baseline_metric_skipped(self) -> None:
        runner = EvalRunner()
        baseline = EvalSummary(dataset_name="test", metrics=())
        current = EvalSummary(
            dataset_name="test",
            metrics=(EvalMetric(name="new_metric", value=0.5),),
        )
        comp = runner.compare_runs(baseline, current)
        assert len(comp.deltas) == 0

    def test_multiple_metrics(self) -> None:
        runner = EvalRunner()
        baseline = EvalSummary(
            dataset_name="test",
            metrics=(
                EvalMetric(name="accuracy", value=0.9),
                EvalMetric(name="recall", value=0.8),
            ),
        )
        current = EvalSummary(
            dataset_name="test",
            metrics=(
                EvalMetric(name="accuracy", value=0.7),
                EvalMetric(name="recall", value=0.85),
            ),
        )
        comp = runner.compare_runs(baseline, current)
        assert len(comp.deltas) == 2
        assert len(comp.regressions) == 1
        assert len(comp.improvements) == 1


# ── Determinism ──────────────────────────────────────────────────


class TestDeterminism:
    def test_stable_outputs(self) -> None:
        runner = EvalRunner(scorers=[ExactMatchScorer()])
        outputs = {"q1": "Paris", "q2": "Berlin", "q3": "Tokyo"}
        run1 = runner.run_dataset(_dataset(), outputs=outputs)
        run2 = runner.run_dataset(_dataset(), outputs=outputs)
        assert run1.summary.passed == run2.summary.passed
        assert run1.summary.pass_rate == run2.summary.pass_rate
        for r1, r2 in zip(
            run1.summary.results, run2.summary.results, strict=True,
        ):
            assert r1.passed == r2.passed
            for s1, s2 in zip(r1.scores, r2.scores, strict=True):
                assert s1.metric.value == s2.metric.value


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_dataset(self) -> None:
        runner = EvalRunner(scorers=[ExactMatchScorer()])
        ds = EvalDataset(name="empty")
        run = runner.run_dataset(ds, outputs={})
        assert run.summary.total == 0
        assert run.summary.pass_rate == 0.0

    def test_no_scorers(self) -> None:
        runner = EvalRunner()
        run = runner.run_dataset(
            _dataset(),
            outputs={"q1": "Paris", "q2": "Berlin", "q3": "Tokyo"},
        )
        assert run.summary.total == 3
        assert run.summary.passed == 3  # No scorers → all pass

    def test_case_without_ground_truth(self) -> None:
        scorer = ThresholdScorer(
            inner=ExactMatchScorer(),
            thresholds={"exact_match": 1.0},
        )
        runner = EvalRunner(scorers=[scorer])
        ds = EvalDataset(
            name="no_gt",
            cases=(EvalCase(case_id="q1", input="Hello"),),
        )
        run = runner.run_dataset(ds, outputs={"q1": "Hi"})
        # ExactMatchScorer returns 0.0, threshold 1.0 → fails
        assert run.summary.passed == 0


# ── Contains scorer in pipeline ──────────────────────────────────


class TestContainsInPipeline:
    def test_contains_eval(self) -> None:
        scorer = ContainsScorer(substrings=("Paris", "France"))
        runner = EvalRunner(scorers=[scorer])
        case = EvalCase(case_id="q1")
        result = runner.score_case(case, "Paris is the capital of France")
        assert result.scores[0].metric.value == 1.0
