"""Tests for electripy.ai.evals.domain."""

from __future__ import annotations

import pytest

from electripy.ai.evals.domain import (
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


class TestGroundTruth:
    def test_basic(self) -> None:
        gt = GroundTruth(reference_output="Paris")
        assert gt.reference_output == "Paris"
        assert gt.acceptable_alternatives == ()

    def test_with_alternatives(self) -> None:
        gt = GroundTruth(
            reference_output="Paris",
            acceptable_alternatives=("paris", "PARIS"),
        )
        assert len(gt.acceptable_alternatives) == 2


class TestToolCallExpectation:
    def test_defaults(self) -> None:
        tc = ToolCallExpectation(tool_name="search")
        assert tc.tool_name == "search"
        assert tc.expected_args == {}
        assert tc.allow_extra_args is True

    def test_with_args(self) -> None:
        tc = ToolCallExpectation(
            tool_name="search",
            expected_args={"query": "weather"},
            allow_extra_args=False,
        )
        assert tc.expected_args["query"] == "weather"
        assert tc.allow_extra_args is False


class TestRetrievalExpectation:
    def test_defaults(self) -> None:
        re_ = RetrievalExpectation(expected_ids=("doc-1", "doc-2"))
        assert re_.k == 5

    def test_custom_k(self) -> None:
        re_ = RetrievalExpectation(expected_ids=("doc-1",), k=10)
        assert re_.k == 10


class TestEvalCase:
    def test_minimal(self) -> None:
        case = EvalCase(case_id="q1")
        assert case.case_id == "q1"
        assert case.input == ""
        assert case.ground_truth is None
        assert case.expected_tool_calls == ()
        assert case.expected_retrieval is None

    def test_full(self) -> None:
        case = EvalCase(
            case_id="q1",
            input="What is the capital of France?",
            ground_truth=GroundTruth(reference_output="Paris"),
            expected_tool_calls=(ToolCallExpectation(tool_name="lookup"),),
            expected_retrieval=RetrievalExpectation(
                expected_ids=("doc-1",),
                k=3,
            ),
            metadata={"category": "geography"},
        )
        assert case.ground_truth is not None
        assert len(case.expected_tool_calls) == 1
        assert case.metadata["category"] == "geography"


class TestEvalDataset:
    def test_empty(self) -> None:
        ds = EvalDataset(name="empty")
        assert ds.size == 0

    def test_with_cases(self) -> None:
        ds = EvalDataset(
            name="test",
            cases=(
                EvalCase(case_id="q1"),
                EvalCase(case_id="q2"),
            ),
        )
        assert ds.size == 2


class TestEvalMetric:
    def test_no_threshold(self) -> None:
        m = EvalMetric(name="accuracy", value=0.85)
        assert m.passed is True

    def test_above_threshold(self) -> None:
        m = EvalMetric(name="accuracy", value=0.85, threshold=0.8)
        assert m.passed is True

    def test_below_threshold(self) -> None:
        m = EvalMetric(name="accuracy", value=0.7, threshold=0.8)
        assert m.passed is False

    def test_equal_threshold(self) -> None:
        m = EvalMetric(name="accuracy", value=0.8, threshold=0.8)
        assert m.passed is True


class TestEvalScore:
    def test_basic(self) -> None:
        s = EvalScore(
            case_id="q1",
            scorer_name="exact_match",
            metric=EvalMetric(name="exact_match", value=1.0),
        )
        assert s.case_id == "q1"
        assert s.metric.value == 1.0


class TestEvalFailure:
    def test_basic(self) -> None:
        f = EvalFailure(case_id="q1", reason="Scorer error")
        assert f.details == ""


class TestEvalResult:
    def test_defaults(self) -> None:
        r = EvalResult(case_id="q1")
        assert r.passed is True
        assert r.scores == ()
        assert r.actual_output == ""

    def test_failed(self) -> None:
        r = EvalResult(case_id="q1", passed=False)
        assert r.passed is False


class TestEvalSummary:
    def test_pass_rate_zero(self) -> None:
        s = EvalSummary(dataset_name="test", total=0)
        assert s.pass_rate == 0.0

    def test_pass_rate(self) -> None:
        s = EvalSummary(dataset_name="test", total=10, passed=7, failed=3)
        assert s.pass_rate == 0.7

    def test_perfect(self) -> None:
        s = EvalSummary(dataset_name="test", total=5, passed=5, failed=0)
        assert s.pass_rate == 1.0


class TestEvalRun:
    def test_has_run_id(self) -> None:
        run = EvalRun()
        assert len(run.run_id) == 12

    def test_has_timestamp(self) -> None:
        run = EvalRun()
        assert "T" in run.timestamp  # ISO format


class TestEvalArtifact:
    def test_basic(self) -> None:
        a = EvalArtifact(name="report.json", format="json", content="{}")
        assert a.format == "json"


class TestRegressionDelta:
    def test_no_regression(self) -> None:
        d = RegressionDelta(
            metric_name="accuracy",
            baseline_value=0.8,
            current_value=0.85,
        )
        assert d.delta == pytest.approx(0.05)
        assert d.regressed is False

    def test_regression(self) -> None:
        d = RegressionDelta(
            metric_name="accuracy",
            baseline_value=0.9,
            current_value=0.7,
        )
        assert d.delta == pytest.approx(-0.2)
        assert d.regressed is True

    def test_within_threshold(self) -> None:
        d = RegressionDelta(
            metric_name="accuracy",
            baseline_value=0.9,
            current_value=0.88,
            threshold=0.05,
        )
        assert d.regressed is False

    def test_beyond_threshold(self) -> None:
        d = RegressionDelta(
            metric_name="accuracy",
            baseline_value=0.9,
            current_value=0.8,
            threshold=0.05,
        )
        assert d.regressed is True


class TestRegressionComparison:
    def test_no_regressions(self) -> None:
        c = RegressionComparison(
            baseline_run_id="b1",
            current_run_id="c1",
            deltas=(RegressionDelta("acc", 0.8, 0.85),),
        )
        assert c.has_regressions is False
        assert len(c.regressions) == 0
        assert len(c.improvements) == 1

    def test_with_regressions(self) -> None:
        c = RegressionComparison(
            baseline_run_id="b1",
            current_run_id="c1",
            deltas=(
                RegressionDelta("acc", 0.9, 0.7),
                RegressionDelta("recall", 0.8, 0.85),
            ),
        )
        assert c.has_regressions is True
        assert len(c.regressions) == 1
        assert c.regressions[0].metric_name == "acc"
        assert len(c.improvements) == 1
