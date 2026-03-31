"""Tests for electripy.ai.evals.scorers."""

from __future__ import annotations

import json

from electripy.ai.evals.domain import (
    EvalCase,
    GroundTruth,
    RetrievalExpectation,
    ToolCallExpectation,
)
from electripy.ai.evals.scorers import (
    CompositeScorer,
    ContainsScorer,
    ExactMatchScorer,
    JsonStructureScorer,
    NormalizedTextScorer,
    RetrievalScorer,
    ThresholdScorer,
    ToolCallScorer,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _case(
    reference: str = "Paris",
    alternatives: tuple[str, ...] = (),
) -> EvalCase:
    return EvalCase(
        case_id="q1",
        input="What is the capital of France?",
        ground_truth=GroundTruth(
            reference_output=reference,
            acceptable_alternatives=alternatives,
        ),
    )


def _case_no_gt() -> EvalCase:
    return EvalCase(case_id="q1", input="Hello")


# ── ExactMatchScorer ─────────────────────────────────────────────────


class TestExactMatchScorer:
    def test_match(self) -> None:
        scores = ExactMatchScorer().score(_case(), "Paris")
        assert scores[0].metric.value == 1.0

    def test_no_match(self) -> None:
        scores = ExactMatchScorer().score(_case(), "London")
        assert scores[0].metric.value == 0.0

    def test_alternative_match(self) -> None:
        case = _case(alternatives=("paris", "PARIS"))
        scores = ExactMatchScorer().score(case, "paris")
        assert scores[0].metric.value == 1.0

    def test_no_ground_truth(self) -> None:
        scores = ExactMatchScorer().score(_case_no_gt(), "anything")
        assert scores[0].metric.value == 0.0

    def test_scorer_name(self) -> None:
        assert ExactMatchScorer().name == "exact_match"


# ── NormalizedTextScorer ──────────────────────────────────────────────


class TestNormalizedTextScorer:
    def test_exact_after_normalization(self) -> None:
        scores = NormalizedTextScorer().score(_case(), "  Paris  ")
        assert scores[0].metric.value == 1.0

    def test_case_insensitive(self) -> None:
        scores = NormalizedTextScorer().score(_case(), "PARIS")
        assert scores[0].metric.value == 1.0

    def test_whitespace_collapse(self) -> None:
        case = _case(reference="New York")
        scores = NormalizedTextScorer().score(case, "  new   york  ")
        assert scores[0].metric.value == 1.0

    def test_no_match(self) -> None:
        scores = NormalizedTextScorer().score(_case(), "London")
        assert scores[0].metric.value == 0.0

    def test_no_ground_truth(self) -> None:
        scores = NormalizedTextScorer().score(_case_no_gt(), "anything")
        assert scores[0].metric.value == 0.0


# ── ContainsScorer ────────────────────────────────────────────────────


class TestContainsScorer:
    def test_all_present(self) -> None:
        scorer = ContainsScorer(substrings=("capital", "France"))
        scores = scorer.score(_case(), "The capital of France is Paris")
        assert scores[0].metric.value == 1.0

    def test_partial(self) -> None:
        scorer = ContainsScorer(substrings=("capital", "Germany"))
        scores = scorer.score(_case(), "The capital of France is Paris")
        assert scores[0].metric.value == 0.5

    def test_none_present(self) -> None:
        scorer = ContainsScorer(substrings=("Berlin", "Germany"))
        scores = scorer.score(_case(), "Paris is the capital of France")
        assert scores[0].metric.value == 0.0

    def test_case_insensitive_default(self) -> None:
        scorer = ContainsScorer(substrings=("PARIS",))
        scores = scorer.score(_case(), "paris is great")
        assert scores[0].metric.value == 1.0

    def test_case_sensitive(self) -> None:
        scorer = ContainsScorer(substrings=("PARIS",), case_sensitive=True)
        scores = scorer.score(_case(), "paris is great")
        assert scores[0].metric.value == 0.0

    def test_empty_substrings(self) -> None:
        scorer = ContainsScorer()
        scores = scorer.score(_case(), "anything")
        assert scores[0].metric.value == 0.0


# ── JsonStructureScorer ──────────────────────────────────────────────


class TestJsonStructureScorer:
    def test_valid_json(self) -> None:
        schema = {
            "required": ["title", "score"],
            "properties": {
                "title": {"type": "string"},
                "score": {"type": "number"},
            },
        }
        scorer = JsonStructureScorer(schema=schema)
        output = json.dumps({"title": "Test", "score": 42})
        scores = scorer.score(_case(), output)
        assert scores[0].metric.value == 1.0

    def test_missing_field(self) -> None:
        schema = {"required": ["title", "score"]}
        scorer = JsonStructureScorer(schema=schema)
        output = json.dumps({"title": "Test"})
        scores = scorer.score(_case(), output)
        assert scores[0].metric.value == 0.5

    def test_wrong_type(self) -> None:
        schema = {
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
        }
        scorer = JsonStructureScorer(schema=schema)
        output = json.dumps({"value": "not_an_int"})
        scores = scorer.score(_case(), output)
        # required check passes (1/1), type check fails (0/1) → 0.5
        assert scores[0].metric.value == 0.5

    def test_invalid_json(self) -> None:
        scorer = JsonStructureScorer(schema={"required": ["x"]})
        scores = scorer.score(_case(), "not json {{{")
        assert scores[0].metric.value == 0.0

    def test_not_object(self) -> None:
        scorer = JsonStructureScorer(schema={"required": ["x"]})
        scores = scorer.score(_case(), json.dumps([1, 2, 3]))
        assert scores[0].metric.value == 0.0

    def test_empty_schema(self) -> None:
        scorer = JsonStructureScorer()
        scores = scorer.score(_case(), json.dumps({"any": "thing"}))
        assert scores[0].metric.value == 1.0


# ── RetrievalScorer ──────────────────────────────────────────────────


class TestRetrievalScorer:
    def test_hit_at_k(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_retrieval=RetrievalExpectation(
                expected_ids=("doc-1",),
                k=3,
            ),
        )
        scores = RetrievalScorer().score(
            case,
            "",
            retrieved_ids=["doc-2", "doc-1", "doc-3"],
        )
        hit = next(s for s in scores if s.metric.name == "hit_at_k")
        assert hit.metric.value == 1.0

    def test_miss_at_k(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_retrieval=RetrievalExpectation(
                expected_ids=("doc-1",),
                k=2,
            ),
        )
        scores = RetrievalScorer().score(
            case,
            "",
            retrieved_ids=["doc-2", "doc-3"],
        )
        hit = next(s for s in scores if s.metric.name == "hit_at_k")
        assert hit.metric.value == 0.0

    def test_recall_at_k(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_retrieval=RetrievalExpectation(
                expected_ids=("doc-1", "doc-2"),
                k=3,
            ),
        )
        scores = RetrievalScorer().score(
            case,
            "",
            retrieved_ids=["doc-1", "doc-3", "doc-4"],
        )
        recall = next(s for s in scores if s.metric.name == "recall_at_k")
        assert recall.metric.value == 0.5

    def test_mrr_at_k(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_retrieval=RetrievalExpectation(
                expected_ids=("doc-1",),
                k=5,
            ),
        )
        scores = RetrievalScorer().score(
            case,
            "",
            retrieved_ids=["x", "x", "doc-1", "x", "x"],
        )
        mrr = next(s for s in scores if s.metric.name == "mrr_at_k")
        assert mrr.metric.value == 1.0 / 3.0

    def test_no_expectation(self) -> None:
        case = EvalCase(case_id="q1")
        scores = RetrievalScorer().score(case, "")
        assert scores == []

    def test_produces_three_metrics(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_retrieval=RetrievalExpectation(
                expected_ids=("doc-1",),
                k=3,
            ),
        )
        scores = RetrievalScorer().score(
            case,
            "",
            retrieved_ids=["doc-1"],
        )
        assert len(scores) == 3
        names = {s.metric.name for s in scores}
        assert names == {"hit_at_k", "recall_at_k", "mrr_at_k"}


# ── ToolCallScorer ───────────────────────────────────────────────────


class TestToolCallScorer:
    def test_name_match(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_tool_calls=(ToolCallExpectation(tool_name="search"),),
        )
        scores = ToolCallScorer().score(
            case,
            "",
            tool_calls=[{"name": "search", "arguments": {}}],
        )
        name_score = next(s for s in scores if s.metric.name == "tool_name_match")
        assert name_score.metric.value == 1.0

    def test_name_mismatch(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_tool_calls=(ToolCallExpectation(tool_name="search"),),
        )
        scores = ToolCallScorer().score(
            case,
            "",
            tool_calls=[{"name": "browse", "arguments": {}}],
        )
        name_score = next(s for s in scores if s.metric.name == "tool_name_match")
        assert name_score.metric.value == 0.0

    def test_arg_match(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_tool_calls=(
                ToolCallExpectation(
                    tool_name="search",
                    expected_args={"query": "weather"},
                ),
            ),
        )
        scores = ToolCallScorer().score(
            case,
            "",
            tool_calls=[{"name": "search", "arguments": {"query": "weather"}}],
        )
        arg_score = next(s for s in scores if s.metric.name == "tool_arg_match")
        assert arg_score.metric.value == 1.0

    def test_partial_arg_match(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_tool_calls=(
                ToolCallExpectation(
                    tool_name="search",
                    expected_args={"query": "weather", "location": "NYC"},
                ),
            ),
        )
        scores = ToolCallScorer().score(
            case,
            "",
            tool_calls=[
                {"name": "search", "arguments": {"query": "weather"}},
            ],
        )
        arg_score = next(s for s in scores if s.metric.name == "tool_arg_match")
        assert arg_score.metric.value == 0.5

    def test_no_extra_args_penalty(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_tool_calls=(
                ToolCallExpectation(
                    tool_name="search",
                    expected_args={"query": "weather"},
                    allow_extra_args=False,
                ),
            ),
        )
        scores = ToolCallScorer().score(
            case,
            "",
            tool_calls=[
                {
                    "name": "search",
                    "arguments": {"query": "weather", "extra": "val"},
                },
            ],
        )
        arg_score = next(s for s in scores if s.metric.name == "tool_arg_match")
        assert arg_score.metric.value == 0.5  # 1.0 * 0.5 penalty

    def test_no_expected_tool_calls(self) -> None:
        case = EvalCase(case_id="q1")
        scores = ToolCallScorer().score(case, "")
        assert scores == []

    def test_no_actual_calls(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_tool_calls=(ToolCallExpectation(tool_name="search"),),
        )
        scores = ToolCallScorer().score(case, "", tool_calls=[])
        name_score = next(s for s in scores if s.metric.name == "tool_name_match")
        assert name_score.metric.value == 0.0

    def test_multiple_expected(self) -> None:
        case = EvalCase(
            case_id="q1",
            expected_tool_calls=(
                ToolCallExpectation(tool_name="search"),
                ToolCallExpectation(tool_name="browse"),
            ),
        )
        scores = ToolCallScorer().score(
            case,
            "",
            tool_calls=[
                {"name": "search", "arguments": {}},
                {"name": "browse", "arguments": {}},
            ],
        )
        assert len(scores) == 4  # 2 name + 2 arg
        name_scores = [s for s in scores if "tool_name_match" in s.metric.name]
        assert all(s.metric.value == 1.0 for s in name_scores)


# ── ThresholdScorer ──────────────────────────────────────────────────


class TestThresholdScorer:
    def test_applies_threshold(self) -> None:
        inner = ExactMatchScorer()
        scorer = ThresholdScorer(
            inner=inner,
            thresholds={"exact_match": 1.0},
        )
        scores = scorer.score(_case(), "Paris")
        assert scores[0].metric.threshold == 1.0
        assert scores[0].metric.passed is True

    def test_fails_threshold(self) -> None:
        inner = ExactMatchScorer()
        scorer = ThresholdScorer(
            inner=inner,
            thresholds={"exact_match": 1.0},
        )
        scores = scorer.score(_case(), "London")
        assert scores[0].metric.passed is False

    def test_no_threshold_for_metric(self) -> None:
        inner = ExactMatchScorer()
        scorer = ThresholdScorer(inner=inner, thresholds={})
        scores = scorer.score(_case(), "Paris")
        assert scores[0].metric.threshold is None

    def test_name(self) -> None:
        scorer = ThresholdScorer(inner=ExactMatchScorer())
        assert scorer.name == "threshold(exact_match)"


# ── CompositeScorer ──────────────────────────────────────────────────


class TestCompositeScorer:
    def test_combines_scorers(self) -> None:
        scorer = CompositeScorer(
            scorers=(ExactMatchScorer(), NormalizedTextScorer()),
        )
        scores = scorer.score(_case(), "Paris")
        assert len(scores) == 2
        assert scores[0].scorer_name == "exact_match"
        assert scores[1].scorer_name == "normalized_match"

    def test_name(self) -> None:
        assert CompositeScorer().name == "composite"

    def test_empty(self) -> None:
        scores = CompositeScorer().score(_case(), "Paris")
        assert scores == []
