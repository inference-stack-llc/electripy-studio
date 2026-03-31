"""Built-in scorers for the evaluation framework.

Each scorer implements :class:`~electripy.ai.evals.ports.ScorerPort`
and is stateless and deterministic.  Scorers produce
:class:`~electripy.ai.evals.domain.EvalScore` instances containing
named metrics with optional pass/fail thresholds.

Example::

    from electripy.ai.evals.scorers import ExactMatchScorer

    scorer = ExactMatchScorer()
    scores = scorer.score(case, "Paris")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from electripy.ai.rag_quality import (
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
)

from .domain import EvalCase, EvalMetric, EvalScore

__all__ = [
    "CompositeScorer",
    "ContainsScorer",
    "ExactMatchScorer",
    "JsonStructureScorer",
    "NormalizedTextScorer",
    "RetrievalScorer",
    "ThresholdScorer",
    "ToolCallScorer",
]


# ── Text scorers ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExactMatchScorer:
    """Scores 1.0 if actual output exactly matches the reference output.

    Compares against ``ground_truth.reference_output`` and all
    ``acceptable_alternatives``.  If no ground truth is set on the
    case, the score is 0.0.
    """

    @property
    def name(self) -> str:
        return "exact_match"

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        if case.ground_truth is None:
            value = 0.0
        else:
            expected = {case.ground_truth.reference_output}
            expected.update(case.ground_truth.acceptable_alternatives)
            value = 1.0 if actual_output in expected else 0.0

        return [
            EvalScore(
                case_id=case.case_id,
                scorer_name=self.name,
                metric=EvalMetric(name="exact_match", value=value),
            ),
        ]


@dataclass(frozen=True, slots=True)
class NormalizedTextScorer:
    """Scores 1.0 if normalized actual output matches reference.

    Normalization: strip whitespace, lowercase, collapse internal
    whitespace to single spaces.
    """

    @property
    def name(self) -> str:
        return "normalized_match"

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        if case.ground_truth is None:
            value = 0.0
        else:
            norm = _normalize(actual_output)
            expected = {_normalize(case.ground_truth.reference_output)}
            expected.update(_normalize(a) for a in case.ground_truth.acceptable_alternatives)
            value = 1.0 if norm in expected else 0.0

        return [
            EvalScore(
                case_id=case.case_id,
                scorer_name=self.name,
                metric=EvalMetric(name="normalized_match", value=value),
            ),
        ]


def _normalize(text: str) -> str:
    """Strip, lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass(frozen=True, slots=True)
class ContainsScorer:
    """Scores 1.0 if all required substrings appear in the output.

    Attributes:
        substrings: Substrings that must all be present.
        case_sensitive: Whether matching is case-sensitive.
    """

    substrings: tuple[str, ...] = ()
    case_sensitive: bool = False

    @property
    def name(self) -> str:
        return "contains"

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        text = actual_output if self.case_sensitive else actual_output.lower()
        targets = (
            self.substrings if self.case_sensitive else tuple(s.lower() for s in self.substrings)
        )
        hits = sum(1 for s in targets if s in text) if targets else 0
        total = len(targets) if targets else 1
        value = hits / total

        return [
            EvalScore(
                case_id=case.case_id,
                scorer_name=self.name,
                metric=EvalMetric(name="contains", value=value),
            ),
        ]


# ── Structured output scorer ─────────────────────────────────────────


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True, slots=True)
class JsonStructureScorer:
    """Scores JSON structure conformance.

    Checks that the output is valid JSON with required fields present
    and correct types.  Uses a lightweight schema format:

    - ``required``: list of field names that must exist.
    - ``properties``: dict of field-name → ``{"type": "<json_type>"}``
      for type checking.

    Attributes:
        schema: The JSON schema to validate against.
    """

    schema: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "json_structure"

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        try:
            obj = json.loads(actual_output)
        except (json.JSONDecodeError, ValueError):
            return [self._make_score(case, 0.0)]

        if not isinstance(obj, dict):
            return [self._make_score(case, 0.0)]

        checks_total = 0
        checks_passed = 0

        for f in self.schema.get("required", []):
            checks_total += 1
            if f in obj:
                checks_passed += 1

        for f, spec in self.schema.get("properties", {}).items():
            if f in obj:
                expected_type = _TYPE_MAP.get(spec.get("type", ""))
                if expected_type:
                    checks_total += 1
                    if isinstance(obj[f], expected_type):
                        checks_passed += 1

        if checks_total == 0:
            value = 1.0  # No checks → vacuously valid
        else:
            value = checks_passed / checks_total

        return [self._make_score(case, value)]

    def _make_score(self, case: EvalCase, value: float) -> EvalScore:
        return EvalScore(
            case_id=case.case_id,
            scorer_name=self.name,
            metric=EvalMetric(name="json_structure", value=value),
        )


# ── Retrieval scorer ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RetrievalScorer:
    """Scores retrieval quality using hit@k, recall@k, and MRR@k.

    Expects the case to have an ``expected_retrieval`` with
    ``expected_ids`` and ``k``.  The actual retrieved IDs are passed
    via kwargs as ``retrieved_ids``.

    Delegates computation to :mod:`electripy.ai.rag_quality`.
    """

    @property
    def name(self) -> str:
        return "retrieval"

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        expectation = case.expected_retrieval
        if expectation is None:
            return []

        retrieved: list[str] = kwargs.get("retrieved_ids", [])
        expected = list(expectation.expected_ids)
        k = expectation.k

        hit = hit_rate_at_k(retrieved, expected, k)
        recall = recall_at_k(retrieved, expected, k)
        mrr = mrr_at_k(retrieved, expected, k)

        return [
            EvalScore(
                case_id=case.case_id,
                scorer_name=self.name,
                metric=EvalMetric(name="hit_at_k", value=hit),
            ),
            EvalScore(
                case_id=case.case_id,
                scorer_name=self.name,
                metric=EvalMetric(name="recall_at_k", value=recall),
            ),
            EvalScore(
                case_id=case.case_id,
                scorer_name=self.name,
                metric=EvalMetric(name="mrr_at_k", value=mrr),
            ),
        ]


# ── Tool-call scorer ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ToolCallScorer:
    """Scores tool-call correctness.

    Checks that expected tools were invoked with correct names and
    (partial) argument matches.  Actual tool calls are passed via
    kwargs as ``tool_calls``: a list of dicts with ``"name"`` and
    ``"arguments"`` keys.
    """

    @property
    def name(self) -> str:
        return "tool_call"

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        if not case.expected_tool_calls:
            return []

        actual_calls: list[dict[str, Any]] = kwargs.get("tool_calls", [])
        scores: list[EvalScore] = []

        for i, expected in enumerate(case.expected_tool_calls):
            name_match = 0.0
            arg_match = 0.0
            suffix = f"_{i}" if len(case.expected_tool_calls) > 1 else ""

            # Find matching actual call by name
            matching_call = next(
                (c for c in actual_calls if c.get("name") == expected.tool_name),
                None,
            )

            if matching_call is not None:
                name_match = 1.0
                actual_args = matching_call.get("arguments", {})

                if not expected.expected_args:
                    arg_match = 1.0
                else:
                    matched = 0
                    total = len(expected.expected_args)
                    for key, val in expected.expected_args.items():
                        if key in actual_args and actual_args[key] == val:
                            matched += 1
                    arg_match = matched / total if total > 0 else 1.0

                    if not expected.allow_extra_args:
                        extra_keys = set(actual_args) - set(expected.expected_args)
                        if extra_keys:
                            arg_match *= 0.5  # Penalize unexpected args

            scores.extend(
                [
                    EvalScore(
                        case_id=case.case_id,
                        scorer_name=self.name,
                        metric=EvalMetric(
                            name=f"tool_name_match{suffix}",
                            value=name_match,
                        ),
                    ),
                    EvalScore(
                        case_id=case.case_id,
                        scorer_name=self.name,
                        metric=EvalMetric(
                            name=f"tool_arg_match{suffix}",
                            value=arg_match,
                        ),
                    ),
                ]
            )

        return scores


# ── Threshold & composite scorers ────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ThresholdScorer:
    """Wraps another scorer and applies a pass/fail threshold.

    Attributes:
        inner: The scorer to delegate to.
        thresholds: Mapping of metric name → minimum required value.
    """

    inner: Any  # ScorerPort — using Any to keep frozen
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"threshold({self.inner.name})"

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        inner_scores = self.inner.score(case, actual_output, **kwargs)
        result: list[EvalScore] = []
        for s in inner_scores:
            threshold = self.thresholds.get(s.metric.name)
            if threshold is not None:
                result.append(
                    EvalScore(
                        case_id=s.case_id,
                        scorer_name=self.name,
                        metric=EvalMetric(
                            name=s.metric.name,
                            value=s.metric.value,
                            threshold=threshold,
                        ),
                    )
                )
            else:
                result.append(s)
        return result


@dataclass(frozen=True, slots=True)
class CompositeScorer:
    """Runs multiple scorers and aggregates their scores.

    Attributes:
        scorers: Tuple of scorers to run.
    """

    scorers: tuple[Any, ...] = ()  # tuple[ScorerPort, ...]

    @property
    def name(self) -> str:
        return "composite"

    def score(
        self,
        case: EvalCase,
        actual_output: str,
        **kwargs: Any,
    ) -> list[EvalScore]:
        result: list[EvalScore] = []
        for scorer in self.scorers:
            result.extend(scorer.score(case, actual_output, **kwargs))
        return result
