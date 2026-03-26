"""Eval Assertions — pytest-native assertion helpers for LLM outputs.

Purpose:
  - Provide declarative assertion functions for validating LLM responses
    in unit and regression tests.
  - Support semantic similarity, JSON schema conformance, regex matching,
    keyword presence, and custom predicates.
  - Produce rich, structured failure reports for CI diagnostics.

Guarantees:
  - No network calls — all checks are local and deterministic.
  - Assertions raise standard ``AssertionError`` for pytest compatibility.
  - All domain models are frozen and immutable.

Usage::

    from electripy.ai.eval_assertions import (
        assert_llm_output,
        contains_keywords,
        matches_regex,
        matches_json_schema,
        passes_predicate,
    )

    assert_llm_output(
        output="The capital of France is Paris.",
        checks=[
            contains_keywords(["Paris", "capital"]),
            matches_regex(r"Paris"),
        ],
    )
"""

from __future__ import annotations

from .domain import AssertionCheck, AssertionResult, AssertionSeverity
from .services import (
    assert_llm_output,
    contains_keywords,
    matches_json_schema,
    matches_regex,
    passes_predicate,
    satisfies_length,
)

__all__ = [
    # Domain models
    "AssertionCheck",
    "AssertionResult",
    "AssertionSeverity",
    # Services
    "assert_llm_output",
    "contains_keywords",
    "matches_json_schema",
    "matches_regex",
    "passes_predicate",
    "satisfies_length",
]
