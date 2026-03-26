"""Tests for Eval Assertions."""

from __future__ import annotations

import pytest

from electripy.ai.eval_assertions import (
    AssertionCheck,
    AssertionResult,
    AssertionSeverity,
    assert_llm_output,
    contains_keywords,
    matches_json_schema,
    matches_regex,
    passes_predicate,
    satisfies_length,
)

# ---------------------------------------------------------------------------
# contains_keywords
# ---------------------------------------------------------------------------


class TestContainsKeywords:
    def test_all_present(self) -> None:
        check = contains_keywords(["Paris", "capital"])
        assert check.check_fn("The capital of France is Paris.") is True

    def test_missing_keyword(self) -> None:
        check = contains_keywords(["Paris", "Berlin"])
        assert check.check_fn("The capital of France is Paris.") is False

    def test_case_insensitive_by_default(self) -> None:
        check = contains_keywords(["paris"])
        assert check.check_fn("PARIS is great") is True

    def test_case_sensitive(self) -> None:
        check = contains_keywords(["paris"], case_sensitive=True)
        assert check.check_fn("PARIS is great") is False


# ---------------------------------------------------------------------------
# matches_regex
# ---------------------------------------------------------------------------


class TestMatchesRegex:
    def test_match_found(self) -> None:
        check = matches_regex(r"\d{4}-\d{2}-\d{2}")
        assert check.check_fn("Date: 2026-03-25") is True

    def test_no_match(self) -> None:
        check = matches_regex(r"\d{4}-\d{2}-\d{2}")
        assert check.check_fn("no date here") is False

    def test_complex_pattern(self) -> None:
        check = matches_regex(r"^(positive|negative|neutral)$")
        assert check.check_fn("positive") is True
        assert check.check_fn("maybe positive") is False


# ---------------------------------------------------------------------------
# matches_json_schema
# ---------------------------------------------------------------------------


class TestMatchesJsonSchema:
    def test_valid_json(self) -> None:
        schema = {
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        check = matches_json_schema(schema)
        assert check.check_fn('{"name": "Alice", "age": 30}') is True

    def test_missing_required_field(self) -> None:
        schema = {"required": ["name", "age"], "properties": {}}
        check = matches_json_schema(schema)
        assert check.check_fn('{"name": "Alice"}') is False

    def test_wrong_type(self) -> None:
        schema = {
            "required": [],
            "properties": {"age": {"type": "integer"}},
        }
        check = matches_json_schema(schema)
        assert check.check_fn('{"age": "thirty"}') is False

    def test_invalid_json(self) -> None:
        schema = {"required": [], "properties": {}}
        check = matches_json_schema(schema)
        assert check.check_fn("not json at all") is False

    def test_non_object_json(self) -> None:
        schema = {"required": [], "properties": {}}
        check = matches_json_schema(schema)
        assert check.check_fn("[1, 2, 3]") is False


# ---------------------------------------------------------------------------
# passes_predicate
# ---------------------------------------------------------------------------


class TestPassesPredicate:
    def test_predicate_passes(self) -> None:
        check = passes_predicate(lambda x: len(x) > 5, name="length_check")
        assert check.check_fn("hello world") is True

    def test_predicate_fails(self) -> None:
        check = passes_predicate(lambda x: len(x) > 100)
        assert check.check_fn("short") is False

    def test_custom_name(self) -> None:
        check = passes_predicate(lambda x: True, name="always_pass")
        assert check.name == "always_pass"


# ---------------------------------------------------------------------------
# satisfies_length
# ---------------------------------------------------------------------------


class TestSatisfiesLength:
    def test_within_bounds(self) -> None:
        check = satisfies_length(min_length=5, max_length=20)
        assert check.check_fn("hello world") is True

    def test_too_short(self) -> None:
        check = satisfies_length(min_length=100)
        assert check.check_fn("short") is False

    def test_too_long(self) -> None:
        check = satisfies_length(max_length=3)
        assert check.check_fn("toolong") is False

    def test_no_max(self) -> None:
        check = satisfies_length(min_length=1)
        assert check.check_fn("any length") is True

    def test_empty_string_min_zero(self) -> None:
        check = satisfies_length(min_length=0)
        assert check.check_fn("") is True


# ---------------------------------------------------------------------------
# assert_llm_output — integration
# ---------------------------------------------------------------------------


class TestAssertLlmOutput:
    def test_all_pass(self) -> None:
        results = assert_llm_output(
            output="The capital of France is Paris.",
            checks=[
                contains_keywords(["Paris", "France"]),
                matches_regex(r"Paris"),
                satisfies_length(min_length=10),
            ],
        )
        assert all(r.passed for r in results)
        assert len(results) == 3

    def test_error_raises_assertion(self) -> None:
        with pytest.raises(AssertionError, match="contains_keywords"):
            assert_llm_output(
                output="Hello world",
                checks=[contains_keywords(["nonexistent"])],
            )

    def test_warning_does_not_raise_by_default(self) -> None:
        results = assert_llm_output(
            output="short",
            checks=[
                satisfies_length(
                    min_length=1000,
                    severity=AssertionSeverity.WARNING,
                ),
            ],
        )
        assert len(results) == 1
        assert results[0].passed is False

    def test_warning_raises_when_configured(self) -> None:
        with pytest.raises(AssertionError, match="WARNING"):
            assert_llm_output(
                output="short",
                checks=[
                    satisfies_length(
                        min_length=1000,
                        severity=AssertionSeverity.WARNING,
                    ),
                ],
                raise_on_warning=True,
            )

    def test_multiple_failures_reported(self) -> None:
        with pytest.raises(AssertionError, match="2 check") as exc_info:
            assert_llm_output(
                output="nope",
                checks=[
                    contains_keywords(["missing"]),
                    matches_regex(r"^\d+$"),
                ],
            )
        assert "contains_keywords" in str(exc_info.value)
        assert "matches_regex" in str(exc_info.value)

    def test_output_snippet_in_error(self) -> None:
        with pytest.raises(AssertionError, match="first 200 chars"):
            assert_llm_output(
                output="bad output",
                checks=[contains_keywords(["good"])],
            )


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------


class TestDomainModels:
    def test_assertion_result_defaults(self) -> None:
        result = AssertionResult(check_name="test", passed=True, severity=AssertionSeverity.ERROR)
        assert result.message == ""

    def test_assertion_check_default_severity(self) -> None:
        check = AssertionCheck(name="t", check_fn=lambda x: True)
        assert check.severity == AssertionSeverity.ERROR
