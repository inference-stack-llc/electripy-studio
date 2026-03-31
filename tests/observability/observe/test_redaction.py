"""Tests for the observe package redaction subsystem."""

from __future__ import annotations

from electripy.observability.observe.domain import (
    RedactionPolicy,
    RedactionRule,
    RedactionRuleKind,
)
from electripy.observability.observe.redaction import DefaultRedactor


class TestDefaultRedactor:
    """DefaultRedactor applying various rule kinds."""

    def test_exact_match_redacts_key(self) -> None:
        """An exact rule redacts a matching key."""
        policy = RedactionPolicy(
            rules=(
                RedactionRule(kind=RedactionRuleKind.EXACT, match="secret"),
            ),
        )
        redactor = DefaultRedactor(policy=policy)

        result = redactor.redact({"secret": "s3cr3t", "model": "gpt-4o"})

        assert result["secret"] == "[REDACTED]"
        assert result["model"] == "gpt-4o"

    def test_exact_match_is_case_insensitive(self) -> None:
        """Exact matching ignores case."""
        policy = RedactionPolicy(
            rules=(
                RedactionRule(kind=RedactionRuleKind.EXACT, match="API_KEY"),
            ),
        )
        redactor = DefaultRedactor(policy=policy)

        result = redactor.redact({"api_key": "k-123"})
        assert result["api_key"] == "[REDACTED]"

    def test_pattern_match_redacts_matching_keys(self) -> None:
        """A regex pattern rule matches attribute keys."""
        policy = RedactionPolicy(
            rules=(
                RedactionRule(
                    kind=RedactionRuleKind.PATTERN,
                    match=r"^auth",
                    replacement="***",
                ),
            ),
        )
        redactor = DefaultRedactor(policy=policy)

        result = redactor.redact({
            "authorization": "Bearer tok",
            "auth_header": "Basic cred",
            "model": "gpt-4o",
        })

        assert result["authorization"] == "***"
        assert result["auth_header"] == "***"
        assert result["model"] == "gpt-4o"

    def test_callable_rule_uses_predicate(self) -> None:
        """A callable rule invokes the predicate with (key, value)."""
        policy = RedactionPolicy(
            rules=(
                RedactionRule(
                    kind=RedactionRuleKind.CALLABLE,
                    predicate=lambda k, v: isinstance(v, str) and "secret" in str(v).lower(),
                ),
            ),
        )
        redactor = DefaultRedactor(policy=policy)

        result = redactor.redact({"data": "my secret data", "count": 42})

        assert result["data"] == "[REDACTED]"
        assert result["count"] == 42

    def test_callable_predicate_exception_treated_as_match(self) -> None:
        """A predicate that raises is conservatively treated as a match."""
        def _bad_predicate(k: str, v: object) -> bool:
            raise RuntimeError("boom")

        policy = RedactionPolicy(
            rules=(
                RedactionRule(
                    kind=RedactionRuleKind.CALLABLE,
                    predicate=_bad_predicate,
                ),
            ),
        )
        redactor = DefaultRedactor(policy=policy)

        result = redactor.redact({"anything": "value"})
        assert result["anything"] == "[REDACTED]"

    def test_first_matching_rule_wins(self) -> None:
        """When multiple rules match, the first one applies."""
        policy = RedactionPolicy(
            rules=(
                RedactionRule(
                    kind=RedactionRuleKind.EXACT,
                    match="key",
                    replacement="[FIRST]",
                ),
                RedactionRule(
                    kind=RedactionRuleKind.PATTERN,
                    match=r"key",
                    replacement="[SECOND]",
                ),
            ),
        )
        redactor = DefaultRedactor(policy=policy)

        result = redactor.redact({"key": "value"})
        assert result["key"] == "[FIRST]"

    def test_disabled_policy_passes_through(self) -> None:
        """A disabled policy returns attributes unchanged."""
        policy = RedactionPolicy(
            rules=(
                RedactionRule(kind=RedactionRuleKind.EXACT, match="secret"),
            ),
            enabled=False,
        )
        redactor = DefaultRedactor(policy=policy)

        result = redactor.redact({"secret": "s3cr3t"})
        assert result["secret"] == "s3cr3t"

    def test_enterprise_default_redacts_common_keys(self) -> None:
        """The enterprise default policy redacts known sensitive keys."""
        redactor = DefaultRedactor()

        result = redactor.redact({
            "prompt": "Tell me a joke",
            "completion": "Why did the chicken...",
            "api_key": "sk-abc",
            "password": "hunter2",
            "model": "gpt-4o",
            "latency_ms": 42.0,
        })

        assert result["prompt"] == "[REDACTED]"
        assert result["completion"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["model"] == "gpt-4o"
        assert result["latency_ms"] == 42.0

    def test_original_dict_is_not_mutated(self) -> None:
        """Redaction returns a copy; the original mapping is untouched."""
        original = {"secret": "value", "safe": "ok"}
        redactor = DefaultRedactor(
            policy=RedactionPolicy(
                rules=(
                    RedactionRule(kind=RedactionRuleKind.EXACT, match="secret"),
                ),
            )
        )

        result = redactor.redact(original)

        assert result["secret"] == "[REDACTED]"
        assert original["secret"] == "value"

    def test_custom_replacement_text(self) -> None:
        """Rules can specify custom replacement text."""
        policy = RedactionPolicy(
            rules=(
                RedactionRule(
                    kind=RedactionRuleKind.EXACT,
                    match="ssn",
                    replacement="XXX-XX-XXXX",
                ),
            ),
        )
        redactor = DefaultRedactor(policy=policy)

        result = redactor.redact({"ssn": "123-45-6789"})
        assert result["ssn"] == "XXX-XX-XXXX"

    def test_empty_attributes_returns_empty(self) -> None:
        """An empty input produces an empty output."""
        redactor = DefaultRedactor()
        assert redactor.redact({}) == {}
