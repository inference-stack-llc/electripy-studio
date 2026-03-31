"""Redaction subsystem for the observe package.

This module provides :class:`DefaultRedactor`, the standard
implementation of :class:`~electripy.observability.observe.ports.RedactorPort`.
It applies a :class:`~electripy.observability.observe.domain.RedactionPolicy`
to attribute mappings, ensuring sensitive values never leave the
application boundary.

The redactor supports three rule kinds:

- **Exact**: attribute keys matching a known-sensitive name.
- **Pattern**: attribute keys matching a regular expression.
- **Callable**: a user-supplied predicate receiving ``(key, value)``.

Usage::

    from electripy.observability.observe import (
        DefaultRedactor,
        RedactionPolicy,
    )

    redactor = DefaultRedactor(policy=RedactionPolicy.enterprise_default())
    clean = redactor.redact({"prompt": "secret input", "model": "gpt-4o"})
    assert clean["prompt"] == "[REDACTED]"
    assert clean["model"] == "gpt-4o"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from electripy.core.logging import get_logger

from .domain import (
    Attributes,
    AttributeValue,
    RedactionPolicy,
    RedactionRule,
    RedactionRuleKind,
)
from .errors import RedactionError

logger = get_logger(__name__)


@dataclass(slots=True)
class DefaultRedactor:
    """Production-ready redactor implementing :class:`RedactorPort`.

    The redactor applies each rule in the policy to every attribute.
    The first matching rule wins and replaces the attribute's value with
    the rule's ``replacement`` text.

    Attributes:
        policy: The redaction policy to enforce.
    """

    policy: RedactionPolicy = field(default_factory=RedactionPolicy.enterprise_default)

    # Compiled regex cache keyed by pattern string.
    _pattern_cache: dict[str, re.Pattern[str]] = field(default_factory=dict, repr=False)

    def redact(self, attributes: Attributes) -> Attributes:
        """Return a redacted copy of *attributes*.

        Each attribute key/value pair is tested against the policy's
        rules in order.  The first matching rule replaces the value
        with its ``replacement`` string.  Attributes that do not match
        any rule are copied unchanged.

        Args:
            attributes: Raw attribute mapping.

        Returns:
            A new mapping with sensitive values replaced.
        """
        if not self.policy.enabled:
            return dict(attributes)

        result: Attributes = {}
        for key, value in attributes.items():
            replacement = self._match(key, value)
            result[key] = replacement if replacement is not None else value
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match(self, key: str, value: AttributeValue) -> str | None:
        """Return the replacement string if any rule matches, else ``None``."""
        for rule in self.policy.rules:
            if self._rule_matches(rule, key, value):
                return rule.replacement or self.policy.default_replacement
        return None

    def _rule_matches(self, rule: RedactionRule, key: str, value: AttributeValue) -> bool:
        """Test whether *rule* matches the given key/value pair."""
        if rule.kind == RedactionRuleKind.EXACT:
            return key.lower() == rule.match.lower()

        if rule.kind == RedactionRuleKind.PATTERN:
            pattern = self._get_compiled_pattern(rule.match)
            return pattern.search(key) is not None

        if rule.kind == RedactionRuleKind.CALLABLE:
            if rule.predicate is None:
                return False
            try:
                return rule.predicate(key, value)
            except Exception:
                logger.warning(
                    "Redaction predicate raised for key=%s; treating as match",
                    key,
                )
                return True

        return False

    def _get_compiled_pattern(self, pattern: str) -> re.Pattern[str]:
        """Return a compiled regex, caching for reuse."""
        compiled = self._pattern_cache.get(pattern)
        if compiled is None:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise RedactionError(f"Invalid redaction regex pattern: {pattern!r}") from exc
            self._pattern_cache[pattern] = compiled
        return compiled


__all__ = ["DefaultRedactor"]
