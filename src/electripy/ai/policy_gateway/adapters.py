"""Adapter implementations for policy gateway ports."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .domain import PolicyFinding, PolicyInput, PolicyRule
from .ports import PolicyDetectorPort, TextSanitizerPort

__all__ = ["RedactionSanitizerAdapter", "RegexPolicyDetectorAdapter"]


@dataclass(slots=True)
class RegexPolicyDetectorAdapter(PolicyDetectorPort):
    """Regex-based policy detector.

    This adapter is deterministic and does not require network calls.
    """

    case_insensitive: bool = True

    def detect(self, policy_input: PolicyInput, rules: Sequence[PolicyRule]) -> list[PolicyFinding]:
        """Detect rule matches for the given text input."""

        flags = re.IGNORECASE if self.case_insensitive else 0
        findings: list[PolicyFinding] = []
        for rule in rules:
            if rule.stage != policy_input.stage:
                continue
            for match in re.finditer(rule.pattern, policy_input.text, flags):
                findings.append(
                    PolicyFinding(
                        rule_id=rule.rule_id,
                        code=rule.code,
                        message=rule.description,
                        severity=rule.severity,
                        action=rule.action,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        findings.sort(key=lambda item: (item.start or -1, item.rule_id, item.code))
        return findings


@dataclass(slots=True)
class RedactionSanitizerAdapter(TextSanitizerPort):
    """Simple text sanitizer that replaces matched spans."""

    default_replacement: str = "[REDACTED]"

    def sanitize(
        self,
        *,
        text: str,
        findings: Sequence[PolicyFinding],
        rules_by_id: Mapping[str, PolicyRule],
    ) -> str:
        """Sanitize text by replacing matched spans from right to left."""

        spans: list[tuple[int, int, str]] = []
        for finding in findings:
            if finding.start is None or finding.end is None:
                continue
            rule = rules_by_id.get(finding.rule_id)
            replacement = rule.replacement if rule is not None else self.default_replacement
            spans.append((finding.start, finding.end, replacement))
        if not spans:
            return text

        # Replace from the end to avoid index shifting.
        sanitized = text
        for start, end, replacement in sorted(spans, key=lambda s: (s[0], s[1]), reverse=True):
            repl = replacement if replacement is not None else self.default_replacement
            sanitized = sanitized[:start] + repl + sanitized[end:]
        return sanitized
