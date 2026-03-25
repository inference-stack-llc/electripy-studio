"""Ports for pluggable policy gateway detectors and sanitizers."""

from __future__ import annotations

from typing import Protocol

from .domain import PolicyFinding, PolicyInput, PolicyRule


class PolicyDetectorPort(Protocol):
    """Protocol for policy detectors."""

    def detect(self, policy_input: PolicyInput, rules: list[PolicyRule]) -> list[PolicyFinding]:
        """Return policy findings for the given input and rule set."""


class TextSanitizerPort(Protocol):
    """Protocol for text sanitizers that redact matched findings."""

    def sanitize(
        self,
        *,
        text: str,
        findings: list[PolicyFinding],
        rules_by_id: dict[str, PolicyRule],
    ) -> str:
        """Return sanitized text for the supplied findings."""
