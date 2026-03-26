"""Ports for pluggable policy gateway detectors and sanitizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .domain import PolicyFinding, PolicyInput, PolicyRule

__all__ = ["PolicyDetectorPort", "TextSanitizerPort"]


class PolicyDetectorPort(Protocol):
    """Protocol for policy detectors."""

    def detect(self, policy_input: PolicyInput, rules: Sequence[PolicyRule]) -> list[PolicyFinding]:
        """Return policy findings for the given input and rule set."""


class TextSanitizerPort(Protocol):
    """Protocol for text sanitizers that redact matched findings."""

    def sanitize(
        self,
        *,
        text: str,
        findings: Sequence[PolicyFinding],
        rules_by_id: Mapping[str, PolicyRule],
    ) -> str:
        """Return sanitized text for the supplied findings."""
