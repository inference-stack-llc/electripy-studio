"""Sensitive Data Scanner — detect PII and secrets before they hit an LLM.

Purpose:
  - Scan outbound text for common PII patterns (email, phone, SSN, etc.)
    and secret patterns (API keys, tokens, passwords).
  - Return structured findings so callers can redact, block, or log.
  - Designed for pre-flight checks in LLM pipelines.

Guarantees:
  - Pure regex — no network, no ML, deterministic results.
  - Extensible with custom patterns via ``add_pattern()``.
  - Thread-safe and stateless per-call.

Usage::

    from electripy.ai.sensitive_data_scanner import scan_text, SensitiveMatch

    matches = scan_text("Contact me at alice@corp.io or call 555-123-4567")
    for m in matches:
        print(m.category, m.matched_text)
    # "email" "alice@corp.io"
    # "phone_us" "555-123-4567"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "SensitiveDataScanner",
    "SensitiveMatch",
    "SensitivePattern",
    "scan_text",
]


@dataclass(frozen=True, slots=True)
class SensitiveMatch:
    """A single sensitive data finding."""

    category: str
    matched_text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SensitivePattern:
    """A named regex pattern for sensitive data detection."""

    category: str
    pattern: re.Pattern[str]
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS: list[SensitivePattern] = [
    SensitivePattern(
        category="email",
        pattern=re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        description="Email address",
    ),
    SensitivePattern(
        category="phone_us",
        pattern=re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        description="US phone number",
    ),
    SensitivePattern(
        category="ssn",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        description="US Social Security Number",
    ),
    SensitivePattern(
        category="credit_card",
        pattern=re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        description="Credit/debit card number",
    ),
    SensitivePattern(
        category="api_key_generic",
        pattern=re.compile(
            r"\b(?:sk|pk|api|key|token|secret|password)[-_]?[a-zA-Z0-9]{20,}\b",
            re.IGNORECASE,
        ),
        description="Generic API key / token pattern",
    ),
    SensitivePattern(
        category="api_key_openai",
        pattern=re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),
        description="OpenAI API key",
    ),
    SensitivePattern(
        category="api_key_anthropic",
        pattern=re.compile(r"\bsk-ant-[a-zA-Z0-9]{20,}\b"),
        description="Anthropic API key",
    ),
    SensitivePattern(
        category="aws_access_key",
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        description="AWS access key ID",
    ),
    SensitivePattern(
        category="ipv4",
        pattern=re.compile(
            r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)" r"(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}\b"
        ),
        description="IPv4 address",
    ),
]


# ---------------------------------------------------------------------------
# Scanner class
# ---------------------------------------------------------------------------


class SensitiveDataScanner:
    """Configurable scanner with built-in and custom patterns.

    Instantiate for custom pattern sets; use the module-level
    :func:`scan_text` for the built-in defaults.
    """

    __slots__ = ("_patterns",)

    def __init__(
        self,
        *,
        include_builtins: bool = True,
        patterns: list[SensitivePattern] | None = None,
    ) -> None:
        self._patterns: list[SensitivePattern] = []
        if include_builtins:
            self._patterns.extend(_BUILTIN_PATTERNS)
        if patterns:
            self._patterns.extend(patterns)

    def add_pattern(self, pattern: SensitivePattern) -> None:
        """Register an additional detection pattern."""
        self._patterns.append(pattern)

    def scan(self, text: str) -> list[SensitiveMatch]:
        """Scan *text* and return all matches, sorted by position."""
        matches: list[SensitiveMatch] = []
        for sp in self._patterns:
            for m in sp.pattern.finditer(text):
                matches.append(
                    SensitiveMatch(
                        category=sp.category,
                        matched_text=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )
        matches.sort(key=lambda m: m.start)
        return matches

    def has_sensitive_data(self, text: str) -> bool:
        """Return True if any pattern matches."""
        return any(sp.pattern.search(text) for sp in self._patterns)

    @property
    def categories(self) -> list[str]:
        """Return the list of registered pattern categories."""
        return [p.category for p in self._patterns]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_DEFAULT_SCANNER = SensitiveDataScanner()


def scan_text(text: str) -> list[SensitiveMatch]:
    """Scan *text* with the default built-in patterns."""
    return _DEFAULT_SCANNER.scan(text)
