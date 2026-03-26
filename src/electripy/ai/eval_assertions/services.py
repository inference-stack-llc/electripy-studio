"""Services for Eval Assertions."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from .domain import AssertionCheck, AssertionResult, AssertionSeverity


def contains_keywords(
    keywords: Sequence[str],
    *,
    case_sensitive: bool = False,
    severity: AssertionSeverity = AssertionSeverity.ERROR,
) -> AssertionCheck:
    """Create a check that verifies all keywords appear in the output.

    Args:
      keywords: Words or phrases that must all be present.
      case_sensitive: Whether matching is case-sensitive.
      severity: Severity if the check fails.

    Returns:
      An :class:`AssertionCheck` instance.
    """
    kw_list = list(keywords)

    def _check(output: str) -> bool:
        text = output if case_sensitive else output.lower()
        return all((k if case_sensitive else k.lower()) in text for k in kw_list)

    return AssertionCheck(
        name="contains_keywords",
        check_fn=_check,
        severity=severity,
        description=f"must contain: {', '.join(kw_list)}",
    )


def matches_regex(
    pattern: str,
    *,
    severity: AssertionSeverity = AssertionSeverity.ERROR,
) -> AssertionCheck:
    """Create a check that the output matches a regex pattern.

    Args:
      pattern: A Python regex pattern (searched, not full-match).
      severity: Severity if the check fails.

    Returns:
      An :class:`AssertionCheck` instance.
    """
    compiled = re.compile(pattern)

    def _check(output: str) -> bool:
        return compiled.search(output) is not None

    return AssertionCheck(
        name="matches_regex",
        check_fn=_check,
        severity=severity,
        description=f"must match: {pattern}",
    )


def matches_json_schema(
    schema: dict[str, Any],
    *,
    severity: AssertionSeverity = AssertionSeverity.ERROR,
) -> AssertionCheck:
    """Create a check that the output is valid JSON conforming to a schema.

    Uses a lightweight field-presence + type check.  For full JSON Schema
    validation, integrate a dedicated library.

    Args:
      schema: A dict with ``"required"`` (list of field names) and
        ``"properties"`` (field-name → ``{"type": ...}`` mapping).
      severity: Severity if the check fails.

    Returns:
      An :class:`AssertionCheck` instance.
    """
    required_fields = schema.get("required", [])
    properties = schema.get("properties", {})
    _type_map: dict[str, type] = {
        "string": str,
        "number": (int, float),  # type: ignore[dict-item]
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def _check(output: str) -> bool:
        try:
            obj = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(obj, dict):
            return False
        for f in required_fields:
            if f not in obj:
                return False
        for f, spec in properties.items():
            if f in obj:
                expected_type = _type_map.get(spec.get("type", ""))
                if expected_type and not isinstance(obj[f], expected_type):
                    return False
        return True

    return AssertionCheck(
        name="matches_json_schema",
        check_fn=_check,
        severity=severity,
        description=f"must conform to schema with required={required_fields}",
    )


def passes_predicate(
    predicate: Callable[[str], bool],
    *,
    name: str = "custom_predicate",
    description: str = "",
    severity: AssertionSeverity = AssertionSeverity.ERROR,
) -> AssertionCheck:
    """Create a check using a custom predicate function.

    Args:
      predicate: A callable that accepts the output string and returns
        True if the check passes.
      name: Label for the check.
      description: Explanation of what the predicate verifies.
      severity: Severity if the check fails.

    Returns:
      An :class:`AssertionCheck` instance.
    """
    return AssertionCheck(
        name=name,
        check_fn=predicate,
        severity=severity,
        description=description,
    )


def satisfies_length(
    *,
    min_length: int = 0,
    max_length: int | None = None,
    severity: AssertionSeverity = AssertionSeverity.ERROR,
) -> AssertionCheck:
    """Create a check that the output length is within bounds.

    Args:
      min_length: Minimum character count (inclusive).
      max_length: Maximum character count (inclusive), or None for no limit.
      severity: Severity if the check fails.

    Returns:
      An :class:`AssertionCheck` instance.
    """

    def _check(output: str) -> bool:
        length = len(output)
        if length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        return True

    desc = f"length in [{min_length}, {max_length or '∞'}]"
    return AssertionCheck(
        name="satisfies_length",
        check_fn=_check,
        severity=severity,
        description=desc,
    )


def assert_llm_output(
    output: str,
    *,
    checks: Sequence[AssertionCheck],
    raise_on_warning: bool = False,
) -> list[AssertionResult]:
    """Run all checks against an LLM output string.

    Args:
      output: The raw LLM output text to validate.
      checks: Sequence of assertion checks to evaluate.
      raise_on_warning: If True, warnings also raise AssertionError.

    Returns:
      A list of :class:`AssertionResult` objects for all checks.

    Raises:
      AssertionError: If any ERROR-severity check fails (or any WARNING
        check when *raise_on_warning* is True).  The error message
        includes a full diagnostic report.
    """
    results: list[AssertionResult] = []
    failures: list[str] = []

    for check in checks:
        passed = check.check_fn(output)
        result = AssertionResult(
            check_name=check.name,
            passed=passed,
            severity=check.severity,
            message=check.description if not passed else "ok",
        )
        results.append(result)

        if not passed:
            is_error = check.severity == AssertionSeverity.ERROR
            if is_error or raise_on_warning:
                failures.append(
                    f"[{check.severity.value.upper()}] {check.name}: {check.description}"
                )

    if failures:
        report = "\n".join(failures)
        raise AssertionError(
            f"LLM output failed {len(failures)} check(s):\n{report}\n\n"
            f"Output (first 200 chars): {output[:200]!r}"
        )

    return results
