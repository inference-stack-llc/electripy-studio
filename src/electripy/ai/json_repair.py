"""JSON Repair — fix common LLM JSON breakage in one call.

Purpose:
  - Strip markdown fences, fix trailing commas, single-quote keys,
    unquoted keys, and truncated JSON from LLM output.
  - Return a parsed ``dict`` or the raw repaired string.

Guarantees:
  - Pure function — no side effects or network calls.
  - Gracefully returns the best-effort result; raises on total failure.
  - Composes with ``response_robustness`` internally.

Usage::

    from electripy.ai.json_repair import json_repair

    text = '''Here is the result:
    ```json
    {"name": "Alice", "age": 30,}
    ```'''

    data = json_repair(text)
    print(data)  # {"name": "Alice", "age": 30}
"""

from __future__ import annotations

import json
import re

__all__ = [
    "json_repair",
    "json_repair_raw",
]

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_OPEN_RE = re.compile(r"\{.*", re.DOTALL)


def json_repair(text: str) -> dict[str, object]:
    """Extract and repair a JSON object from *text*.

    Applies the following repairs in order:

    1. Strip markdown ````` fences.
    2. Extract the outermost ``{…}`` block.
    3. Remove trailing commas before ``}`` or ``]``.
    4. Replace single-quoted strings with double quotes.
    5. Quote bare (unquoted) keys.
    6. Attempt to close truncated JSON (missing closing braces).

    Raises:
        ValueError: If no JSON object can be recovered.
    """
    raw = _extract_object(text)
    return _parse_with_repairs(raw)


def json_repair_raw(text: str) -> str:
    """Like :func:`json_repair` but return the repaired JSON string.

    Useful when you need the normalised JSON text rather than a dict.
    """
    raw = _extract_object(text)
    repaired = _apply_repairs(raw)
    # Validate it parses, then return the string.
    json.loads(repaired)
    return repaired


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_object(text: str) -> str:
    """Pull the JSON object substring from surrounding prose."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    match = _JSON_OBJECT_RE.search(text)
    if match:
        return match.group(0).strip()

    # Truncated JSON — starts with { but may lack closing brace.
    match = _JSON_OPEN_RE.search(text)
    if match:
        return match.group(0).strip()

    raise ValueError("No JSON object found in text.")


def _parse_with_repairs(raw: str) -> dict[str, object]:
    """Try progressively more aggressive repairs until parsing succeeds."""
    # Attempt 1: raw as-is.
    parsed = _try_parse(raw)
    if parsed is not None:
        return parsed

    # Attempt 2: standard repairs.
    repaired = _apply_repairs(raw)
    parsed = _try_parse(repaired)
    if parsed is not None:
        return parsed

    # Attempt 3: close truncated JSON.
    closed = _close_truncated(repaired)
    parsed = _try_parse(closed)
    if parsed is not None:
        return parsed

    raise ValueError(f"Unable to repair JSON: {raw[:200]}")


def _apply_repairs(raw: str) -> str:
    """Apply all safe string-level repairs."""
    result = raw
    result = _fix_trailing_commas(result)
    result = _fix_single_quotes(result)
    result = _fix_unquoted_keys(result)
    result = _fix_mismatched_brackets(result)
    return result


def _fix_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_single_quotes(text: str) -> str:
    """Replace single-quoted strings with double quotes.

    Only activates if the text contains no double quotes at all
    (i.e. the LLM used single quotes exclusively), to avoid breaking
    strings that legitimately contain apostrophes.
    """
    if '"' not in text:
        return text.replace("'", '"')
    return text


def _fix_unquoted_keys(text: str) -> str:
    """Quote bare JavaScript-style object keys."""
    return re.sub(
        r"(?<=[\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
        r' "\1":',
        text,
    )


def _fix_mismatched_brackets(text: str) -> str:
    """Insert missing closing brackets when a closer matches a deeper stack entry.

    Handles cases like ``{"items": [1, 2, 3}`` where the ``]`` is missing
    before the ``}``.  The ``]`` is inserted automatically.
    """
    result: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\":
            result.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            result.append(ch)
            continue
        if ch == "{":
            stack.append("}")
            result.append(ch)
        elif ch == "[":
            stack.append("]")
            result.append(ch)
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()
                result.append(ch)
            elif ch in stack:
                # Insert missing closers for unclosed brackets above the match.
                while stack and stack[-1] != ch:
                    result.append(stack.pop())
                if stack:
                    stack.pop()
                result.append(ch)
            else:
                result.append(ch)
        else:
            result.append(ch)

    return "".join(result)


def _close_truncated(text: str) -> str:
    """Append missing closing braces/brackets for truncated JSON."""
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    # Remove a dangling trailing comma before closing.
    text = re.sub(r",\s*$", "", text)
    # Close in reverse order (innermost first).
    return text + "".join(reversed(stack))


def _try_parse(text: str) -> dict[str, object] | None:
    """Parse text as JSON dict or return None."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None
