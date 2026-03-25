"""Services for robust parsing and fallback response handling."""

from __future__ import annotations

import json
import re

from .domain import JsonRepairResult

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> str:
    """Extract a JSON object from raw model text.

    Raises:
      ValueError: If no object-like content is found.
    """

    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)

    match = _JSON_OBJECT_RE.search(text)
    if match:
        return match.group(0)

    raise ValueError("no JSON object found")


def _remove_trailing_commas(raw: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", raw)


def parse_json_with_repair(text: str) -> JsonRepairResult:
    """Parse model output JSON with minimal safe repairs."""

    raw_json = extract_json_object(text)
    try:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError("expected top-level JSON object")
        return JsonRepairResult(value=parsed, repaired=False, raw_json=raw_json)
    except json.JSONDecodeError:
        repaired = _remove_trailing_commas(raw_json)
        parsed = json.loads(repaired)
        if not isinstance(parsed, dict):
            raise ValueError("expected top-level JSON object") from None
        return JsonRepairResult(value=parsed, repaired=True, raw_json=repaired)


def require_fields(value: dict[str, object], fields: list[str]) -> None:
    """Ensure required fields exist in parsed output."""

    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")


def coalesce_non_empty(candidates: list[str]) -> str:
    """Return first non-empty stripped string from candidates."""

    for candidate in candidates:
        stripped = candidate.strip()
        if stripped:
            return stripped
    raise ValueError("all candidates are empty")
