"""Domain models for response robustness helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class JsonRepairResult:
    """Result for JSON parsing with best-effort repair."""

    value: dict[str, object]
    repaired: bool
    raw_json: str
