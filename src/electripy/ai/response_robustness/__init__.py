"""Utilities for robust model output handling and repair."""

from __future__ import annotations

from .domain import JsonRepairResult
from .services import (
    coalesce_non_empty,
    extract_json_object,
    parse_json_with_repair,
    require_fields,
)

__all__ = [
    "JsonRepairResult",
    "extract_json_object",
    "parse_json_with_repair",
    "require_fields",
    "coalesce_non_empty",
]
