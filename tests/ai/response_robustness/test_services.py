from __future__ import annotations

from electripy.ai.response_robustness import (
    coalesce_non_empty,
    parse_json_with_repair,
    require_fields,
)


def test_parse_json_with_repair_trailing_comma() -> None:
    text = '```json\n{"a": 1,}\n```'
    result = parse_json_with_repair(text)

    assert result.value == {"a": 1}
    assert result.repaired is True


def test_require_fields_and_coalesce() -> None:
    payload = {"answer": "ok", "confidence": 0.8}
    require_fields(payload, ["answer", "confidence"])
    assert coalesce_non_empty(["", "  ", "hello"]) == "hello"
