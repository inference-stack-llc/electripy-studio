"""Tests for json_repair."""

from __future__ import annotations

import pytest

from electripy.ai.json_repair import json_repair, json_repair_raw


class TestJsonRepair:
    def test_clean_json(self) -> None:
        assert json_repair('{"name": "Alice"}') == {"name": "Alice"}

    def test_markdown_fenced(self) -> None:
        text = 'Here\'s the result:\n```json\n{"key": 1}\n```\nDone.'
        assert json_repair(text) == {"key": 1}

    def test_markdown_fenced_no_lang(self) -> None:
        text = '```\n{"x": true}\n```'
        assert json_repair(text) == {"x": True}

    def test_trailing_comma(self) -> None:
        assert json_repair('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self) -> None:
        assert json_repair('{"items": [1, 2, 3,]}') == {"items": [1, 2, 3]}

    def test_single_quotes(self) -> None:
        assert json_repair("{'name': 'Bob'}") == {"name": "Bob"}

    def test_unquoted_keys(self) -> None:
        assert json_repair('{name: "Alice", age: 30}') == {"name": "Alice", "age": 30}

    def test_truncated_json_missing_brace(self) -> None:
        assert json_repair('{"name": "Alice"') == {"name": "Alice"}

    def test_truncated_json_missing_array_bracket(self) -> None:
        assert json_repair('{"items": [1, 2, 3}') == {"items": [1, 2, 3]}

    def test_json_embedded_in_prose(self) -> None:
        text = 'Sure, here you go: {"result": "hello"} Hope that helps!'
        assert json_repair(text) == {"result": "hello"}

    def test_combined_issues(self) -> None:
        # Trailing comma + embedded in prose.
        text = 'Output: {"name": "X", "val": 42,}'
        assert json_repair(text) == {"name": "X", "val": 42}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            json_repair("no json here at all")

    def test_totally_broken_raises(self) -> None:
        with pytest.raises(ValueError, match="Unable to repair"):
            json_repair("{this is not json at all ::: ///}")

    def test_json_repair_raw_returns_string(self) -> None:
        raw = json_repair_raw('{"a": 1,}')
        assert isinstance(raw, str)
        assert '"a"' in raw
        # Should be valid JSON.
        import json

        assert json.loads(raw) == {"a": 1}

    def test_nested_objects(self) -> None:
        text = '{"outer": {"inner": "value",},}'
        result = json_repair(text)
        assert result == {"outer": {"inner": "value"}}

    def test_truncated_with_trailing_comma(self) -> None:
        text = '{"items": [1, 2,'
        result = json_repair(text)
        assert result == {"items": [1, 2]}
