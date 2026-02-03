"""Tests for io.jsonl module."""

import json
import tempfile
from pathlib import Path

import pytest

from electripy.io.jsonl import append_jsonl, read_jsonl, write_jsonl


def test_write_and_read_jsonl() -> None:
    """Test writing and reading JSONL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]
        
        write_jsonl(path, data)
        
        # Read back and verify
        result = list(read_jsonl(path))
        assert len(result) == 3
        assert result[0]["name"] == "Alice"
        assert result[1]["id"] == 2


def test_append_jsonl() -> None:
    """Test appending to JSONL file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        # Append multiple records
        append_jsonl(path, {"id": 1, "value": "first"})
        append_jsonl(path, {"id": 2, "value": "second"})
        
        # Read back and verify
        result = list(read_jsonl(path))
        assert len(result) == 2
        assert result[0]["value"] == "first"
        assert result[1]["value"] == "second"


def test_read_jsonl_empty_lines() -> None:
    """Test reading JSONL with empty lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        # Write file with empty lines
        with path.open("w") as f:
            f.write('{"id": 1}\n')
            f.write('\n')
            f.write('{"id": 2}\n')
        
        result = list(read_jsonl(path))
        assert len(result) == 2


def test_write_jsonl_creates_directory() -> None:
    """Test write_jsonl creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "nested" / "test.jsonl"
        
        data = [{"id": 1}]
        write_jsonl(path, data)
        
        assert path.exists()
        result = list(read_jsonl(path))
        assert len(result) == 1


def test_read_jsonl_unicode() -> None:
    """Test JSONL handles Unicode correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        data = [
            {"text": "Hello 世界"},
            {"text": "Привет мир"},
        ]
        
        write_jsonl(path, data)
        result = list(read_jsonl(path))
        
        assert result[0]["text"] == "Hello 世界"
        assert result[1]["text"] == "Привет мир"
