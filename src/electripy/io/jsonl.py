"""JSONL (JSON Lines) file read/write utilities."""

import json
from collections.abc import Generator
from pathlib import Path

from electripy.core.typing import JSONDict


def read_jsonl(
    path: str | Path,
    encoding: str = "utf-8",
) -> Generator[JSONDict, None, None]:
    """Read JSONL file line by line.

    Args:
        path: Path to JSONL file
        encoding: File encoding (default: utf-8)

    Yields:
        Parsed JSON objects from each line

    Example:
        for record in read_jsonl("data.jsonl"):
            print(record)
    """
    path = Path(path)
    with path.open("r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(
    path: str | Path,
    data: list[JSONDict],
    encoding: str = "utf-8",
) -> None:
    """Write data to JSONL file.

    Args:
        path: Path to JSONL file
        data: List of JSON-serializable dictionaries
        encoding: File encoding (default: utf-8)

    Example:
        records = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        write_jsonl("output.jsonl", records)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding=encoding) as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(
    path: str | Path,
    record: JSONDict,
    encoding: str = "utf-8",
) -> None:
    """Append a single record to JSONL file.

    Args:
        path: Path to JSONL file
        record: JSON-serializable dictionary
        encoding: File encoding (default: utf-8)

    Example:
        append_jsonl("log.jsonl", {"timestamp": "2024-01-01", "event": "login"})
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding=encoding) as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
