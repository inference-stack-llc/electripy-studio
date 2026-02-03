"""Type definitions and utilities for ElectriPy."""

from typing import Any

# JSON types
JSONValue = str | int | float | bool | None | dict[str, Any] | list[Any]
JSONDict = dict[str, JSONValue]
