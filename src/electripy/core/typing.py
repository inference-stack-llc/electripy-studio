"""Type definitions and utilities for ElectriPy."""

from typing import Any, Union

# JSON types
JSONValue = Union[str, int, float, bool, None, dict[str, Any], list[Any]]
JSONDict = dict[str, JSONValue]
