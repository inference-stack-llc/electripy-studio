"""Ports for agent-runtime tool execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ToolPort(Protocol):
    """Protocol for invoking external tools used by the runtime."""

    def execute(self, name: str, args: dict[str, object]) -> str:
        """Execute tool by name with structured arguments."""

        ...
