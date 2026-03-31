"""Exception hierarchy for the realtime package."""

from __future__ import annotations

from electripy.core.errors import ElectriPyError


class RealtimeError(ElectriPyError):
    """Base error for all realtime orchestration failures."""


class SessionStateError(RealtimeError):
    """Raised when a session state transition is invalid."""

    def __init__(self, current: str, target: str) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Cannot transition from {current!r} to {target!r}")


class SessionNotFoundError(RealtimeError):
    """Raised when a session ID cannot be resolved."""


class TransportError(RealtimeError):
    """Raised for transport-level failures (send/receive)."""


class ToolExecutionError(RealtimeError):
    """Raised when a tool execution fails within a session."""

    def __init__(self, tool_name: str, detail: str) -> None:
        self.tool_name = tool_name
        self.detail = detail
        super().__init__(f"Tool {tool_name!r} failed: {detail}")
