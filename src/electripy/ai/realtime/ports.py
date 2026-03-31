"""Ports (Protocol interfaces) for realtime orchestration.

Purpose:
  - Define dependency-inversion boundaries for transport, storage,
    tool execution, and observability.

Guarantees:
  - All ports are ``Protocol``-based and runtime-checkable.
  - No concrete implementation details leak into this module.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .domain import (
    EventEnvelope,
    RealtimeSession,
    SessionState,
    ToolCallEvent,
    ToolResultEvent,
)


@runtime_checkable
class RealtimeTransportPort(Protocol):
    """Sends and receives events over a transport layer.

    Implementations may wrap WebSockets, queues, or in-memory channels.
    """

    async def send_event(self, envelope: EventEnvelope) -> None:
        """Send an event to the remote end."""
        ...

    async def receive_event(self) -> EventEnvelope:
        """Block until the next event arrives."""
        ...

    async def close(self) -> None:
        """Gracefully close the transport."""
        ...


@runtime_checkable
class SessionStorePort(Protocol):
    """Persistence layer for realtime sessions."""

    def save(self, session: RealtimeSession) -> None:
        """Persist or update a session."""
        ...

    def load(self, session_id: str) -> RealtimeSession | None:
        """Load a session by ID, or return ``None``."""
        ...

    def delete(self, session_id: str) -> None:
        """Remove a session from the store."""
        ...


@runtime_checkable
class ToolExecutionPort(Protocol):
    """Executes a tool call and returns the result."""

    async def execute(
        self,
        event: ToolCallEvent,
    ) -> ToolResultEvent:
        """Run a tool and return the result.

        Implementations should handle timeouts and error wrapping
        internally.
        """
        ...


@runtime_checkable
class RealtimeObserverPort(Protocol):
    """Optional observability hooks for session lifecycle."""

    def on_state_change(
        self,
        session_id: str,
        previous: SessionState,
        current: SessionState,
    ) -> None:
        """Called when a session transitions state."""
        ...

    def on_event(self, envelope: EventEnvelope) -> None:
        """Called when an event is processed."""
        ...

    def on_error(self, session_id: str, error: Exception) -> None:
        """Called when an error occurs in the session."""
        ...
