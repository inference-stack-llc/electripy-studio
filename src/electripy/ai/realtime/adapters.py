"""Concrete adapter implementations for realtime ports.

Purpose:
  - Provide in-memory and no-op adapters for testing, development,
    and lightweight production use.

Guarantees:
  - All adapters satisfy the corresponding port protocols.
  - In-memory adapters are fully deterministic and thread-safe for
    single-event-loop usage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .domain import (
    EventEnvelope,
    RealtimeSession,
    SessionState,
    ToolCallEvent,
    ToolResultEvent,
)


# ── Transport ────────────────────────────────────────────────────────


class InMemoryTransport:
    """Async queue-backed transport for testing and local development.

    Events sent via :meth:`send_event` are placed on an outbound queue.
    Events to be received are placed on the inbound queue by the test
    harness or paired transport.
    """

    __slots__ = ("_inbound", "_outbound", "_closed")

    def __init__(self, *, maxsize: int = 0) -> None:
        self._inbound: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=maxsize)
        self._outbound: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    async def send_event(self, envelope: EventEnvelope) -> None:
        """Place an event on the outbound queue."""
        if self._closed:
            return
        await self._outbound.put(envelope)

    async def receive_event(self) -> EventEnvelope:
        """Wait for the next inbound event."""
        return await self._inbound.get()

    async def close(self) -> None:
        """Mark the transport as closed."""
        self._closed = True

    # -- Test helpers --------------------------------------------------

    async def inject(self, envelope: EventEnvelope) -> None:
        """Inject an event into the inbound queue (test helper)."""
        await self._inbound.put(envelope)

    def drain_outbound(self) -> list[EventEnvelope]:
        """Drain all outbound events into a list (test helper)."""
        items: list[EventEnvelope] = []
        while not self._outbound.empty():
            items.append(self._outbound.get_nowait())
        return items

    @property
    def closed(self) -> bool:
        return self._closed


# ── Session store ────────────────────────────────────────────────────


class InMemorySessionStore:
    """Dict-backed session store for testing and development."""

    __slots__ = ("_sessions",)

    def __init__(self) -> None:
        self._sessions: dict[str, RealtimeSession] = {}

    def save(self, session: RealtimeSession) -> None:
        self._sessions[session.session_id] = session

    def load(self, session_id: str) -> RealtimeSession | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ── Observer ─────────────────────────────────────────────────────────


class NoOpObserver:
    """Zero-cost observer that discards all notifications."""

    __slots__ = ()

    def on_state_change(
        self,
        session_id: str,
        previous: SessionState,
        current: SessionState,
    ) -> None:
        pass

    def on_event(self, envelope: EventEnvelope) -> None:
        pass

    def on_error(self, session_id: str, error: Exception) -> None:
        pass


@dataclass(slots=True)
class InMemoryObserver:
    """Captures all observer notifications for assertions in tests.

    Attributes:
        state_changes: List of (session_id, previous, current) tuples.
        events: List of recorded event envelopes.
        errors: List of (session_id, exception) tuples.
    """

    state_changes: list[tuple[str, SessionState, SessionState]] = field(
        default_factory=list,
    )
    events: list[EventEnvelope] = field(default_factory=list)
    errors: list[tuple[str, Exception]] = field(default_factory=list)

    def on_state_change(
        self,
        session_id: str,
        previous: SessionState,
        current: SessionState,
    ) -> None:
        self.state_changes.append((session_id, previous, current))

    def on_event(self, envelope: EventEnvelope) -> None:
        self.events.append(envelope)

    def on_error(self, session_id: str, error: Exception) -> None:
        self.errors.append((session_id, error))


# ── Tool execution ───────────────────────────────────────────────────


class EchoToolExecutor:
    """Test executor that echoes tool arguments as the result."""

    __slots__ = ()

    async def execute(self, event: ToolCallEvent) -> ToolResultEvent:
        return ToolResultEvent(
            call_id=event.call_id,
            tool_name=event.tool_name,
            result=dict(event.arguments),
        )
