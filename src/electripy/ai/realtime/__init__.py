"""Realtime session and event orchestration for streaming AI systems.

Purpose:
  - Provide a provider-neutral runtime substrate for managing
    realtime streaming sessions, events, tool calls, interruptions,
    and lifecycle transitions.
  - Replace fragile WebSocket/event-loop glue code with reusable,
    strongly typed orchestration primitives.

Guarantees:
  - Session state transitions are explicit and guarded.
  - Events are sequenced, typed, and transport-agnostic.
  - Tool execution is async-safe and decoupled from providers.
  - All external dependencies are behind Protocol ports.

Usage::

    from electripy.ai.realtime import (
        RealtimeSessionService,
        OutputStreamChunk,
        ChunkStatus,
    )

    svc = RealtimeSessionService()
    session = svc.create_session()
    svc.start_session(session.session_id)
"""

from __future__ import annotations

from .adapters import (
    EchoToolExecutor,
    InMemoryObserver,
    InMemorySessionStore,
    InMemoryTransport,
    NoOpObserver,
)
from .domain import (
    BackpressureDirective,
    ChunkStatus,
    EventEnvelope,
    EventKind,
    EventPayload,
    InputStreamChunk,
    InterruptEvent,
    OutputStreamChunk,
    RealtimeConfig,
    RealtimeErrorPayload,
    RealtimeSession,
    SessionLifecycleEvent,
    SessionState,
    ToolCallEvent,
    ToolResultEvent,
    VALID_TRANSITIONS,
)
from .errors import (
    RealtimeError,
    SessionNotFoundError,
    SessionStateError,
    ToolExecutionError,
    TransportError,
)
from .ports import (
    RealtimeObserverPort,
    RealtimeTransportPort,
    SessionStorePort,
    ToolExecutionPort,
)
from .services import (
    RealtimeSessionService,
    async_collect_output_text,
    async_iter_output_text,
    collect_output_text,
    iter_output_text,
)

__all__ = [
    # Domain — enums
    "SessionState",
    "EventKind",
    "ChunkStatus",
    # Domain — config
    "RealtimeConfig",
    # Domain — stream chunks
    "InputStreamChunk",
    "OutputStreamChunk",
    # Domain — tool events
    "ToolCallEvent",
    "ToolResultEvent",
    # Domain — control events
    "InterruptEvent",
    "SessionLifecycleEvent",
    "BackpressureDirective",
    "RealtimeErrorPayload",
    # Domain — envelope & session
    "EventEnvelope",
    "EventPayload",
    "RealtimeSession",
    "VALID_TRANSITIONS",
    # Errors
    "RealtimeError",
    "SessionStateError",
    "SessionNotFoundError",
    "TransportError",
    "ToolExecutionError",
    # Ports
    "RealtimeTransportPort",
    "SessionStorePort",
    "ToolExecutionPort",
    "RealtimeObserverPort",
    # Adapters
    "InMemoryTransport",
    "InMemorySessionStore",
    "NoOpObserver",
    "InMemoryObserver",
    "EchoToolExecutor",
    # Services
    "RealtimeSessionService",
    # Streaming helpers
    "iter_output_text",
    "collect_output_text",
    "async_iter_output_text",
    "async_collect_output_text",
]
