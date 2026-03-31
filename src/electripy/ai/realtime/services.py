"""Realtime session service — primary orchestration entry-point.

Purpose:
  - Wire session store, transport, tool execution, and observer into
    one facade.
  - Manage session lifecycle, event routing, and state transitions.

Guarantees:
  - State transitions are guarded and raise :class:`SessionStateError`
    on illegal moves.
  - All events are sequenced, recorded, and forwarded to the observer.
  - Tool calls are async-safe and decoupled from transport.

Example::

    from electripy.ai.realtime import RealtimeSessionService

    svc = RealtimeSessionService()
    session = svc.create_session()
    svc.start_session(session.session_id)
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator

from electripy.core.logging import get_logger

from .adapters import (
    InMemorySessionStore,
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
    _make_event_id,
)
from .errors import (
    RealtimeError,
    SessionNotFoundError,
    SessionStateError,
    ToolExecutionError,
)
from .ports import (
    RealtimeObserverPort,
    RealtimeTransportPort,
    SessionStorePort,
    ToolExecutionPort,
)

__all__ = [
    "RealtimeSessionService",
    "collect_output_text",
    "async_collect_output_text",
    "iter_output_text",
    "async_iter_output_text",
]

logger = get_logger(__name__)


# ── Mapping from EventKind to allowed session states ─────────────────

_INGEST_REQUIRES_ACTIVE: frozenset[EventKind] = frozenset({
    EventKind.INPUT_TEXT,
    EventKind.INPUT_AUDIO,
    EventKind.OUTPUT_TEXT,
    EventKind.OUTPUT_AUDIO,
})


class RealtimeSessionService:
    """Primary orchestrator for realtime session lifecycle.

    Attributes:
        store: Persistence layer for sessions.
        observer: Optional lifecycle/event observer.
        tool_executor: Optional tool-call executor.
        transport: Optional transport for sending events outbound.
    """

    __slots__ = ("_store", "_observer", "_tool_executor", "_transport")

    def __init__(
        self,
        *,
        store: SessionStorePort | None = None,
        observer: RealtimeObserverPort | None = None,
        tool_executor: ToolExecutionPort | None = None,
        transport: RealtimeTransportPort | None = None,
    ) -> None:
        self._store: SessionStorePort = store or InMemorySessionStore()
        self._observer: RealtimeObserverPort = observer or NoOpObserver()
        self._tool_executor = tool_executor
        self._transport = transport

    # ── Session lifecycle ────────────────────────────────────────────

    def create_session(
        self,
        *,
        config: RealtimeConfig | None = None,
        metadata: dict[str, object] | None = None,
    ) -> RealtimeSession:
        """Create a new session in INITIALIZED state.

        Args:
            config: Session configuration (defaults apply if omitted).
            metadata: Arbitrary metadata attached to the session.

        Returns:
            The newly created session.
        """
        session = RealtimeSession(
            session_id=uuid.uuid4().hex,
            config=config or RealtimeConfig(),
            metadata=metadata or {},
        )
        self._store.save(session)
        logger.info("Created session %s", session.session_id)
        return session

    def start_session(self, session_id: str) -> RealtimeSession:
        """Transition a session from INITIALIZED to ACTIVE.

        Args:
            session_id: ID of the session to start.

        Returns:
            The updated session.

        Raises:
            SessionNotFoundError: Session does not exist.
            SessionStateError: Session is not in INITIALIZED state.
        """
        session = self._get_session(session_id)
        self._transition(session, SessionState.ACTIVE, reason="session started")
        return session

    def close_session(self, session_id: str) -> RealtimeSession:
        """Close a session (terminal state).

        Args:
            session_id: ID of the session to close.

        Returns:
            The closed session.
        """
        session = self._get_session(session_id)
        self._transition(session, SessionState.CLOSED, reason="session closed")
        return session

    def complete_session(self, session_id: str) -> RealtimeSession:
        """Mark a session as COMPLETED (natural end of generation).

        Args:
            session_id: ID of the session.

        Returns:
            The completed session.
        """
        session = self._get_session(session_id)
        self._transition(session, SessionState.COMPLETED, reason="generation complete")
        return session

    def fail_session(self, session_id: str, reason: str = "") -> RealtimeSession:
        """Mark a session as FAILED.

        Args:
            session_id: ID of the session.
            reason: Human-readable failure explanation.

        Returns:
            The failed session.
        """
        session = self._get_session(session_id)
        self._transition(session, SessionState.FAILED, reason=reason or "session failed")
        return session

    def get_session(self, session_id: str) -> RealtimeSession:
        """Retrieve a session by ID.

        Raises:
            SessionNotFoundError: Session does not exist.
        """
        return self._get_session(session_id)

    # ── Event ingestion ──────────────────────────────────────────────

    def ingest_event(
        self,
        session_id: str,
        kind: EventKind,
        payload: EventPayload,
    ) -> EventEnvelope:
        """Ingest an event into a session.

        The event is validated, routed, recorded, and the observer
        is notified.

        Args:
            session_id: Target session.
            kind: The event discriminator.
            payload: The event data.

        Returns:
            The sequenced event envelope.

        Raises:
            SessionNotFoundError: Session does not exist.
            SessionStateError: Session state does not allow this event.
        """
        session = self._get_session(session_id)

        # Validate state for input/output events.
        if kind in _INGEST_REQUIRES_ACTIVE and session.state != SessionState.ACTIVE:
            raise SessionStateError(session.state.value, "active")

        envelope = EventEnvelope(
            event_id=_make_event_id(),
            session_id=session_id,
            kind=kind,
            payload=payload,
            sequence=session.next_sequence(),
        )
        session.event_log.append(envelope)
        self._store.save(session)
        self._observer.on_event(envelope)

        logger.debug(
            "Ingested %s event seq=%d for session %s",
            kind.value,
            envelope.sequence,
            session_id,
        )
        return envelope

    def emit_output(
        self,
        session_id: str,
        chunk: OutputStreamChunk,
    ) -> EventEnvelope:
        """Convenience: ingest an output text/audio chunk.

        Selects ``OUTPUT_TEXT`` or ``OUTPUT_AUDIO`` based on content.
        """
        kind = EventKind.OUTPUT_AUDIO if chunk.audio_bytes else EventKind.OUTPUT_TEXT
        return self.ingest_event(session_id, kind, chunk)

    # ── Interruption ─────────────────────────────────────────────────

    def interrupt(
        self,
        session_id: str,
        *,
        reason: str = "",
        hard: bool = False,
    ) -> EventEnvelope:
        """Interrupt the current generation or tool execution.

        Transitions the session to INTERRUPTED and records the event.

        Args:
            session_id: Target session.
            reason: Human-readable reason.
            hard: If ``True``, discard buffered output.

        Returns:
            The interrupt event envelope.
        """
        session = self._get_session(session_id)
        self._transition(session, SessionState.INTERRUPTED, reason=reason or "interrupted")
        payload = InterruptEvent(reason=reason, hard=hard)
        return self.ingest_event(session_id, EventKind.INTERRUPT, payload)

    def resume(self, session_id: str) -> RealtimeSession:
        """Resume an interrupted session back to ACTIVE.

        Args:
            session_id: Target session.

        Returns:
            The resumed session.
        """
        session = self._get_session(session_id)
        self._transition(session, SessionState.ACTIVE, reason="resumed")
        return session

    # ── Tool calls ───────────────────────────────────────────────────

    async def handle_tool_call(
        self,
        session_id: str,
        tool_call: ToolCallEvent,
    ) -> ToolResultEvent:
        """Execute a tool call within a session.

        Transitions the session to WAITING_ON_TOOL, executes the tool,
        records the result, and transitions back to ACTIVE.

        Args:
            session_id: Target session.
            tool_call: The tool call event.

        Returns:
            The tool result.

        Raises:
            ToolExecutionError: If the tool executor fails.
            RealtimeError: If no tool executor is configured.
        """
        if self._tool_executor is None:
            raise RealtimeError("No tool executor configured")

        session = self._get_session(session_id)
        self._transition(session, SessionState.WAITING_ON_TOOL, reason=f"tool:{tool_call.tool_name}")

        # Record tool call event.
        self.ingest_event(session_id, EventKind.TOOL_CALL, tool_call)

        try:
            result = await self._tool_executor.execute(tool_call)
        except Exception as exc:
            error = ToolExecutionError(tool_call.tool_name, str(exc))
            self._observer.on_error(session_id, error)
            # Stay in WAITING_ON_TOOL or transition to FAILED.
            self._transition(session, SessionState.FAILED, reason=str(error))
            raise error from exc

        # Record result and return to ACTIVE.
        self.ingest_event(session_id, EventKind.TOOL_RESULT, result)
        self._transition(session, SessionState.ACTIVE, reason="tool completed")
        return result

    # ── Backpressure ─────────────────────────────────────────────────

    def emit_backpressure(
        self,
        session_id: str,
        directive: BackpressureDirective,
    ) -> EventEnvelope:
        """Record a backpressure directive in the event log."""
        return self.ingest_event(session_id, EventKind.BACKPRESSURE, directive)

    # ── Transport integration ────────────────────────────────────────

    async def send_to_transport(
        self,
        session_id: str,
        envelope: EventEnvelope,
    ) -> None:
        """Forward an event to the configured transport.

        No-ops silently if no transport is configured.
        """
        if self._transport is None:
            return
        await self._transport.send_event(envelope)

    # ── Replay ───────────────────────────────────────────────────────

    def replay_events(
        self,
        session_id: str,
        *,
        kinds: frozenset[EventKind] | None = None,
    ) -> list[EventEnvelope]:
        """Return buffered events from a session, optionally filtered.

        Args:
            session_id: Target session.
            kinds: If provided, only return events of these kinds.

        Returns:
            Ordered list of matching events.
        """
        session = self._get_session(session_id)
        if kinds is None:
            return list(session.event_log)
        return [e for e in session.event_log if e.kind in kinds]

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_session(self, session_id: str) -> RealtimeSession:
        session = self._store.load(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id!r}")
        return session

    def _transition(
        self,
        session: RealtimeSession,
        target: SessionState,
        *,
        reason: str = "",
    ) -> None:
        """Guard and execute a state transition."""
        if not session.can_transition_to(target):
            raise SessionStateError(session.state.value, target.value)

        previous = session.state
        session.state = target
        self._store.save(session)

        # Record lifecycle event.
        lifecycle = SessionLifecycleEvent(
            previous_state=previous,
            new_state=target,
            reason=reason,
        )
        envelope = EventEnvelope(
            event_id=_make_event_id(),
            session_id=session.session_id,
            kind=EventKind.LIFECYCLE,
            payload=lifecycle,
            sequence=session.next_sequence(),
        )
        session.event_log.append(envelope)

        self._observer.on_state_change(session.session_id, previous, target)
        logger.info(
            "Session %s: %s -> %s (%s)",
            session.session_id,
            previous.value,
            target.value,
            reason,
        )


# ── Streaming helpers ────────────────────────────────────────────────


def iter_output_text(chunks: Iterable[OutputStreamChunk]) -> Iterator[str]:
    """Yield text deltas from output chunks."""
    for chunk in chunks:
        if chunk.text:
            yield chunk.text


def collect_output_text(chunks: Iterable[OutputStreamChunk]) -> str:
    """Concatenate all text deltas from output chunks."""
    return "".join(iter_output_text(chunks))


async def async_iter_output_text(
    chunks: AsyncIterable[OutputStreamChunk],
) -> AsyncIterator[str]:
    """Yield text deltas from an async stream of output chunks."""
    async for chunk in chunks:
        if chunk.text:
            yield chunk.text


async def async_collect_output_text(
    chunks: AsyncIterable[OutputStreamChunk],
) -> str:
    """Concatenate all text from an async output chunk stream."""
    parts: list[str] = []
    async for text in async_iter_output_text(chunks):
        parts.append(text)
    return "".join(parts)
