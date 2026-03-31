"""Domain models for realtime session and event orchestration.

Purpose:
  - Define the core types for managing streaming AI sessions, events,
    state transitions, and tool-call lifecycle in a provider-neutral way.

Guarantees:
  - All value objects are frozen and immutable.
  - Session state transitions are explicit via :class:`SessionState`.
  - Event kinds are typed via :class:`EventKind`.
  - No transport or provider dependencies leak into this layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeAlias

# ── Enumerations ─────────────────────────────────────────────────────


class SessionState(StrEnum):
    """Lifecycle state of a realtime session."""

    INITIALIZED = "initialized"
    ACTIVE = "active"
    INTERRUPTED = "interrupted"
    WAITING_ON_TOOL = "waiting_on_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    CLOSED = "closed"


class EventKind(StrEnum):
    """Discriminator for realtime event types."""

    INPUT_TEXT = "input_text"
    INPUT_AUDIO = "input_audio"
    OUTPUT_TEXT = "output_text"
    OUTPUT_AUDIO = "output_audio"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INTERRUPT = "interrupt"
    LIFECYCLE = "lifecycle"
    ERROR = "error"
    BACKPRESSURE = "backpressure"


class ChunkStatus(StrEnum):
    """Whether a stream chunk is partial or final."""

    PARTIAL = "partial"
    FINAL = "final"


# ── Type aliases ─────────────────────────────────────────────────────

Metadata: TypeAlias = dict[str, object]


# ── Configuration ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RealtimeConfig:
    """Session-level configuration.

    Attributes:
        model: Model identifier (provider-neutral).
        timeout_seconds: Maximum session duration in seconds.
        max_events: Upper bound on buffered events (0 = unlimited).
        metadata: Arbitrary configuration metadata.
    """

    model: str = ""
    timeout_seconds: float = 300.0
    max_events: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


# ── Stream chunks ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InputStreamChunk:
    """A chunk of input submitted to a realtime session.

    Attributes:
        index: Sequence position within the stream.
        text: Text content (empty for audio-only).
        audio_bytes: Raw audio payload (empty for text-only).
        status: Whether this is a partial or final chunk.
        metadata: Optional non-sensitive metadata.
    """

    index: int
    text: str = ""
    audio_bytes: bytes = b""
    status: ChunkStatus = ChunkStatus.PARTIAL
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputStreamChunk:
    """A chunk of output emitted by a realtime session.

    Attributes:
        index: Sequence position within the stream.
        text: Text content (empty for audio-only).
        audio_bytes: Raw audio payload (empty for text-only).
        status: Whether this is a partial or final chunk.
        metadata: Optional non-sensitive metadata.
    """

    index: int
    text: str = ""
    audio_bytes: bytes = b""
    status: ChunkStatus = ChunkStatus.PARTIAL
    metadata: dict[str, object] = field(default_factory=dict)


# ── Tool events ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """Request to execute a tool during a realtime session.

    Attributes:
        call_id: Unique identifier for this tool invocation.
        tool_name: Name of the tool to execute.
        arguments: JSON-serializable arguments for the tool.
        metadata: Optional metadata.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """Result of a tool execution.

    Attributes:
        call_id: ID of the originating :class:`ToolCallEvent`.
        tool_name: Name of the executed tool.
        result: JSON-serializable result payload.
        error: Error message if the tool failed.
        metadata: Optional metadata.
    """

    call_id: str
    tool_name: str
    result: dict[str, object] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


# ── Control events ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InterruptEvent:
    """Signal to interrupt the current generation or tool execution.

    Attributes:
        reason: Human-readable reason for the interruption.
        hard: If ``True``, discard buffered output; otherwise drain first.
    """

    reason: str = ""
    hard: bool = False


@dataclass(frozen=True, slots=True)
class SessionLifecycleEvent:
    """Records a state transition in the session.

    Attributes:
        previous_state: State before the transition.
        new_state: State after the transition.
        reason: Optional explanation.
    """

    previous_state: SessionState
    new_state: SessionState
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BackpressureDirective:
    """Signal that the consumer or producer should slow down.

    Attributes:
        queue_depth: Current number of buffered events.
        recommended_delay_ms: Suggested delay in milliseconds.
        reason: Human-readable explanation.
    """

    queue_depth: int = 0
    recommended_delay_ms: int = 100
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RealtimeErrorPayload:
    """Payload for error events within the event stream.

    Attributes:
        code: Machine-readable error code.
        message: Human-readable description.
        recoverable: Whether the session can continue.
    """

    code: str
    message: str
    recoverable: bool = True


# ── Event envelope ───────────────────────────────────────────────────

EventPayload: TypeAlias = (
    InputStreamChunk
    | OutputStreamChunk
    | ToolCallEvent
    | ToolResultEvent
    | InterruptEvent
    | SessionLifecycleEvent
    | BackpressureDirective
    | RealtimeErrorPayload
)


def _make_event_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Typed wrapper around any realtime event.

    Attributes:
        event_id: Unique event identifier.
        session_id: Owning session.
        kind: Discriminator for routing.
        payload: The concrete event data.
        timestamp: When the event was created.
        sequence: Monotonic ordering index within the session.
    """

    event_id: str
    session_id: str
    kind: EventKind
    payload: EventPayload
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    sequence: int = 0


# ── Session ──────────────────────────────────────────────────────────

# Valid transitions: mapping from current state to set of allowed next states.
VALID_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.INITIALIZED: frozenset(
        {SessionState.ACTIVE, SessionState.CLOSED, SessionState.FAILED}
    ),
    SessionState.ACTIVE: frozenset(
        {
            SessionState.INTERRUPTED,
            SessionState.WAITING_ON_TOOL,
            SessionState.COMPLETED,
            SessionState.FAILED,
            SessionState.CLOSED,
        }
    ),
    SessionState.INTERRUPTED: frozenset(
        {SessionState.ACTIVE, SessionState.FAILED, SessionState.CLOSED}
    ),
    SessionState.WAITING_ON_TOOL: frozenset(
        {SessionState.ACTIVE, SessionState.FAILED, SessionState.CLOSED}
    ),
    SessionState.COMPLETED: frozenset({SessionState.CLOSED}),
    SessionState.FAILED: frozenset({SessionState.CLOSED}),
    SessionState.CLOSED: frozenset(),
}


@dataclass(slots=True)
class RealtimeSession:
    """Mutable representation of an active realtime session.

    Attributes:
        session_id: Unique session identifier.
        state: Current lifecycle state.
        config: Session configuration.
        created_at: When the session was created.
        event_log: Ordered list of recorded events.
        metadata: Arbitrary session metadata.
    """

    session_id: str
    state: SessionState = SessionState.INITIALIZED
    config: RealtimeConfig = field(default_factory=RealtimeConfig)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    event_log: list[EventEnvelope] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    _sequence_counter: int = field(default=0, repr=False)

    def next_sequence(self) -> int:
        """Return and increment the monotonic sequence counter."""
        seq = self._sequence_counter
        self._sequence_counter += 1
        return seq

    def can_transition_to(self, target: SessionState) -> bool:
        """Check whether a transition to *target* is allowed."""
        return target in VALID_TRANSITIONS.get(self.state, frozenset())
