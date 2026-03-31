"""Tests for realtime domain models and error hierarchy."""

from __future__ import annotations

from datetime import datetime

import pytest

from electripy.ai.realtime import (
    VALID_TRANSITIONS,
    BackpressureDirective,
    ChunkStatus,
    EventEnvelope,
    EventKind,
    InputStreamChunk,
    InterruptEvent,
    OutputStreamChunk,
    RealtimeConfig,
    # Errors
    RealtimeError,
    RealtimeErrorPayload,
    RealtimeSession,
    SessionLifecycleEvent,
    SessionNotFoundError,
    SessionState,
    SessionStateError,
    ToolCallEvent,
    ToolExecutionError,
    ToolResultEvent,
    TransportError,
)
from electripy.core.errors import ElectriPyError

# ── SessionState ─────────────────────────────────────────────────────


class TestSessionState:
    def test_all_values(self) -> None:
        expected = {
            "initialized",
            "active",
            "interrupted",
            "waiting_on_tool",
            "completed",
            "failed",
            "closed",
        }
        assert {s.value for s in SessionState} == expected

    def test_str_value(self) -> None:
        assert str(SessionState.ACTIVE) == "active"


# ── EventKind ────────────────────────────────────────────────────────


class TestEventKind:
    def test_all_kinds(self) -> None:
        expected = {
            "input_text",
            "input_audio",
            "output_text",
            "output_audio",
            "tool_call",
            "tool_result",
            "interrupt",
            "lifecycle",
            "error",
            "backpressure",
        }
        assert {k.value for k in EventKind} == expected


# ── ChunkStatus ──────────────────────────────────────────────────────


class TestChunkStatus:
    def test_values(self) -> None:
        assert ChunkStatus.PARTIAL.value == "partial"
        assert ChunkStatus.FINAL.value == "final"


# ── RealtimeConfig ───────────────────────────────────────────────────


class TestRealtimeConfig:
    def test_defaults(self) -> None:
        cfg = RealtimeConfig()
        assert cfg.model == ""
        assert cfg.timeout_seconds == 300.0
        assert cfg.max_events == 0

    def test_custom(self) -> None:
        cfg = RealtimeConfig(model="gpt-4o", timeout_seconds=60.0)
        assert cfg.model == "gpt-4o"

    def test_frozen(self) -> None:
        cfg = RealtimeConfig()
        with pytest.raises(AttributeError):
            cfg.model = "changed"  # type: ignore[misc]


# ── InputStreamChunk ─────────────────────────────────────────────────


class TestInputStreamChunk:
    def test_text_chunk(self) -> None:
        c = InputStreamChunk(index=0, text="hello")
        assert c.text == "hello"
        assert c.audio_bytes == b""
        assert c.status == ChunkStatus.PARTIAL

    def test_audio_chunk(self) -> None:
        c = InputStreamChunk(index=1, audio_bytes=b"\x00\xff")
        assert c.audio_bytes == b"\x00\xff"
        assert c.text == ""

    def test_final_chunk(self) -> None:
        c = InputStreamChunk(index=2, text="end", status=ChunkStatus.FINAL)
        assert c.status == ChunkStatus.FINAL


# ── OutputStreamChunk ────────────────────────────────────────────────


class TestOutputStreamChunk:
    def test_creation(self) -> None:
        c = OutputStreamChunk(index=0, text="world")
        assert c.index == 0
        assert c.text == "world"

    def test_frozen(self) -> None:
        c = OutputStreamChunk(index=0, text="x")
        with pytest.raises(AttributeError):
            c.text = "y"  # type: ignore[misc]


# ── ToolCallEvent ────────────────────────────────────────────────────


class TestToolCallEvent:
    def test_creation(self) -> None:
        tc = ToolCallEvent(call_id="c1", tool_name="search", arguments={"q": "test"})
        assert tc.call_id == "c1"
        assert tc.tool_name == "search"
        assert tc.arguments == {"q": "test"}


# ── ToolResultEvent ──────────────────────────────────────────────────


class TestToolResultEvent:
    def test_success(self) -> None:
        tr = ToolResultEvent(call_id="c1", tool_name="search", result={"hits": 3})
        assert tr.error == ""

    def test_error(self) -> None:
        tr = ToolResultEvent(call_id="c1", tool_name="search", error="timeout")
        assert tr.error == "timeout"


# ── InterruptEvent ───────────────────────────────────────────────────


class TestInterruptEvent:
    def test_defaults(self) -> None:
        ie = InterruptEvent()
        assert ie.reason == ""
        assert ie.hard is False

    def test_hard_interrupt(self) -> None:
        ie = InterruptEvent(reason="user cancel", hard=True)
        assert ie.hard is True


# ── SessionLifecycleEvent ────────────────────────────────────────────


class TestSessionLifecycleEvent:
    def test_creation(self) -> None:
        le = SessionLifecycleEvent(
            previous_state=SessionState.INITIALIZED,
            new_state=SessionState.ACTIVE,
            reason="started",
        )
        assert le.previous_state == SessionState.INITIALIZED
        assert le.new_state == SessionState.ACTIVE


# ── BackpressureDirective ────────────────────────────────────────────


class TestBackpressureDirective:
    def test_defaults(self) -> None:
        bp = BackpressureDirective()
        assert bp.queue_depth == 0
        assert bp.recommended_delay_ms == 100

    def test_custom(self) -> None:
        bp = BackpressureDirective(queue_depth=50, recommended_delay_ms=500)
        assert bp.queue_depth == 50


# ── RealtimeErrorPayload ────────────────────────────────────────────


class TestRealtimeErrorPayload:
    def test_recoverable(self) -> None:
        ep = RealtimeErrorPayload(code="RATE_LIMIT", message="too fast")
        assert ep.recoverable is True

    def test_not_recoverable(self) -> None:
        ep = RealtimeErrorPayload(code="FATAL", message="boom", recoverable=False)
        assert ep.recoverable is False


# ── EventEnvelope ────────────────────────────────────────────────────


class TestEventEnvelope:
    def test_creation(self) -> None:
        env = EventEnvelope(
            event_id="e1",
            session_id="s1",
            kind=EventKind.INPUT_TEXT,
            payload=InputStreamChunk(index=0, text="hi"),
            sequence=0,
        )
        assert env.kind == EventKind.INPUT_TEXT
        assert env.sequence == 0

    def test_has_timestamp(self) -> None:
        env = EventEnvelope(
            event_id="e1",
            session_id="s1",
            kind=EventKind.OUTPUT_TEXT,
            payload=OutputStreamChunk(index=0, text="out"),
        )
        assert isinstance(env.timestamp, datetime)


# ── RealtimeSession ─────────────────────────────────────────────────


class TestRealtimeSession:
    def test_initial_state(self) -> None:
        s = RealtimeSession(session_id="s1")
        assert s.state == SessionState.INITIALIZED

    def test_can_transition_to(self) -> None:
        s = RealtimeSession(session_id="s1")
        assert s.can_transition_to(SessionState.ACTIVE)
        assert not s.can_transition_to(SessionState.COMPLETED)

    def test_next_sequence(self) -> None:
        s = RealtimeSession(session_id="s1")
        assert s.next_sequence() == 0
        assert s.next_sequence() == 1
        assert s.next_sequence() == 2

    def test_mutable(self) -> None:
        s = RealtimeSession(session_id="s1")
        s.state = SessionState.ACTIVE
        assert s.state == SessionState.ACTIVE


# ── VALID_TRANSITIONS ────────────────────────────────────────────────


class TestValidTransitions:
    def test_closed_is_terminal(self) -> None:
        assert VALID_TRANSITIONS[SessionState.CLOSED] == frozenset()

    def test_initialized_can_start(self) -> None:
        assert SessionState.ACTIVE in VALID_TRANSITIONS[SessionState.INITIALIZED]

    def test_all_states_have_entries(self) -> None:
        for state in SessionState:
            assert state in VALID_TRANSITIONS


# ── Error hierarchy ──────────────────────────────────────────────────


class TestErrorHierarchy:
    def test_base_inherits_electripy_error(self) -> None:
        assert issubclass(RealtimeError, ElectriPyError)

    def test_all_inherit_base(self) -> None:
        for cls in (SessionStateError, SessionNotFoundError, TransportError, ToolExecutionError):
            assert issubclass(cls, RealtimeError)

    def test_session_state_error_message(self) -> None:
        err = SessionStateError("initialized", "completed")
        assert "initialized" in str(err)
        assert "completed" in str(err)

    def test_tool_execution_error_message(self) -> None:
        err = ToolExecutionError("search", "timeout")
        assert "search" in str(err)
        assert "timeout" in str(err)

    def test_catchable_as_electripy(self) -> None:
        with pytest.raises(ElectriPyError):
            raise SessionNotFoundError("gone")
