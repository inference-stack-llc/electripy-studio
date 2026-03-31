"""Tests for RealtimeSessionService and streaming helpers."""

from __future__ import annotations

import pytest

from electripy.ai.realtime import (
    BackpressureDirective,
    ChunkStatus,
    EventKind,
    InMemoryObserver,
    InMemoryTransport,
    InputStreamChunk,
    OutputStreamChunk,
    RealtimeConfig,
    RealtimeError,
    RealtimeSessionService,
    SessionNotFoundError,
    SessionState,
    SessionStateError,
    ToolCallEvent,
    ToolExecutionError,
    async_collect_output_text,
    async_iter_output_text,
    collect_output_text,
    iter_output_text,
)

# ── Session lifecycle ────────────────────────────────────────────────


class TestSessionLifecycle:
    def test_create_session(self, svc: RealtimeSessionService) -> None:
        session = svc.create_session()
        assert session.state == SessionState.INITIALIZED
        assert session.session_id

    def test_create_with_config(self, svc: RealtimeSessionService) -> None:
        session = svc.create_session(config=RealtimeConfig(model="gpt-4o"))
        assert session.config.model == "gpt-4o"

    def test_start_session(self, svc: RealtimeSessionService) -> None:
        session = svc.create_session()
        started = svc.start_session(session.session_id)
        assert started.state == SessionState.ACTIVE

    def test_start_already_active_raises(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        with pytest.raises(SessionStateError, match="active.*active"):
            svc.start_session(active_session_id)

    def test_complete_session(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        session = svc.complete_session(active_session_id)
        assert session.state == SessionState.COMPLETED

    def test_fail_session(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        session = svc.fail_session(active_session_id, reason="oom")
        assert session.state == SessionState.FAILED

    def test_close_session(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        session = svc.close_session(active_session_id)
        assert session.state == SessionState.CLOSED

    def test_close_completed(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        svc.complete_session(active_session_id)
        session = svc.close_session(active_session_id)
        assert session.state == SessionState.CLOSED

    def test_close_failed(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        svc.fail_session(active_session_id)
        session = svc.close_session(active_session_id)
        assert session.state == SessionState.CLOSED

    def test_double_close_raises(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        svc.close_session(active_session_id)
        with pytest.raises(SessionStateError, match="closed"):
            svc.close_session(active_session_id)

    def test_get_session(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        session = svc.get_session(active_session_id)
        assert session.session_id == active_session_id

    def test_get_session_not_found(self, svc: RealtimeSessionService) -> None:
        with pytest.raises(SessionNotFoundError, match="not found"):
            svc.get_session("nonexistent")


# ── State transition guards ──────────────────────────────────────────


class TestStateTransitionGuards:
    def test_initialized_cannot_complete(self, svc: RealtimeSessionService) -> None:
        session = svc.create_session()
        with pytest.raises(SessionStateError):
            svc.complete_session(session.session_id)

    def test_completed_cannot_start(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        svc.complete_session(active_session_id)
        with pytest.raises(SessionStateError):
            svc.start_session(active_session_id)

    def test_closed_cannot_transition(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        svc.close_session(active_session_id)
        with pytest.raises(SessionStateError):
            svc.start_session(active_session_id)
        with pytest.raises(SessionStateError):
            svc.fail_session(active_session_id)


# ── Event ingestion ──────────────────────────────────────────────────


class TestEventIngestion:
    def test_ingest_input_text(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        chunk = InputStreamChunk(index=0, text="hello")
        env = svc.ingest_event(active_session_id, EventKind.INPUT_TEXT, chunk)
        assert env.kind == EventKind.INPUT_TEXT
        assert env.sequence >= 1  # after lifecycle event from start

    def test_ingest_requires_active(self, svc: RealtimeSessionService) -> None:
        session = svc.create_session()
        chunk = InputStreamChunk(index=0, text="hello")
        with pytest.raises(SessionStateError, match="active"):
            svc.ingest_event(session.session_id, EventKind.INPUT_TEXT, chunk)

    def test_events_are_sequenced(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        for i in range(5):
            svc.ingest_event(
                active_session_id,
                EventKind.INPUT_TEXT,
                InputStreamChunk(index=i, text=f"chunk-{i}"),
            )
        session = svc.get_session(active_session_id)
        sequences = [e.sequence for e in session.event_log]
        assert sequences == sorted(sequences)

    def test_emit_output_text(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        chunk = OutputStreamChunk(index=0, text="response")
        env = svc.emit_output(active_session_id, chunk)
        assert env.kind == EventKind.OUTPUT_TEXT

    def test_emit_output_audio(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        chunk = OutputStreamChunk(index=0, audio_bytes=b"\x00\x01")
        env = svc.emit_output(active_session_id, chunk)
        assert env.kind == EventKind.OUTPUT_AUDIO


# ── Observer integration ─────────────────────────────────────────────


class TestObserverIntegration:
    def test_observer_notified_on_state_change(
        self, svc: RealtimeSessionService, observer: InMemoryObserver
    ) -> None:
        session = svc.create_session()
        svc.start_session(session.session_id)
        # Should capture initialized->active
        found = [
            (prev, cur) for sid, prev, cur in observer.state_changes if sid == session.session_id
        ]
        assert (SessionState.INITIALIZED, SessionState.ACTIVE) in found

    def test_observer_notified_on_event(
        self, svc: RealtimeSessionService, active_session_id: str, observer: InMemoryObserver
    ) -> None:
        svc.ingest_event(
            active_session_id,
            EventKind.INPUT_TEXT,
            InputStreamChunk(index=0, text="hi"),
        )
        kinds = [e.kind for e in observer.events]
        assert EventKind.INPUT_TEXT in kinds

    def test_observer_not_required(self) -> None:
        svc = RealtimeSessionService()
        session = svc.create_session()
        svc.start_session(session.session_id)  # should not raise


# ── Interruption ─────────────────────────────────────────────────────


class TestInterruption:
    def test_interrupt_active_session(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        env = svc.interrupt(active_session_id, reason="user cancel")
        assert env.kind == EventKind.INTERRUPT
        session = svc.get_session(active_session_id)
        assert session.state == SessionState.INTERRUPTED

    def test_resume_interrupted(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        svc.interrupt(active_session_id)
        session = svc.resume(active_session_id)
        assert session.state == SessionState.ACTIVE

    def test_cannot_interrupt_initialized(self, svc: RealtimeSessionService) -> None:
        session = svc.create_session()
        with pytest.raises(SessionStateError):
            svc.interrupt(session.session_id)

    def test_interrupt_records_reason(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        svc.interrupt(active_session_id, reason="too slow", hard=True)
        session = svc.get_session(active_session_id)
        interrupt_events = [e for e in session.event_log if e.kind == EventKind.INTERRUPT]
        assert len(interrupt_events) == 1
        assert interrupt_events[0].payload.hard is True  # type: ignore[union-attr]


# ── Tool calls ───────────────────────────────────────────────────────


class TestToolCalls:
    async def test_tool_call_roundtrip(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        call = ToolCallEvent(call_id="c1", tool_name="echo", arguments={"q": "test"})
        result = await svc.handle_tool_call(active_session_id, call)
        assert result.call_id == "c1"
        assert result.result == {"q": "test"}
        # Session should return to active.
        session = svc.get_session(active_session_id)
        assert session.state == SessionState.ACTIVE

    async def test_tool_call_records_events(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        call = ToolCallEvent(call_id="c2", tool_name="echo", arguments={"x": 1})
        await svc.handle_tool_call(active_session_id, call)
        session = svc.get_session(active_session_id)
        kinds = [e.kind for e in session.event_log]
        assert EventKind.TOOL_CALL in kinds
        assert EventKind.TOOL_RESULT in kinds

    async def test_tool_call_no_executor_raises(self, active_session_id: str) -> None:
        svc = RealtimeSessionService()
        session = svc.create_session()
        svc.start_session(session.session_id)
        call = ToolCallEvent(call_id="c1", tool_name="x")
        with pytest.raises(RealtimeError, match="No tool executor"):
            await svc.handle_tool_call(session.session_id, call)

    async def test_tool_execution_failure(self, active_session_id: str) -> None:
        class FailingExecutor:
            async def execute(self, event: ToolCallEvent) -> None:
                raise RuntimeError("tool broke")

        svc = RealtimeSessionService(tool_executor=FailingExecutor())  # type: ignore[arg-type]
        session = svc.create_session()
        svc.start_session(session.session_id)
        call = ToolCallEvent(call_id="c1", tool_name="broken")
        with pytest.raises(ToolExecutionError, match="broken"):
            await svc.handle_tool_call(session.session_id, call)
        # Session should be FAILED.
        s = svc.get_session(session.session_id)
        assert s.state == SessionState.FAILED


# ── Backpressure ─────────────────────────────────────────────────────


class TestBackpressure:
    def test_emit_backpressure(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        bp = BackpressureDirective(queue_depth=100, recommended_delay_ms=500)
        env = svc.emit_backpressure(active_session_id, bp)
        assert env.kind == EventKind.BACKPRESSURE


# ── Event replay ─────────────────────────────────────────────────────


class TestEventReplay:
    def test_replay_all(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        svc.ingest_event(
            active_session_id, EventKind.INPUT_TEXT, InputStreamChunk(index=0, text="a")
        )
        svc.ingest_event(
            active_session_id, EventKind.OUTPUT_TEXT, OutputStreamChunk(index=0, text="b")
        )
        events = svc.replay_events(active_session_id)
        assert len(events) >= 2

    def test_replay_filtered(self, svc: RealtimeSessionService, active_session_id: str) -> None:
        svc.ingest_event(
            active_session_id, EventKind.INPUT_TEXT, InputStreamChunk(index=0, text="a")
        )
        svc.ingest_event(
            active_session_id, EventKind.OUTPUT_TEXT, OutputStreamChunk(index=0, text="b")
        )
        filtered = svc.replay_events(
            active_session_id,
            kinds=frozenset({EventKind.OUTPUT_TEXT}),
        )
        for e in filtered:
            assert e.kind == EventKind.OUTPUT_TEXT


# ── Transport integration ────────────────────────────────────────────


class TestTransportIntegration:
    async def test_send_to_transport(self) -> None:
        transport = InMemoryTransport()
        svc = RealtimeSessionService(transport=transport)
        session = svc.create_session()
        svc.start_session(session.session_id)

        chunk = OutputStreamChunk(index=0, text="hi")
        env = svc.emit_output(session.session_id, chunk)
        await svc.send_to_transport(session.session_id, env)

        outbound = transport.drain_outbound()
        assert len(outbound) == 1
        assert outbound[0].event_id == env.event_id

    async def test_send_without_transport_is_noop(
        self, svc: RealtimeSessionService, active_session_id: str
    ) -> None:
        chunk = OutputStreamChunk(index=0, text="hi")
        env = svc.emit_output(active_session_id, chunk)
        # Should not raise when no transport is set.
        await svc.send_to_transport(active_session_id, env)


# ── Streaming helpers ────────────────────────────────────────────────


class TestStreamingHelpers:
    def test_iter_output_text(self) -> None:
        chunks = [
            OutputStreamChunk(index=0, text="Hello"),
            OutputStreamChunk(index=1, text=" "),
            OutputStreamChunk(index=2, text="world"),
        ]
        assert list(iter_output_text(chunks)) == ["Hello", " ", "world"]

    def test_collect_output_text(self) -> None:
        chunks = [
            OutputStreamChunk(index=0, text="A"),
            OutputStreamChunk(index=1, text="B"),
            OutputStreamChunk(index=2, text="C", status=ChunkStatus.FINAL),
        ]
        assert collect_output_text(chunks) == "ABC"

    def test_empty_text_skipped(self) -> None:
        chunks = [
            OutputStreamChunk(index=0, text=""),
            OutputStreamChunk(index=1, audio_bytes=b"\x00"),
            OutputStreamChunk(index=2, text="hi"),
        ]
        assert list(iter_output_text(chunks)) == ["hi"]

    async def test_async_collect_output_text(self) -> None:
        async def _chunks():
            yield OutputStreamChunk(index=0, text="X")
            yield OutputStreamChunk(index=1, text="Y")

        result = await async_collect_output_text(_chunks())
        assert result == "XY"

    async def test_async_iter_output_text(self) -> None:
        async def _chunks():
            yield OutputStreamChunk(index=0, text="a")
            yield OutputStreamChunk(index=1, text="b")

        parts = []
        async for text in async_iter_output_text(_chunks()):
            parts.append(text)
        assert parts == ["a", "b"]
