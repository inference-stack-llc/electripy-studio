"""Tests for realtime adapters."""

from __future__ import annotations

import asyncio

import pytest

from electripy.ai.realtime import (
    EventEnvelope,
    EventKind,
    InMemoryObserver,
    InMemorySessionStore,
    InMemoryTransport,
    InputStreamChunk,
    OutputStreamChunk,
    RealtimeSession,
    SessionState,
    ToolCallEvent,
    ToolResultEvent,
)
from electripy.ai.realtime.adapters import EchoToolExecutor, NoOpObserver


# ── InMemoryTransport ────────────────────────────────────────────────


class TestInMemoryTransport:
    async def test_send_and_drain(self) -> None:
        transport = InMemoryTransport()
        env = EventEnvelope(
            event_id="e1",
            session_id="s1",
            kind=EventKind.OUTPUT_TEXT,
            payload=OutputStreamChunk(index=0, text="hi"),
        )
        await transport.send_event(env)
        drained = transport.drain_outbound()
        assert len(drained) == 1
        assert drained[0].event_id == "e1"

    async def test_inject_and_receive(self) -> None:
        transport = InMemoryTransport()
        env = EventEnvelope(
            event_id="e1",
            session_id="s1",
            kind=EventKind.INPUT_TEXT,
            payload=InputStreamChunk(index=0, text="hello"),
        )
        await transport.inject(env)
        received = await transport.receive_event()
        assert received.event_id == "e1"

    async def test_close_prevents_send(self) -> None:
        transport = InMemoryTransport()
        await transport.close()
        assert transport.closed is True
        env = EventEnvelope(
            event_id="e1",
            session_id="s1",
            kind=EventKind.OUTPUT_TEXT,
            payload=OutputStreamChunk(index=0, text="x"),
        )
        await transport.send_event(env)
        assert transport.drain_outbound() == []

    async def test_drain_empty(self) -> None:
        transport = InMemoryTransport()
        assert transport.drain_outbound() == []


# ── InMemorySessionStore ─────────────────────────────────────────────


class TestInMemorySessionStore:
    def test_save_and_load(self) -> None:
        store = InMemorySessionStore()
        session = RealtimeSession(session_id="s1")
        store.save(session)
        loaded = store.load("s1")
        assert loaded is session

    def test_load_missing(self) -> None:
        store = InMemorySessionStore()
        assert store.load("nonexistent") is None

    def test_delete(self) -> None:
        store = InMemorySessionStore()
        session = RealtimeSession(session_id="s1")
        store.save(session)
        store.delete("s1")
        assert store.load("s1") is None

    def test_delete_missing_is_noop(self) -> None:
        store = InMemorySessionStore()
        store.delete("nonexistent")  # should not raise


# ── NoOpObserver ─────────────────────────────────────────────────────


class TestNoOpObserver:
    def test_all_methods_are_noop(self) -> None:
        obs = NoOpObserver()
        obs.on_state_change("s1", SessionState.INITIALIZED, SessionState.ACTIVE)
        env = EventEnvelope(
            event_id="e1",
            session_id="s1",
            kind=EventKind.LIFECYCLE,
            payload=InputStreamChunk(index=0),
        )
        obs.on_event(env)
        obs.on_error("s1", RuntimeError("test"))


# ── InMemoryObserver ─────────────────────────────────────────────────


class TestInMemoryObserver:
    def test_captures_state_changes(self) -> None:
        obs = InMemoryObserver()
        obs.on_state_change("s1", SessionState.INITIALIZED, SessionState.ACTIVE)
        assert len(obs.state_changes) == 1
        sid, prev, cur = obs.state_changes[0]
        assert sid == "s1"
        assert prev == SessionState.INITIALIZED
        assert cur == SessionState.ACTIVE

    def test_captures_events(self) -> None:
        obs = InMemoryObserver()
        env = EventEnvelope(
            event_id="e1",
            session_id="s1",
            kind=EventKind.OUTPUT_TEXT,
            payload=OutputStreamChunk(index=0, text="hi"),
        )
        obs.on_event(env)
        assert len(obs.events) == 1

    def test_captures_errors(self) -> None:
        obs = InMemoryObserver()
        obs.on_error("s1", ValueError("boom"))
        assert len(obs.errors) == 1
        assert obs.errors[0][0] == "s1"


# ── EchoToolExecutor ─────────────────────────────────────────────────


class TestEchoToolExecutor:
    async def test_echoes_arguments(self) -> None:
        executor = EchoToolExecutor()
        call = ToolCallEvent(call_id="c1", tool_name="echo", arguments={"x": 1})
        result = await executor.execute(call)
        assert result.call_id == "c1"
        assert result.tool_name == "echo"
        assert result.result == {"x": 1}
        assert result.error == ""
