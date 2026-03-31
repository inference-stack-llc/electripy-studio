"""Shared fixtures for realtime tests."""

from __future__ import annotations

import pytest

from electripy.ai.realtime import (
    InMemoryObserver,
    InMemorySessionStore,
    RealtimeConfig,
    RealtimeSessionService,
)
from electripy.ai.realtime.adapters import EchoToolExecutor


@pytest.fixture()
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture()
def observer() -> InMemoryObserver:
    return InMemoryObserver()


@pytest.fixture()
def svc(store: InMemorySessionStore, observer: InMemoryObserver) -> RealtimeSessionService:
    """Service wired with in-memory store, observer, and echo tool executor."""
    return RealtimeSessionService(
        store=store,
        observer=observer,
        tool_executor=EchoToolExecutor(),
    )


@pytest.fixture()
def active_session_id(svc: RealtimeSessionService) -> str:
    """Create and start a session, returning its ID."""
    session = svc.create_session(config=RealtimeConfig(model="test-model"))
    svc.start_session(session.session_id)
    return session.session_id
