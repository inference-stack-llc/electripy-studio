"""Domain models for deterministic multi-agent collaboration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class CollaborationTask:
    """Top-level collaboration task for a run."""

    task_id: str
    objective: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AgentMessage:
    """Message exchanged between collaborating agents."""

    message_id: str
    task_id: str
    from_agent: str
    to_agent: str
    content: str
    trace_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass(slots=True)
class AgentTurnResult:
    """Output of one agent turn within collaboration runtime."""

    produced_messages: list[AgentMessage] = field(default_factory=list)
    completed: bool = False
    outcome: str | None = None


@dataclass(slots=True)
class CollaborationRunResult:
    """Final result summary for a collaboration run."""

    task_id: str
    success: bool
    terminal_status: str
    hop_count: int
    transcript: list[AgentMessage] = field(default_factory=list)
