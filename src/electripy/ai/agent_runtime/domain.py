"""Domain models for the lightweight agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ToolInvocation:
    """Represents one tool call in a deterministic agent plan."""

    name: str
    args: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AgentStepResult:
    """Result of one tool invocation attempt."""

    tool_name: str
    success: bool
    output: str
    attempts: int


@dataclass(slots=True)
class AgentRunResult:
    """Aggregate result for a full plan execution."""

    steps: list[AgentStepResult]

    @property
    def all_successful(self) -> bool:
        """Return True when every step succeeded."""

        return all(step.success for step in self.steps)
