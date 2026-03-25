"""Services for deterministic tool-plan execution."""

from __future__ import annotations

from .domain import AgentRunResult, AgentStepResult, ToolInvocation
from .ports import ToolPort


class AgentExecutor:
    """Execute ordered tool invocation plans with bounded retries."""

    def __init__(self, *, tool_port: ToolPort, max_attempts: int = 2) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self._tool_port = tool_port
        self._max_attempts = max_attempts

    def run(self, plan: list[ToolInvocation]) -> AgentRunResult:
        """Run a plan in order and stop on first persistent failure."""

        results: list[AgentStepResult] = []
        for step in plan:
            attempts = 0
            last_error = ""
            while attempts < self._max_attempts:
                attempts += 1
                try:
                    output = self._tool_port.execute(step.name, step.args)
                    results.append(
                        AgentStepResult(
                            tool_name=step.name,
                            success=True,
                            output=output,
                            attempts=attempts,
                        )
                    )
                    break
                except Exception as exc:  # pragma: no cover - defensive conversion
                    last_error = str(exc)
            else:
                results.append(
                    AgentStepResult(
                        tool_name=step.name,
                        success=False,
                        output=last_error,
                        attempts=attempts,
                    )
                )
                return AgentRunResult(steps=results)

        return AgentRunResult(steps=results)
