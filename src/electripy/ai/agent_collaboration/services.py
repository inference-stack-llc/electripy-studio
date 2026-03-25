"""Services and orchestration for deterministic multi-agent collaboration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from electripy.ai.policy_gateway import PolicyAction, PolicyGateway

from .domain import AgentMessage, CollaborationRunResult, CollaborationTask
from .errors import HopLimitExceededError, UnknownAgentError
from .ports import CollaborationAgentPort


@dataclass(slots=True)
class CollaborationRuntimeSettings:
    """Runtime settings for collaboration orchestration."""

    max_hops: int = 12
    fail_on_blocked_handoff: bool = True


class AgentCollaborationRuntime:
    """Orchestrate deterministic message handoffs across specialist agents."""

    def __init__(
        self,
        *,
        agents: dict[str, CollaborationAgentPort],
        settings: CollaborationRuntimeSettings | None = None,
        policy_gateway: PolicyGateway | None = None,
    ) -> None:
        if not agents:
            raise ValueError("agents must not be empty")
        self._agents = dict(agents)
        self._settings = settings or CollaborationRuntimeSettings()
        self._policy_gateway = policy_gateway

    def run(
        self,
        *,
        task: CollaborationTask,
        entry_agent: str,
        input_text: str,
        trace_id: str | None = None,
    ) -> CollaborationRunResult:
        """Execute collaboration until completion, exhaustion, or hop limit."""

        initial = AgentMessage(
            message_id=f"{task.task_id}:0",
            task_id=task.task_id,
            from_agent="caller",
            to_agent=entry_agent,
            content=input_text,
            trace_id=trace_id,
        )

        queue: deque[AgentMessage] = deque([initial])
        transcript: list[AgentMessage] = []
        hops = 0

        while queue:
            if hops >= self._settings.max_hops:
                raise HopLimitExceededError(
                    f"Collaboration exceeded max_hops={self._settings.max_hops}"
                )

            current = queue.popleft()
            transcript.append(current)
            hops += 1

            agent = self._agents.get(current.to_agent)
            if agent is None:
                raise UnknownAgentError(f"Unknown target agent: {current.to_agent!r}")

            if self._policy_gateway is not None:
                inbound = self._policy_gateway.evaluate_preflight(current.content)
                if inbound.action in (PolicyAction.DENY, PolicyAction.REQUIRE_APPROVAL):
                    return CollaborationRunResult(
                        task_id=task.task_id,
                        success=False,
                        terminal_status="blocked_inbound",
                        hop_count=hops,
                        transcript=transcript,
                    )
                if inbound.action == PolicyAction.SANITIZE and inbound.sanitized_text is not None:
                    current = AgentMessage(
                        message_id=current.message_id,
                        task_id=current.task_id,
                        from_agent=current.from_agent,
                        to_agent=current.to_agent,
                        content=inbound.sanitized_text,
                        trace_id=current.trace_id,
                        metadata=dict(current.metadata),
                        created_at=current.created_at,
                    )
                    transcript[-1] = current

            turn = agent.handle(current, task=task)
            if turn.completed:
                for outbound in self._sorted_messages(turn.produced_messages):
                    transcript.append(outbound)
                return CollaborationRunResult(
                    task_id=task.task_id,
                    success=True,
                    terminal_status=turn.outcome or "completed",
                    hop_count=hops,
                    transcript=transcript,
                )

            for outbound in self._sorted_messages(turn.produced_messages):
                reviewed = self._review_handoff(outbound)
                if reviewed is None:
                    if self._settings.fail_on_blocked_handoff:
                        transcript.append(outbound)
                        return CollaborationRunResult(
                            task_id=task.task_id,
                            success=False,
                            terminal_status="blocked_handoff",
                            hop_count=hops,
                            transcript=transcript,
                        )
                    continue
                queue.append(reviewed)

        return CollaborationRunResult(
            task_id=task.task_id,
            success=False,
            terminal_status="exhausted",
            hop_count=hops,
            transcript=transcript,
        )

    def _review_handoff(self, message: AgentMessage) -> AgentMessage | None:
        if self._policy_gateway is None:
            return message

        decision = self._policy_gateway.evaluate_postflight(message.content)
        if decision.action in (PolicyAction.DENY, PolicyAction.REQUIRE_APPROVAL):
            return None
        if decision.action == PolicyAction.SANITIZE and decision.sanitized_text is not None:
            return AgentMessage(
                message_id=message.message_id,
                task_id=message.task_id,
                from_agent=message.from_agent,
                to_agent=message.to_agent,
                content=decision.sanitized_text,
                trace_id=message.trace_id,
                metadata=dict(message.metadata),
                created_at=message.created_at,
            )
        return message

    @staticmethod
    def _sorted_messages(messages: list[AgentMessage]) -> list[AgentMessage]:
        return sorted(messages, key=lambda msg: (msg.created_at.isoformat(), msg.message_id))


def make_message(
    *,
    task_id: str,
    seq: int,
    from_agent: str,
    to_agent: str,
    content: str,
    trace_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> AgentMessage:
    """Build a deterministic collaboration message."""

    return AgentMessage(
        message_id=f"{task_id}:{seq}",
        task_id=task_id,
        from_agent=from_agent,
        to_agent=to_agent,
        content=content,
        trace_id=trace_id,
        metadata=dict(metadata or {}),
    )
