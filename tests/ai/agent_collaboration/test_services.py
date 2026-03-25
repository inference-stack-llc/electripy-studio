from __future__ import annotations

import pytest

from electripy.ai.agent_collaboration import (
    AgentCollaborationRuntime,
    AgentTurnResult,
    CollaborationTask,
    HopLimitExceededError,
    make_message,
)
from electripy.ai.policy_gateway import PolicyAction, PolicyGateway, PolicyRule, PolicyStage


class PlannerAgent:
    def handle(self, message, *, task):
        return AgentTurnResult(
            produced_messages=[
                make_message(
                    task_id=task.task_id,
                    seq=1,
                    from_agent="planner",
                    to_agent="verifier",
                    content=f"plan::{message.content}",
                )
            ]
        )


class VerifierAgent:
    def handle(self, message, *, task):
        return AgentTurnResult(completed=True, outcome="verified")


class LoopAgent:
    def handle(self, message, *, task):
        return AgentTurnResult(
            produced_messages=[
                make_message(
                    task_id=task.task_id,
                    seq=999,
                    from_agent="loop",
                    to_agent="loop",
                    content="again",
                )
            ]
        )


def test_agent_collaboration_runtime_completes_successfully() -> None:
    runtime = AgentCollaborationRuntime(
        agents={"planner": PlannerAgent(), "verifier": VerifierAgent()}
    )

    result = runtime.run(
        task=CollaborationTask(task_id="task-1", objective="create response"),
        entry_agent="planner",
        input_text="start",
    )

    assert result.success is True
    assert result.terminal_status == "verified"
    assert result.hop_count == 2


def test_agent_collaboration_runtime_enforces_hop_limit() -> None:
    runtime = AgentCollaborationRuntime(agents={"loop": LoopAgent()})

    with pytest.raises(HopLimitExceededError):
        runtime.run(
            task=CollaborationTask(task_id="task-loop", objective="loop"),
            entry_agent="loop",
            input_text="go",
        )


def test_agent_collaboration_runtime_blocks_handoff_via_policy_gateway() -> None:
    gateway = PolicyGateway(
        rules=[
            PolicyRule(
                rule_id="ban-plan",
                code="PLAN_BLOCKED",
                description="Block plan content",
                stage=PolicyStage.POSTFLIGHT,
                pattern=r"plan::",
                action=PolicyAction.DENY,
            )
        ]
    )
    runtime = AgentCollaborationRuntime(
        agents={"planner": PlannerAgent(), "verifier": VerifierAgent()},
        policy_gateway=gateway,
    )

    result = runtime.run(
        task=CollaborationTask(task_id="task-2", objective="create response"),
        entry_agent="planner",
        input_text="start",
    )

    assert result.success is False
    assert result.terminal_status == "blocked_handoff"
