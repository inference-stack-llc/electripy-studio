# AI Agent Collaboration Runtime

The Agent Collaboration Runtime orchestrates bounded, deterministic handoffs between specialist agents.

## Why it exists

As AI systems move from single-agent flows to specialist-agent teams, reliability depends on explicit message contracts and hop limits. This runtime coordinates those handoffs in-process and works with the Policy Gateway for safety.

## Core concepts

- `CollaborationTask`: top-level objective and metadata.
- `AgentMessage`: typed message envelope between agents.
- `CollaborationAgentPort`: handler protocol each agent implements.
- `AgentCollaborationRuntime`: deterministic orchestration service.

## Quick example

```python
from electripy.ai.agent_collaboration import (
    AgentCollaborationRuntime,
    AgentTurnResult,
    CollaborationTask,
    make_message,
)

class PlannerAgent:
    def handle(self, message, *, task):
        return AgentTurnResult(
            produced_messages=[
                make_message(
                    task_id=task.task_id,
                    seq=1,
                    from_agent="planner",
                    to_agent="verifier",
                    content="plan ready",
                )
            ]
        )

class VerifierAgent:
    def handle(self, message, *, task):
        return AgentTurnResult(completed=True, outcome="verified")

runtime = AgentCollaborationRuntime(
    agents={"planner": PlannerAgent(), "verifier": VerifierAgent()}
)
result = runtime.run(
    task=CollaborationTask(task_id="incident-1", objective="triage outage"),
    entry_agent="planner",
    input_text="begin",
)
```

## Reliability guardrails

- Deterministic message ordering.
- Configurable max-hop limits.
- Optional policy checks on inbound/outbound handoffs.
- Full transcript output for replay and debugging.
