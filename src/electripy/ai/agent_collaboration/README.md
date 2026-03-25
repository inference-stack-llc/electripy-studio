# Agent Collaboration Runtime

Deterministic in-process orchestration for agent-to-agent handoff workflows.

## What it solves

- Coordinates specialist agents (planner/retriever/executor/verifier) with explicit message contracts.
- Enforces bounded hops to avoid runaway loops.
- Supports optional policy-gateway checks on inbound and outbound messages.
- Produces a deterministic transcript for replay/debugging.

## Quick Start

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
                    content=f"plan for: {task.objective}",
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
    task=CollaborationTask(task_id="t1", objective="draft incident checklist"),
    entry_agent="planner",
    input_text="start",
)
```
