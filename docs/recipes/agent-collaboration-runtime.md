# Recipe: Specialist Agent Collaboration

This recipe demonstrates a planner -> retriever -> verifier pipeline using the Agent Collaboration Runtime.

## Scenario

You want to run multiple specialist agents while keeping execution bounded and deterministic.

## Example

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
                    to_agent="retriever",
                    content=f"find evidence for: {task.objective}",
                )
            ]
        )

class RetrieverAgent:
    def handle(self, message, *, task):
        return AgentTurnResult(
            produced_messages=[
                make_message(
                    task_id=task.task_id,
                    seq=2,
                    from_agent="retriever",
                    to_agent="verifier",
                    content="evidence: runbook#42",
                )
            ]
        )

class VerifierAgent:
    def handle(self, message, *, task):
        return AgentTurnResult(completed=True, outcome="verified")

runtime = AgentCollaborationRuntime(
    agents={
        "planner": PlannerAgent(),
        "retriever": RetrieverAgent(),
        "verifier": VerifierAgent(),
    }
)
result = runtime.run(
    task=CollaborationTask(task_id="incident-7", objective="recover API service"),
    entry_agent="planner",
    input_text="start",
)

print(result.success, result.terminal_status, result.hop_count)
```

## Notes

- Keep message payloads concise and structured.
- Enforce max hops to prevent runaway loops.
- Combine with Policy Gateway for handoff safety checks.
