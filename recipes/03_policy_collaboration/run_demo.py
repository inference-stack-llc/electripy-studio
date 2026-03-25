"""End-to-end demo: policy gateway + llm gateway hooks + agent collaboration.

This script runs fully offline using fake adapters and agents.
"""

from __future__ import annotations

from dataclasses import dataclass

from electripy.ai.agent_collaboration import (
    AgentCollaborationRuntime,
    AgentTurnResult,
    CollaborationTask,
    make_message,
)
from electripy.ai.llm_gateway import (
    LlmGatewaySettings,
    LlmGatewaySyncClient,
    LlmMessage,
    LlmRequest,
    LlmResponse,
)
from electripy.ai.llm_gateway.ports import SyncLlmPort
from electripy.ai.policy_gateway import (
    PolicyAction,
    PolicyGateway,
    PolicyRule,
    PolicySeverity,
    PolicyStage,
    build_llm_policy_hooks,
)
from electripy.observability.ai_telemetry import InMemoryTelemetryAdapter
from electripy.observability.ai_telemetry.services import record_policy_decision


@dataclass
class FakeSyncPort(SyncLlmPort):
    """Fake LLM port for deterministic offline demo responses."""

    text: str = "Summary: keep user email private."

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        del request, timeout
        return LlmResponse(text=self.text, model="fake-model")


class PlannerAgent:
    def handle(self, message, *, task):
        return AgentTurnResult(
            produced_messages=[
                make_message(
                    task_id=task.task_id,
                    seq=1,
                    from_agent="planner",
                    to_agent="verifier",
                    content=f"plan::{task.objective}::{message.content}",
                )
            ]
        )


class VerifierAgent:
    def handle(self, message, *, task):
        del message, task
        return AgentTurnResult(completed=True, outcome="verified")


def main() -> None:
    telemetry = InMemoryTelemetryAdapter()

    policy = PolicyGateway(
        rules=[
            PolicyRule(
                rule_id="email-redact",
                code="PII_EMAIL",
                description="Redact email addresses.",
                stage=PolicyStage.PREFLIGHT,
                pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
                action=PolicyAction.SANITIZE,
                severity=PolicySeverity.MEDIUM,
            ),
            PolicyRule(
                rule_id="postflight-secret",
                code="SECRET_LEAK",
                description="Block leaked secret markers in output.",
                stage=PolicyStage.POSTFLIGHT,
                pattern=r"SECRET_[A-Z0-9]+",
                action=PolicyAction.DENY,
                severity=PolicySeverity.CRITICAL,
            ),
        ],
        telemetry=telemetry,
    )

    request_hook, response_hook = build_llm_policy_hooks(policy)
    settings = LlmGatewaySettings(
        request_hook=request_hook,
        response_hook=response_hook,
    )

    client = LlmGatewaySyncClient(port=FakeSyncPort(), settings=settings)
    response = client.complete(
        LlmRequest(
            model="fake-model",
            messages=[LlmMessage.user("Summarize for admin@example.com")],
        )
    )

    runtime = AgentCollaborationRuntime(
        agents={"planner": PlannerAgent(), "verifier": VerifierAgent()},
        policy_gateway=policy,
    )
    collab = runtime.run(
        task=CollaborationTask(task_id="demo-1", objective="incident update"),
        entry_agent="planner",
        input_text=response.text,
    )

    record_policy_decision(
        telemetry,
        decision="allow" if collab.success else "deny",
        violation_codes=[],
        redactions_applied=False,
    )

    print("LLM response:", response.text)
    print("Collaboration:", collab.terminal_status, "hops=", collab.hop_count)
    print("Telemetry events:", len(telemetry.events))


if __name__ == "__main__":
    main()
