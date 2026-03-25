"""Demo subcommands that showcase ElectriPy AI capabilities offline."""

from __future__ import annotations

from dataclasses import dataclass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from electripy.ai.agent_collaboration import (
    AgentCollaborationRuntime,
    AgentMessage,
    AgentTurnResult,
    CollaborationRunResult,
    CollaborationRuntimeSettings,
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

app = typer.Typer(
    name="demo",
    help="Run offline demos that showcase ElectriPy AI components.",
    no_args_is_help=True,
)

console = Console()


# -- Fake adapters used by the demo (no network, fully deterministic) --------


@dataclass
class _FakeLlmPort(SyncLlmPort):
    """Returns a deterministic response for the demo."""

    text: str = "Summary: keep user email private."

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        del request, timeout
        return LlmResponse(text=self.text, model="fake-model")


class _PlannerAgent:
    def handle(self, message: AgentMessage, *, task: CollaborationTask) -> AgentTurnResult:
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


class _VerifierAgent:
    def handle(self, message: AgentMessage, *, task: CollaborationTask) -> AgentTurnResult:
        del message, task
        return AgentTurnResult(completed=True, outcome="verified")


# -- The command itself -------------------------------------------------------


@app.command("policy-collab")
def policy_collab(
    user_prompt: str = typer.Option(
        "Summarize for admin@example.com",
        "--prompt",
        "-p",
        help="User prompt sent through the policy + LLM pipeline.",
    ),
    max_hops: int = typer.Option(
        12,
        "--max-hops",
        help="Maximum agent handoffs before stopping.",
    ),
) -> None:
    """Run the policy-gateway + agent-collaboration demo end-to-end.

    Uses fake adapters so no API keys or network are required.
    """
    telemetry = InMemoryTelemetryAdapter()

    # 1. Build policy gateway with two sample rules
    policy = PolicyGateway(
        rules=[
            PolicyRule(
                rule_id="email-redact",
                code="PII_EMAIL",
                description="Redact email addresses",
                stage=PolicyStage.PREFLIGHT,
                pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
                action=PolicyAction.SANITIZE,
                severity=PolicySeverity.MEDIUM,
            ),
            PolicyRule(
                rule_id="postflight-secret",
                code="SECRET_LEAK",
                description="Block leaked secret markers in output",
                stage=PolicyStage.POSTFLIGHT,
                pattern=r"SECRET_[A-Z0-9]+",
                action=PolicyAction.DENY,
                severity=PolicySeverity.CRITICAL,
            ),
        ],
        telemetry=telemetry,
    )

    # 2. Wire policy hooks into the LLM gateway
    request_hook, response_hook = build_llm_policy_hooks(policy)
    settings = LlmGatewaySettings(
        request_hook=request_hook,
        response_hook=response_hook,
    )

    client = LlmGatewaySyncClient(port=_FakeLlmPort(), settings=settings)

    # 3. Send the user prompt through the gateway
    response = client.complete(
        LlmRequest(
            model="fake-model",
            messages=[LlmMessage.user(user_prompt)],
        )
    )

    # 4. Run the agent collaboration loop
    runtime = AgentCollaborationRuntime(
        agents={"planner": _PlannerAgent(), "verifier": _VerifierAgent()},
        settings=CollaborationRuntimeSettings(max_hops=max_hops),
        policy_gateway=policy,
    )
    collab = runtime.run(
        task=CollaborationTask(task_id="demo-1", objective="incident update"),
        entry_agent="planner",
        input_text=response.text,
    )

    # -- Render the report ----------------------------------------------------
    _render_report(user_prompt, response.text, collab, telemetry)


def _render_report(
    prompt: str,
    llm_text: str,
    collab: CollaborationRunResult,
    telemetry: InMemoryTelemetryAdapter,
) -> None:
    """Print a structured Rich report to the console."""
    console.print()

    # Header
    console.print(
        Panel.fit(
            "[bold cyan]ElectriPy Demo:[/bold cyan] Policy Gateway + Agent Collaboration",
            border_style="cyan",
        )
    )

    # Pipeline table
    pipeline = Table(title="Pipeline Summary", show_header=True, header_style="bold magenta")
    pipeline.add_column("Stage", style="cyan", width=22)
    pipeline.add_column("Result", width=52)

    pipeline.add_row("User Prompt", prompt)
    pipeline.add_row("LLM Response", llm_text)
    pipeline.add_row(
        "Collaboration Status",
        (
            f"[green]{collab.terminal_status}[/green]"
            if collab.success
            else f"[red]{collab.terminal_status}[/red]"
        ),
    )
    pipeline.add_row("Hops", str(collab.hop_count))
    pipeline.add_row("Telemetry Events", str(len(telemetry.events)))

    console.print(pipeline)

    # Transcript table
    if collab.transcript:
        transcript = Table(
            title="Agent Transcript",
            show_header=True,
            header_style="bold magenta",
        )
        transcript.add_column("#", style="dim", width=4)
        transcript.add_column("From", style="cyan", width=14)
        transcript.add_column("To", style="cyan", width=14)
        transcript.add_column("Content", width=44)

        for idx, msg in enumerate(collab.transcript, 1):
            transcript.add_row(str(idx), msg.from_agent, msg.to_agent, msg.content[:80])

        console.print(transcript)

    console.print()
