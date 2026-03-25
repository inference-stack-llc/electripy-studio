"""Services and hooks for deterministic policy gateway enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

from electripy.core.logging import get_logger
from electripy.observability.ai_telemetry import TelemetryPort
from electripy.observability.ai_telemetry.services import record_policy_decision

from .adapters import RedactionSanitizerAdapter, RegexPolicyDetectorAdapter
from .domain import (
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    PolicyFinding,
    PolicyInput,
    PolicyRule,
    PolicySeverity,
    PolicyStage,
)
from .errors import PolicyConfigurationError
from .ports import PolicyDetectorPort, TextSanitizerPort

logger = get_logger(__name__)


@dataclass(slots=True)
class PolicyGatewaySettings:
    """Settings controlling policy decision behavior."""

    policy_version: str = "v1"
    deny_on_critical: bool = True
    require_approval_on_high: bool = True


@dataclass(slots=True)
class PolicyGateway:
    """Evaluate policy rules across request, response, stream, and tool stages."""

    rules: list[PolicyRule]
    detector: PolicyDetectorPort = field(default_factory=RegexPolicyDetectorAdapter)
    sanitizer: TextSanitizerPort = field(default_factory=RedactionSanitizerAdapter)
    settings: PolicyGatewaySettings = field(default_factory=PolicyGatewaySettings)
    telemetry: TelemetryPort | None = None
    _rules_by_id: dict[str, PolicyRule] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.rules:
            raise PolicyConfigurationError("rules must not be empty")
        self._rules_by_id = {rule.rule_id: rule for rule in self.rules}

    def evaluate_preflight(
        self,
        text: str,
        *,
        context: PolicyContext | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PolicyDecision:
        """Evaluate an inbound prompt or request payload before model/tool execution."""

        return self._evaluate(
            PolicyInput(
                stage=PolicyStage.PREFLIGHT,
                text=text,
                metadata=dict(metadata or {}),
                context=context,
            )
        )

    def evaluate_postflight(
        self,
        text: str,
        *,
        context: PolicyContext | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PolicyDecision:
        """Evaluate an outbound model response payload."""

        return self._evaluate(
            PolicyInput(
                stage=PolicyStage.POSTFLIGHT,
                text=text,
                metadata=dict(metadata or {}),
                context=context,
            )
        )

    def evaluate_stream_chunk(
        self,
        delta_text: str,
        *,
        context: PolicyContext | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PolicyDecision:
        """Evaluate a streamed token/delta payload."""

        return self._evaluate(
            PolicyInput(
                stage=PolicyStage.STREAM,
                text=delta_text,
                metadata=dict(metadata or {}),
                context=context,
            )
        )

    def evaluate_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, object],
        *,
        context: PolicyContext | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PolicyDecision:
        """Evaluate tool invocation names and arguments."""

        rendered_args = ", ".join(f"{k}={tool_args[k]!r}" for k in sorted(tool_args))
        text = f"{tool_name}({rendered_args})"
        return self._evaluate(
            PolicyInput(
                stage=PolicyStage.TOOL_CALL,
                text=text,
                tool_name=tool_name,
                tool_args=dict(tool_args),
                metadata=dict(metadata or {}),
                context=context,
            )
        )

    def _evaluate(self, policy_input: PolicyInput) -> PolicyDecision:
        findings = self.detector.detect(policy_input, self.rules)
        decision = self._build_decision(policy_input.text, findings)
        self._record_telemetry(decision)
        return decision

    def _build_decision(self, text: str, findings: list[PolicyFinding]) -> PolicyDecision:
        if not findings:
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                policy_version=self.settings.policy_version,
            )

        actions = {finding.action for finding in findings}
        severities = {finding.severity for finding in findings}
        reason_codes = sorted({finding.code for finding in findings})

        if PolicyAction.DENY in actions:
            return PolicyDecision(
                action=PolicyAction.DENY,
                policy_version=self.settings.policy_version,
                findings=findings,
                reason_codes=reason_codes,
            )

        if self.settings.deny_on_critical and PolicySeverity.CRITICAL in severities:
            return PolicyDecision(
                action=PolicyAction.DENY,
                policy_version=self.settings.policy_version,
                findings=findings,
                reason_codes=reason_codes,
            )

        if PolicyAction.REQUIRE_APPROVAL in actions:
            return PolicyDecision(
                action=PolicyAction.REQUIRE_APPROVAL,
                policy_version=self.settings.policy_version,
                findings=findings,
                reason_codes=reason_codes,
                requires_approval=True,
            )

        if self.settings.require_approval_on_high and PolicySeverity.HIGH in severities:
            return PolicyDecision(
                action=PolicyAction.REQUIRE_APPROVAL,
                policy_version=self.settings.policy_version,
                findings=findings,
                reason_codes=reason_codes,
                requires_approval=True,
            )

        sanitized_text = self.sanitizer.sanitize(
            text=text,
            findings=findings,
            rules_by_id=self._rules_by_id,
        )
        return PolicyDecision(
            action=PolicyAction.SANITIZE,
            policy_version=self.settings.policy_version,
            findings=findings,
            reason_codes=reason_codes,
            sanitized_text=sanitized_text,
        )

    def _record_telemetry(self, decision: PolicyDecision) -> None:
        if self.telemetry is None:
            return
        record_policy_decision(
            self.telemetry,
            decision=decision.action.value,
            violation_codes=decision.reason_codes,
            redactions_applied=decision.action == PolicyAction.SANITIZE,
        )


def before_llm_request(
    gateway: PolicyGateway,
    prompt_text: str,
    *,
    context: PolicyContext | None = None,
    metadata: dict[str, object] | None = None,
) -> PolicyDecision:
    """Evaluate policy before an LLM request."""

    return gateway.evaluate_preflight(prompt_text, context=context, metadata=metadata)


def after_llm_response(
    gateway: PolicyGateway,
    response_text: str,
    *,
    context: PolicyContext | None = None,
    metadata: dict[str, object] | None = None,
) -> PolicyDecision:
    """Evaluate policy after an LLM response."""

    return gateway.evaluate_postflight(response_text, context=context, metadata=metadata)


def on_stream_chunk(
    gateway: PolicyGateway,
    chunk_text: str,
    *,
    context: PolicyContext | None = None,
    metadata: dict[str, object] | None = None,
) -> PolicyDecision:
    """Evaluate policy for a streaming text chunk."""

    return gateway.evaluate_stream_chunk(chunk_text, context=context, metadata=metadata)


def authorize_tool_call(
    gateway: PolicyGateway,
    tool_name: str,
    tool_args: dict[str, object],
    *,
    context: PolicyContext | None = None,
    metadata: dict[str, object] | None = None,
) -> PolicyDecision:
    """Evaluate policy for a tool invocation."""

    return gateway.evaluate_tool_call(
        tool_name,
        tool_args,
        context=context,
        metadata=metadata,
    )
