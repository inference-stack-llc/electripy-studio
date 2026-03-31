"""Policy engine service — the primary orchestration entry-point.

The :class:`PolicyEngine` wires evaluator, repository, approval store,
escalation handler, and observer ports into a single facade for
evaluating requests and managing approval workflows.

Example::

    from electripy.ai.policy import (
        PolicyEngine,
        InMemoryPolicyRepository,
        DefaultPolicyEvaluator,
    )

    repo = InMemoryPolicyRepository()
    engine = PolicyEngine(repository=repo)
    decision = engine.evaluate(context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from electripy.core.logging import get_logger

from .adapters import DefaultPolicyEvaluator, InMemoryApprovalStore
from .domain import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalToken,
    DecisionOutcome,
    EvidenceItem,
    EvidenceKind,
    PolicyContext,
    PolicyDecision,
    PolicyPack,
)
from .errors import ApprovalError, EvidenceError, PolicyPackError
from .ports import (
    ApprovalStorePort,
    EscalationHandlerPort,
    PolicyEvaluatorPort,
    PolicyObserverPort,
    PolicyRepositoryPort,
)

__all__ = [
    "PolicyEngine",
    "PolicyEngineSettings",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class PolicyEngineSettings:
    """Settings controlling policy engine behavior.

    Attributes:
        default_approval_ttl: Default time-to-live for approval requests.
        enforce_evidence: Raise on missing evidence instead of denying.
        policy_version: Label attached to decisions for audit trails.
    """

    default_approval_ttl: timedelta = field(default_factory=lambda: timedelta(hours=1))
    enforce_evidence: bool = False
    policy_version: str = "v1"


@dataclass(slots=True)
class PolicyEngine:
    """Primary orchestrator for policy evaluation and approval workflows.

    Attributes:
        repository: Source of policy rules.
        evaluator: Evaluates context against rules.
        approval_store: Persists approval requests and tokens.
        escalation_handler: Dispatches escalation directives.
        observer: Receives decision notifications.
        settings: Engine configuration.
    """

    repository: PolicyRepositoryPort
    evaluator: PolicyEvaluatorPort = field(default_factory=DefaultPolicyEvaluator)
    approval_store: ApprovalStorePort = field(default_factory=InMemoryApprovalStore)
    escalation_handler: EscalationHandlerPort | None = None
    observer: PolicyObserverPort | None = None
    settings: PolicyEngineSettings = field(default_factory=PolicyEngineSettings)

    # ── public API ───────────────────────────────────────────────────

    def evaluate(self, context: PolicyContext) -> PolicyDecision:
        """Evaluate a policy context and return a decision.

        Loads rules from the repository, delegates to the evaluator,
        and handles escalation and observation side-effects.
        """
        rules = self.repository.list_rules(resource_type=context.resource.resource_type)
        decision = self.evaluator.evaluate(context, rules)

        # Attach policy version from settings.
        decision = PolicyDecision(
            outcome=decision.outcome,
            violations=decision.violations,
            reasons=decision.reasons,
            redactions=decision.redactions,
            escalation=decision.escalation,
            policy_version=self.settings.policy_version,
            request_id=decision.request_id or context.request_id,
            evaluated_at=decision.evaluated_at,
        )

        if decision.escalation is not None and self.escalation_handler is not None:
            self.escalation_handler.escalate(decision.escalation, context)
            logger.info(
                "Escalation dispatched",
                extra={"rule_ids": [v.rule_id for v in decision.violations]},
            )

        if self.observer is not None:
            self.observer.on_decision(context, decision)

        logger.debug(
            "Policy evaluated",
            extra={
                "outcome": decision.outcome.value,
                "request_id": decision.request_id,
                "violation_count": len(decision.violations),
            },
        )
        return decision

    def request_approval(self, context: PolicyContext) -> ApprovalRequest:
        """Create an approval request from a require_approval decision.

        The caller should first call :meth:`evaluate` and check for
        ``DecisionOutcome.REQUIRE_APPROVAL`` before calling this.
        """
        decision = self.evaluate(context)
        if decision.outcome != DecisionOutcome.REQUIRE_APPROVAL:
            raise ApprovalError(
                f"Cannot request approval: decision is {decision.outcome.value}, "
                f"not require_approval"
            )

        # Collect required evidence kinds from violated rules.
        rules = self.repository.list_rules(resource_type=context.resource.resource_type)
        required_evidence: set[EvidenceKind] = set()
        violated_ids = {v.rule_id for v in decision.violations}
        ttl_seconds: int | None = None
        for rule in rules:
            if rule.rule_id in violated_ids:
                required_evidence.update(rule.required_evidence)
                if rule.ttl_seconds is not None:
                    if ttl_seconds is None or rule.ttl_seconds < ttl_seconds:
                        ttl_seconds = rule.ttl_seconds

        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self.settings.default_approval_ttl

        approval = ApprovalRequest(
            subject=context.subject,
            resource=context.resource,
            action=context.action,
            required_evidence=tuple(sorted(required_evidence, key=lambda e: e.value)),
            ttl=ttl,
        )
        self.approval_store.save_request(approval)
        logger.info(
            "Approval request created",
            extra={"request_id": approval.request_id},
        )
        return approval

    def grant_approval(
        self,
        request_id: str,
        *,
        granted_by: str,
        evidence: tuple[EvidenceItem, ...] = (),
    ) -> ApprovalToken:
        """Grant an approval and produce a time-bound token.

        Raises:
            ApprovalError: If the request is not found, already resolved,
                or has expired.
            EvidenceError: If required evidence is not provided.
        """
        request = self.approval_store.get_request(request_id)
        if request is None:
            raise ApprovalError(f"Approval request not found: {request_id}")
        if request.status != ApprovalStatus.PENDING:
            raise ApprovalError(
                f"Approval request {request_id} is {request.status.value}, not pending"
            )
        if request.expired:
            raise ApprovalError(f"Approval request {request_id} has expired")

        # Validate evidence.
        if request.required_evidence:
            provided_kinds = {e.kind for e in evidence}
            missing = set(request.required_evidence) - provided_kinds
            if missing:
                if self.settings.enforce_evidence:
                    raise EvidenceError(
                        f"Missing required evidence: {', '.join(e.value for e in missing)}"
                    )
                logger.warning(
                    "Granting approval with missing evidence",
                    extra={"missing": [e.value for e in missing]},
                )

        token = ApprovalToken(
            request_id=request_id,
            granted_by=granted_by,
            evidence=evidence,
            expires_at=datetime.now(tz=UTC) + request.ttl,
        )
        self.approval_store.save_token(token)
        logger.info(
            "Approval token granted",
            extra={"token_id": token.token_id, "request_id": request_id},
        )
        return token

    def validate_token(self, token_id: str) -> ApprovalToken:
        """Validate an approval token and return it if still valid.

        Raises:
            ApprovalError: If the token is not found or has expired.
        """
        token = self.approval_store.get_token(token_id)
        if token is None:
            raise ApprovalError(f"Approval token not found: {token_id}")
        if not token.valid:
            raise ApprovalError(f"Approval token {token_id} has expired")
        return token

    def load_pack(self, pack: PolicyPack) -> None:
        """Load all rules from a policy pack into the repository.

        Raises:
            PolicyPackError: If the pack contains no rules or duplicate IDs.
        """
        if not pack.rules:
            raise PolicyPackError(f"Policy pack '{pack.pack_id}' contains no rules")

        ids = [r.rule_id for r in pack.rules]
        if len(ids) != len(set(ids)):
            raise PolicyPackError(f"Policy pack '{pack.pack_id}' contains duplicate rule IDs")

        existing = self.repository.list_rules()
        existing_ids = {r.rule_id for r in existing}
        for rule in pack.rules:
            if rule.rule_id in existing_ids:
                raise PolicyPackError(
                    f"Rule '{rule.rule_id}' from pack '{pack.pack_id}' "
                    f"conflicts with an existing rule"
                )
            self.repository.add_rule(rule)  # type: ignore[attr-defined]

        logger.info(
            "Policy pack loaded",
            extra={
                "pack_id": pack.pack_id,
                "version": pack.version,
                "rule_count": len(pack.rules),
            },
        )
