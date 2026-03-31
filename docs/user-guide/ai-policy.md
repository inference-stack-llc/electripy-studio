# AI Policy Engine

The AI Policy Engine provides enterprise-grade governance for AI, agent, and tool operations.  It complements the lower-level [`policy_gateway`](ai-policy-gateway.md) (regex-based text detection and sanitization) with subject/resource/action-level rules, approval workflows, evidence requirements, escalation directives, and redaction metadata.

## Why it exists

AI systems need governance layers that can enforce *who* may do *what* against *which* resources, with auditable decisions.  The Policy Engine answers questions like:

- Can this user invoke this tool?
- Does the request carry enough evidence (e.g. a JIRA ticket, justification)?
- Should the operation be denied, redacted, escalated, or require human approval?
- How long does an approval token last?

## Decision model

A policy evaluation returns one of five outcomes:

| Outcome             | Meaning |
|---------------------|---------|
| `allow`             | Proceed normally. |
| `deny`              | Block execution. |
| `redact`            | Allow but redact specified fields. |
| `require_approval`  | Block until a time-bound approval token is granted. |
| `escalate`          | Route to a designated escalation channel. |

## Core concepts

### Subject, Resource, Action

Every evaluation starts with a **PolicyContext** that describes:

- **PolicySubject** — *who* (actor ID, roles, teams).
- **PolicyResource** — *what* (resource type and ID, e.g. `tool:web_search`).
- **PolicyAction** — *how* (action type, e.g. `invoke`, `delete`).

### Evidence

Rules can require evidence items before granting access:

- `justification` — free-text rationale.
- `ticket_reference` — link to an issue tracker.
- `approval_token` — a previously granted approval.
- `mfa_challenge` — multi-factor authentication.
- `audit_log` — audit trail reference.

### Approval workflow

When the outcome is `require_approval`:

1. Call `engine.request_approval(context)` → `ApprovalRequest`.
2. A reviewer calls `engine.grant_approval(request_id, granted_by=..., evidence=...)` → `ApprovalToken`.
3. Subsequent calls use `engine.validate_token(token_id)` to verify the token is still valid.

Approval tokens are **time-bound** — each rule can specify a `ttl_seconds` and the shortest TTL wins.

### Policy packs

Group rules into versioned packs for distribution:

```python
from electripy.ai.policy import PolicyPack, PolicyRule

pack = PolicyPack(
    pack_id="enterprise-v1",
    version="1.0.0",
    rules=(
        PolicyRule(rule_id="admin-tools", description="..."),
        PolicyRule(rule_id="pii-redact", description="..."),
    ),
)

engine.load_pack(pack)
```

## Basic usage

```python
from electripy.ai.policy import (
    DecisionOutcome,
    EvidenceItem,
    EvidenceKind,
    InMemoryPolicyRepository,
    PolicyAction,
    PolicyContext,
    PolicyEngine,
    PolicyResource,
    PolicyRule,
    PolicySubject,
    RulePriority,
)

# 1. Define rules
repo = InMemoryPolicyRepository()
repo.add_rule(
    PolicyRule(
        rule_id="admin-tools",
        description="Only admins may invoke tools in production.",
        priority=RulePriority.HIGH,
        outcome=DecisionOutcome.DENY,
        resource_types=("tool",),
        action_types=("invoke",),
        required_roles=("admin",),
    )
)

# 2. Build engine
engine = PolicyEngine(repository=repo)

# 3. Evaluate
context = PolicyContext(
    subject=PolicySubject(actor_id="alice", roles=("viewer",)),
    resource=PolicyResource(resource_type="tool", resource_id="web_search"),
    action=PolicyAction(action_type="invoke"),
)

decision = engine.evaluate(context)
if decision.blocked:
    print(f"Blocked: {decision.outcome.value}")
    for v in decision.violations:
        print(f"  - {v.message}")
```

## Approval flow example

```python
from electripy.ai.policy import (
    DecisionOutcome,
    EvidenceItem,
    EvidenceKind,
    InMemoryPolicyRepository,
    PolicyAction,
    PolicyContext,
    PolicyEngine,
    PolicyResource,
    PolicyRule,
    PolicySubject,
)

repo = InMemoryPolicyRepository()
repo.add_rule(
    PolicyRule(
        rule_id="tool-approval",
        description="Tool invocations require approval.",
        outcome=DecisionOutcome.REQUIRE_APPROVAL,
        resource_types=("tool",),
        required_roles=("admin",),
        required_evidence=(EvidenceKind.JUSTIFICATION,),
        ttl_seconds=600,
    )
)

engine = PolicyEngine(repository=repo)

context = PolicyContext(
    subject=PolicySubject(actor_id="bob", roles=("developer",)),
    resource=PolicyResource(resource_type="tool", resource_id="deploy"),
    action=PolicyAction(action_type="invoke"),
)

# Step 1: Request approval
approval = engine.request_approval(context)

# Step 2: Grant (with evidence)
token = engine.grant_approval(
    approval.request_id,
    granted_by="team-lead",
    evidence=(EvidenceItem(kind=EvidenceKind.JUSTIFICATION, value="Emergency fix"),),
)

# Step 3: Validate before execution
validated = engine.validate_token(token.token_id)
assert validated.valid
```

## Escalation example

```python
from electripy.ai.policy import (
    DecisionOutcome,
    EscalationDirective,
    EscalationLevel,
    InMemoryPolicyRepository,
    LoggingEscalationHandler,
    PolicyEngine,
    PolicyRule,
    RulePriority,
)

repo = InMemoryPolicyRepository()
repo.add_rule(
    PolicyRule(
        rule_id="delete-guard",
        description="Delete operations require security review.",
        priority=RulePriority.CRITICAL,
        outcome=DecisionOutcome.ESCALATE,
        action_types=("delete",),
        required_roles=("security-team",),
        escalation=EscalationDirective(
            level=EscalationLevel.SECURITY,
            reason="Delete operations on production resources.",
            notify_channels=("slack:#security-alerts",),
        ),
    )
)

handler = LoggingEscalationHandler()
engine = PolicyEngine(repository=repo, escalation_handler=handler)
```

## Ports and adapters

The engine follows Ports & Adapters architecture with five protocols:

| Port                    | Purpose |
|-------------------------|---------|
| `PolicyEvaluatorPort`   | Evaluates context against rules. |
| `PolicyRepositoryPort`  | Loads and stores rules. |
| `ApprovalStorePort`     | Persists approval requests and tokens. |
| `EscalationHandlerPort` | Dispatches escalation directives. |
| `PolicyObserverPort`    | Receives decision notifications (observability). |

Built-in adapters: `DefaultPolicyEvaluator`, `InMemoryPolicyRepository`, `InMemoryApprovalStore`, `LoggingEscalationHandler`.

## Relationship to Policy Gateway

| Concern | Policy Gateway | Policy Engine |
|---------|---------------|--------------|
| Scope | Text-level detection (regex, sanitization) | Subject/resource/action governance |
| Rules | Pattern-based (`PolicyRule` with regex) | Role/evidence/escalation rules |
| Decisions | `allow`, `sanitize`, `deny`, `require_approval` | `allow`, `deny`, `redact`, `require_approval`, `escalate` |
| Approval | Flag only | Full workflow (request → grant → validate) |
| Best for | LLM I/O guardrails | Enterprise tool/agent governance |

Use both together: the Policy Gateway for request/response sanitization, and the Policy Engine for authorization and approval workflows.
