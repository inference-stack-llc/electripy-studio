"""Domain models for structured observability.

This module defines the core data structures for tracing, span metadata,
and redaction policies.  All models are vendor-neutral, serialisable,
and designed for use across both sync and async instrumentation paths.

The following models are provided:

- :class:`TraceContext` — Correlation identifiers for a trace.
- :class:`SpanKind` / :class:`SpanStatusCode` / :class:`SpanStatus` —
  Standard span lifecycle types.
- :class:`SpanAttributes` — Typed attribute bag for span metadata.
- :class:`GenAIRequestMetadata` / :class:`GenAIResponseMetadata` —
  Metadata for LLM request/response pairs.
- :class:`ToolInvocationMetadata` — Metadata for tool/function calls.
- :class:`RetrievalMetadata` — Metadata for RAG retrieval operations.
- :class:`PolicyDecisionMetadata` — Metadata for policy checks.
- :class:`MCPMetadata` — Metadata for MCP server/tool interactions.
- :class:`RedactionRule` / :class:`RedactionPolicy` — Redaction
  configuration primitives.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AttributeValue: TypeAlias = str | int | float | bool | None
"""Primitive attribute value compatible with OpenTelemetry conventions."""

Attributes: TypeAlias = dict[str, AttributeValue]
"""Mutable attribute mapping."""


# ---------------------------------------------------------------------------
# Span lifecycle enums
# ---------------------------------------------------------------------------


class SpanKind(StrEnum):
    """Classification of a span's role in a trace.

    The values intentionally mirror OpenTelemetry conventions so that
    mapping adapters can translate them without loss.
    """

    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    # AI-specific span kinds for richer semantic categorisation.
    LLM = "llm"
    AGENT = "agent"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    WORKFLOW = "workflow"
    POLICY = "policy"
    MCP = "mcp"


class SpanStatusCode(StrEnum):
    """Terminal status of a span."""

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class SpanStatus:
    """Immutable status attached to a completed span.

    Attributes:
        code: Terminal status code.
        description: Optional human-readable description of the status.
    """

    code: SpanStatusCode = SpanStatusCode.UNSET
    description: str | None = None


# ---------------------------------------------------------------------------
# Trace / span context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TraceContext:
    """Correlation context for a single trace.

    ``TraceContext`` carries identifiers that allow spans and events to
    be correlated across process boundaries.  A child context inherits
    the ``trace_id`` while receiving its own ``span_id``.

    Attributes:
        trace_id: Stable identifier for the logical trace.
        span_id: Identifier for the current span.
        parent_span_id: Optional parent span identifier.
        request_id: Optional per-request identifier.
        actor_id: Optional end-user or service actor.
        tenant_id: Optional multi-tenant identifier.
        environment: Deployment environment label.
        baggage: Arbitrary string-valued propagation items.
    """

    trace_id: str
    span_id: str | None = None
    parent_span_id: str | None = None
    request_id: str | None = None
    actor_id: str | None = None
    tenant_id: str | None = None
    environment: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def child(self, *, span_id: str) -> TraceContext:
        """Return a child context with a new ``span_id``.

        The child inherits all correlation identifiers from this
        context except ``span_id`` which is set to the given value.
        The current ``span_id`` becomes the child's
        ``parent_span_id``.

        Args:
            span_id: Identifier for the child span.

        Returns:
            A new :class:`TraceContext` for the child span.
        """
        return TraceContext(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=self.span_id,
            request_id=self.request_id,
            actor_id=self.actor_id,
            tenant_id=self.tenant_id,
            environment=self.environment,
            baggage=dict(self.baggage),
        )


# ---------------------------------------------------------------------------
# Span attributes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SpanAttributes:
    """Typed attribute bag for a span.

    Callers accumulate attributes via :meth:`set` and retrieve the
    full mapping via :meth:`as_dict`.  Keys follow a dotted namespace
    convention (e.g. ``gen_ai.model``, ``tool.name``).
    """

    _data: dict[str, AttributeValue] = field(default_factory=dict, repr=False)

    def set(self, key: str, value: AttributeValue) -> None:
        """Set a single attribute.

        Args:
            key: Dotted attribute key.
            value: Primitive attribute value.
        """
        self._data[key] = value

    def get(self, key: str, default: AttributeValue = None) -> AttributeValue:
        """Return the value for *key* or *default*.

        Args:
            key: Dotted attribute key to look up.
            default: Value returned when the key is absent.

        Returns:
            The attribute value or *default*.
        """
        return self._data.get(key, default)

    def as_dict(self) -> Attributes:
        """Return a shallow copy of the internal attribute mapping."""
        return dict(self._data)

    def merge(self, other: Mapping[str, AttributeValue]) -> None:
        """Merge additional attributes into this bag.

        Args:
            other: Mapping of attributes to merge.
        """
        self._data.update(other)


# ---------------------------------------------------------------------------
# AI / LLM metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GenAIRequestMetadata:
    """Metadata captured before an LLM call is dispatched.

    Attributes:
        provider: Logical provider name (``"openai"``, ``"anthropic"``).
        model: Model identifier (``"gpt-4o"``).
        temperature: Sampling temperature, if set.
        max_tokens: Maximum output tokens requested.
        input_tokens: Estimated input token count, if available.
        stop_sequences: Stop sequences provided to the model.
        tools: Tool/function names available to the model.
        stream: Whether the request uses streaming.
    """

    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    input_tokens: int | None = None
    stop_sequences: Sequence[str] | None = None
    tools: Sequence[str] | None = None
    stream: bool = False

    def to_attributes(self) -> Attributes:
        """Convert to a flat attribute mapping suitable for span attrs.

        Returns:
            Flat key-value mapping with ``gen_ai.`` prefix.
        """
        attrs: Attributes = {
            "gen_ai.system": self.provider,
            "gen_ai.request.model": self.model,
            "gen_ai.request.stream": self.stream,
        }
        if self.temperature is not None:
            attrs["gen_ai.request.temperature"] = self.temperature
        if self.max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = self.max_tokens
        if self.input_tokens is not None:
            attrs["gen_ai.usage.input_tokens"] = self.input_tokens
        if self.stop_sequences:
            attrs["gen_ai.request.stop_sequences"] = ",".join(self.stop_sequences)
        if self.tools:
            attrs["gen_ai.request.tools"] = ",".join(self.tools)
        return attrs


@dataclass(slots=True)
class GenAIResponseMetadata:
    """Metadata captured after an LLM call completes.

    Attributes:
        output_tokens: Number of output tokens produced.
        input_tokens: Actual input token count from the provider.
        finish_reason: Model-reported finish reason.
        model: Model identifier returned by the provider.
        latency_ms: End-to-end call latency in milliseconds.
        cache_hit: Whether a cache was used.
    """

    output_tokens: int | None = None
    input_tokens: int | None = None
    finish_reason: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    cache_hit: bool | None = None

    def to_attributes(self) -> Attributes:
        """Convert to a flat attribute mapping.

        Returns:
            Flat key-value mapping with ``gen_ai.`` prefix.
        """
        attrs: Attributes = {}
        if self.output_tokens is not None:
            attrs["gen_ai.usage.output_tokens"] = self.output_tokens
        if self.input_tokens is not None:
            attrs["gen_ai.usage.input_tokens"] = self.input_tokens
        if self.finish_reason is not None:
            attrs["gen_ai.response.finish_reason"] = self.finish_reason
        if self.model is not None:
            attrs["gen_ai.response.model"] = self.model
        if self.latency_ms is not None:
            attrs["gen_ai.latency_ms"] = self.latency_ms
        if self.cache_hit is not None:
            attrs["gen_ai.cache_hit"] = self.cache_hit
        return attrs


# ---------------------------------------------------------------------------
# Tool invocation metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolInvocationMetadata:
    """Metadata for a tool or function call.

    Attributes:
        tool_name: Logical name of the tool.
        tool_version: Optional version identifier.
        status: Outcome status (``"success"``, ``"error"``).
        latency_ms: Execution time in milliseconds.
        error_type: Exception class name on failure, if any.
    """

    tool_name: str
    tool_version: str | None = None
    status: str | None = None
    latency_ms: float | None = None
    error_type: str | None = None

    def to_attributes(self) -> Attributes:
        """Convert to a flat attribute mapping.

        Returns:
            Flat key-value mapping with ``tool.`` prefix.
        """
        attrs: Attributes = {"tool.name": self.tool_name}
        if self.tool_version is not None:
            attrs["tool.version"] = self.tool_version
        if self.status is not None:
            attrs["tool.status"] = self.status
        if self.latency_ms is not None:
            attrs["tool.latency_ms"] = self.latency_ms
        if self.error_type is not None:
            attrs["tool.error_type"] = self.error_type
        return attrs


# ---------------------------------------------------------------------------
# Retrieval metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RetrievalMetadata:
    """Metadata for a RAG / retrieval operation.

    Attributes:
        source: Logical data source name (e.g. ``"pinecone"``).
        query_text_hash: SHA-256 hash of the query text (never raw text).
        top_k: Number of results requested.
        results_returned: Number of results actually returned.
        latency_ms: Retrieval latency in milliseconds.
        score_min: Minimum similarity score among returned results.
        score_max: Maximum similarity score among returned results.
    """

    source: str
    query_text_hash: str | None = None
    top_k: int | None = None
    results_returned: int | None = None
    latency_ms: float | None = None
    score_min: float | None = None
    score_max: float | None = None

    def to_attributes(self) -> Attributes:
        """Convert to a flat attribute mapping.

        Returns:
            Flat key-value mapping with ``retrieval.`` prefix.
        """
        attrs: Attributes = {"retrieval.source": self.source}
        if self.query_text_hash is not None:
            attrs["retrieval.query_text_hash"] = self.query_text_hash
        if self.top_k is not None:
            attrs["retrieval.top_k"] = self.top_k
        if self.results_returned is not None:
            attrs["retrieval.results_returned"] = self.results_returned
        if self.latency_ms is not None:
            attrs["retrieval.latency_ms"] = self.latency_ms
        if self.score_min is not None:
            attrs["retrieval.score_min"] = self.score_min
        if self.score_max is not None:
            attrs["retrieval.score_max"] = self.score_max
        return attrs


# ---------------------------------------------------------------------------
# Policy decision metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PolicyDecisionMetadata:
    """Metadata for a policy evaluation outcome.

    Attributes:
        action: Decision outcome (``"allow"``, ``"deny"``, ``"sanitize"``).
        policy_version: Version of the policy set evaluated.
        violation_codes: Codes of any violations detected.
        redactions_applied: Whether redactions were performed.
        latency_ms: Evaluation latency in milliseconds.
    """

    action: str
    policy_version: str | None = None
    violation_codes: Sequence[str] | None = None
    redactions_applied: bool = False
    latency_ms: float | None = None

    def to_attributes(self) -> Attributes:
        """Convert to a flat attribute mapping.

        Returns:
            Flat key-value mapping with ``policy.`` prefix.
        """
        attrs: Attributes = {"policy.action": self.action}
        if self.policy_version is not None:
            attrs["policy.version"] = self.policy_version
        if self.violation_codes:
            attrs["policy.violation_codes"] = ",".join(self.violation_codes)
        attrs["policy.redactions_applied"] = self.redactions_applied
        if self.latency_ms is not None:
            attrs["policy.latency_ms"] = self.latency_ms
        return attrs


# ---------------------------------------------------------------------------
# MCP metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MCPMetadata:
    """Metadata for a Model Context Protocol interaction.

    Attributes:
        server_name: Name of the MCP server.
        tool_name: Name of the MCP tool invoked.
        protocol_version: MCP protocol version.
        status: Outcome status (``"success"``, ``"error"``).
        latency_ms: Call latency in milliseconds.
        error_type: Exception class name on failure, if any.
    """

    server_name: str
    tool_name: str | None = None
    protocol_version: str | None = None
    status: str | None = None
    latency_ms: float | None = None
    error_type: str | None = None

    def to_attributes(self) -> Attributes:
        """Convert to a flat attribute mapping.

        Returns:
            Flat key-value mapping with ``mcp.`` prefix.
        """
        attrs: Attributes = {"mcp.server_name": self.server_name}
        if self.tool_name is not None:
            attrs["mcp.tool_name"] = self.tool_name
        if self.protocol_version is not None:
            attrs["mcp.protocol_version"] = self.protocol_version
        if self.status is not None:
            attrs["mcp.status"] = self.status
        if self.latency_ms is not None:
            attrs["mcp.latency_ms"] = self.latency_ms
        if self.error_type is not None:
            attrs["mcp.error_type"] = self.error_type
        return attrs


# ---------------------------------------------------------------------------
# Redaction models
# ---------------------------------------------------------------------------


class RedactionRuleKind(StrEnum):
    """The matching strategy used by a :class:`RedactionRule`."""

    EXACT = "exact"
    PATTERN = "pattern"
    CALLABLE = "callable"


@dataclass(frozen=True, slots=True)
class RedactionRule:
    """A single redaction rule.

    Rules match attribute keys (or values) depending on the
    :attr:`kind`:

    - ``EXACT``: the attribute key must match :attr:`match` exactly
      (case-insensitive).
    - ``PATTERN``: :attr:`match` is a regular-expression pattern
      applied to attribute keys.
    - ``CALLABLE``: :attr:`predicate` receives the key and value and
      returns ``True`` if the attribute should be redacted.

    Attributes:
        kind: Matching strategy.
        match: String used for ``EXACT`` and ``PATTERN`` matching.
        replacement: Value to substitute for redacted attributes.
        predicate: Callable for ``CALLABLE`` matching.
    """

    kind: RedactionRuleKind
    match: str = ""
    replacement: str = "[REDACTED]"
    predicate: Callable[[str, AttributeValue], bool] | None = None

    def __post_init__(self) -> None:
        if self.kind == RedactionRuleKind.CALLABLE and self.predicate is None:
            raise ValueError("RedactionRule with kind=CALLABLE requires a predicate")
        if self.kind == RedactionRuleKind.PATTERN and self.match:
            # Validate the regex at construction time.
            re.compile(self.match)


# Default keys that are always redacted in enterprise-safe mode.
_DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "completion",
        "response",
        "content",
        "request_body",
        "response_body",
        "authorization",
        "api_key",
        "secret",
        "password",
        "token",
        "ssn",
        "credit_card",
        "auth_header",
    }
)


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """A collection of redaction rules applied before attributes leave
    the application boundary.

    By default the policy includes a set of exact-match rules for
    commonly sensitive keys.  Additional rules can be supplied at
    construction time.

    Attributes:
        rules: Sequence of redaction rules.
        enabled: Master switch — when ``False`` no redaction occurs.
        default_replacement: Replacement text used when a rule does not
            specify its own.
    """

    rules: tuple[RedactionRule, ...] = ()
    enabled: bool = True
    default_replacement: str = "[REDACTED]"

    @classmethod
    def enterprise_default(cls) -> RedactionPolicy:
        """Return a conservatively-configured default policy.

        The policy redacts attributes whose keys match commonly
        sensitive names (prompts, completions, secrets, PII fields).

        Returns:
            A :class:`RedactionPolicy` suitable for enterprise use.
        """
        rules = tuple(
            RedactionRule(
                kind=RedactionRuleKind.EXACT,
                match=key,
                replacement="[REDACTED]",
            )
            for key in sorted(_DEFAULT_SENSITIVE_KEYS)
        )
        return cls(rules=rules, enabled=True)


__all__ = [
    "AttributeValue",
    "Attributes",
    "SpanKind",
    "SpanStatusCode",
    "SpanStatus",
    "TraceContext",
    "SpanAttributes",
    "GenAIRequestMetadata",
    "GenAIResponseMetadata",
    "ToolInvocationMetadata",
    "RetrievalMetadata",
    "PolicyDecisionMetadata",
    "MCPMetadata",
    "RedactionRuleKind",
    "RedactionRule",
    "RedactionPolicy",
]
