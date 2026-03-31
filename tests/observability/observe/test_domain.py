"""Tests for the observe package domain models."""

from __future__ import annotations

import re

import pytest

from electripy.observability.observe.domain import (
    GenAIRequestMetadata,
    GenAIResponseMetadata,
    MCPMetadata,
    PolicyDecisionMetadata,
    RedactionPolicy,
    RedactionRule,
    RedactionRuleKind,
    RetrievalMetadata,
    SpanAttributes,
    ToolInvocationMetadata,
    TraceContext,
)


class TestTraceContext:
    """TraceContext creation and child-span derivation."""

    def test_child_inherits_trace_id(self) -> None:
        """Child context shares the parent's trace_id."""
        parent = TraceContext(trace_id="t1", span_id="s1")
        child = parent.child(span_id="s2")

        assert child.trace_id == "t1"
        assert child.span_id == "s2"
        assert child.parent_span_id == "s1"

    def test_child_inherits_correlation_fields(self) -> None:
        """request_id, actor_id, tenant_id carry through."""
        parent = TraceContext(
            trace_id="t1",
            span_id="s1",
            request_id="r1",
            actor_id="a1",
            tenant_id="tn1",
            environment="prod",
            baggage={"k": "v"},
        )
        child = parent.child(span_id="s2")

        assert child.request_id == "r1"
        assert child.actor_id == "a1"
        assert child.tenant_id == "tn1"
        assert child.environment == "prod"
        assert child.baggage == {"k": "v"}

    def test_child_baggage_is_independent_copy(self) -> None:
        """Mutating the child's baggage does not affect the parent."""
        parent = TraceContext(trace_id="t1", baggage={"a": "1"})
        child = parent.child(span_id="s2")
        child.baggage["b"] = "2"

        assert "b" not in parent.baggage


class TestSpanAttributes:
    """SpanAttributes set/get/merge/as_dict operations."""

    def test_set_and_get(self) -> None:
        """Basic set and get round-trip."""
        attrs = SpanAttributes()
        attrs.set("gen_ai.model", "gpt-4o")
        assert attrs.get("gen_ai.model") == "gpt-4o"

    def test_get_default(self) -> None:
        """get() returns default for missing keys."""
        attrs = SpanAttributes()
        assert attrs.get("missing") is None
        assert attrs.get("missing", "fallback") == "fallback"

    def test_as_dict_returns_copy(self) -> None:
        """as_dict returns a shallow copy."""
        attrs = SpanAttributes()
        attrs.set("k", "v")
        d = attrs.as_dict()
        d["k"] = "mutated"
        assert attrs.get("k") == "v"

    def test_merge(self) -> None:
        """merge() adds new attributes and overwrites existing ones."""
        attrs = SpanAttributes()
        attrs.set("a", 1)
        attrs.merge({"a": 2, "b": 3})
        assert attrs.get("a") == 2
        assert attrs.get("b") == 3


class TestGenAIMetadata:
    """GenAIRequestMetadata and GenAIResponseMetadata serialisation."""

    def test_request_to_attributes(self) -> None:
        """Request metadata produces gen_ai namespace attributes."""
        meta = GenAIRequestMetadata(
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=256,
            input_tokens=100,
            stop_sequences=["END"],
            tools=["calculator", "search"],
            stream=True,
        )
        attrs = meta.to_attributes()

        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.request.model"] == "gpt-4o"
        assert attrs["gen_ai.request.temperature"] == 0.7
        assert attrs["gen_ai.request.max_tokens"] == 256
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.request.stop_sequences"] == "END"
        assert attrs["gen_ai.request.tools"] == "calculator,search"
        assert attrs["gen_ai.request.stream"] is True

    def test_request_minimal(self) -> None:
        """Minimal request metadata omits optional fields."""
        meta = GenAIRequestMetadata(provider="anthropic", model="claude-3")
        attrs = meta.to_attributes()

        assert "gen_ai.request.temperature" not in attrs
        assert "gen_ai.request.max_tokens" not in attrs

    def test_response_to_attributes(self) -> None:
        """Response metadata produces gen_ai namespace attributes."""
        meta = GenAIResponseMetadata(
            output_tokens=50,
            input_tokens=100,
            finish_reason="stop",
            model="gpt-4o",
            latency_ms=123.4,
            cache_hit=False,
        )
        attrs = meta.to_attributes()

        assert attrs["gen_ai.usage.output_tokens"] == 50
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.response.finish_reason"] == "stop"
        assert attrs["gen_ai.response.model"] == "gpt-4o"
        assert attrs["gen_ai.latency_ms"] == 123.4
        assert attrs["gen_ai.cache_hit"] is False


class TestToolInvocationMetadata:
    """ToolInvocationMetadata serialisation."""

    def test_to_attributes(self) -> None:
        """Full tool metadata maps correctly."""
        meta = ToolInvocationMetadata(
            tool_name="calculator",
            tool_version="1.0",
            status="success",
            latency_ms=10.0,
        )
        attrs = meta.to_attributes()

        assert attrs["tool.name"] == "calculator"
        assert attrs["tool.version"] == "1.0"
        assert attrs["tool.status"] == "success"
        assert attrs["tool.latency_ms"] == 10.0

    def test_minimal(self) -> None:
        """Minimal metadata only includes the tool name."""
        meta = ToolInvocationMetadata(tool_name="search")
        attrs = meta.to_attributes()

        assert attrs == {"tool.name": "search"}


class TestRetrievalMetadata:
    """RetrievalMetadata serialisation."""

    def test_to_attributes(self) -> None:
        """Full retrieval metadata maps correctly."""
        meta = RetrievalMetadata(
            source="pinecone",
            query_text_hash="sha256:abc",
            top_k=5,
            results_returned=3,
            latency_ms=45.2,
            score_min=0.72,
            score_max=0.95,
        )
        attrs = meta.to_attributes()

        assert attrs["retrieval.source"] == "pinecone"
        assert attrs["retrieval.query_text_hash"] == "sha256:abc"
        assert attrs["retrieval.top_k"] == 5
        assert attrs["retrieval.results_returned"] == 3
        assert attrs["retrieval.latency_ms"] == 45.2
        assert attrs["retrieval.score_min"] == 0.72
        assert attrs["retrieval.score_max"] == 0.95


class TestPolicyDecisionMetadata:
    """PolicyDecisionMetadata serialisation."""

    def test_to_attributes(self) -> None:
        """Full policy metadata maps correctly."""
        meta = PolicyDecisionMetadata(
            action="deny",
            policy_version="2.1",
            violation_codes=["PII_DETECTED", "JAILBREAK"],
            redactions_applied=True,
            latency_ms=5.0,
        )
        attrs = meta.to_attributes()

        assert attrs["policy.action"] == "deny"
        assert attrs["policy.version"] == "2.1"
        assert attrs["policy.violation_codes"] == "PII_DETECTED,JAILBREAK"
        assert attrs["policy.redactions_applied"] is True
        assert attrs["policy.latency_ms"] == 5.0


class TestMCPMetadata:
    """MCPMetadata serialisation."""

    def test_to_attributes(self) -> None:
        """Full MCP metadata maps correctly."""
        meta = MCPMetadata(
            server_name="code-server",
            tool_name="run_code",
            protocol_version="1.0",
            status="success",
            latency_ms=200.0,
        )
        attrs = meta.to_attributes()

        assert attrs["mcp.server_name"] == "code-server"
        assert attrs["mcp.tool_name"] == "run_code"
        assert attrs["mcp.protocol_version"] == "1.0"
        assert attrs["mcp.status"] == "success"
        assert attrs["mcp.latency_ms"] == 200.0


class TestRedactionRule:
    """RedactionRule construction and validation."""

    def test_callable_without_predicate_raises(self) -> None:
        """A CALLABLE rule without a predicate is invalid."""
        with pytest.raises(ValueError, match="predicate"):
            RedactionRule(kind=RedactionRuleKind.CALLABLE)

    def test_invalid_pattern_raises(self) -> None:
        """An invalid regex pattern is rejected at construction."""
        with pytest.raises(re.error):
            RedactionRule(kind=RedactionRuleKind.PATTERN, match="[invalid")


class TestRedactionPolicy:
    """RedactionPolicy factory methods."""

    def test_enterprise_default_covers_sensitive_keys(self) -> None:
        """The enterprise default includes rules for common sensitive keys."""
        policy = RedactionPolicy.enterprise_default()

        assert policy.enabled is True
        assert len(policy.rules) > 0

        matched_keys = {r.match for r in policy.rules}
        for key in ("prompt", "completion", "api_key", "password", "secret"):
            assert key in matched_keys

    def test_disabled_policy(self) -> None:
        """A disabled policy can be constructed with enabled=False."""
        policy = RedactionPolicy(enabled=False)
        assert policy.enabled is False
