"""Deterministic policy gateway utilities for AI workflows.

Purpose:
  - Enforce stable pre/post/tool/stream policy checks in-process.
  - Keep safety decisions provider-agnostic and easy to test.
"""

from __future__ import annotations

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
from .errors import PolicyConfigurationError, PolicyEvaluationError, PolicyGatewayError
from .integrations import build_llm_policy_hooks
from .ports import PolicyDetectorPort, TextSanitizerPort
from .services import (
    PolicyGateway,
    PolicyGatewaySettings,
    after_llm_response,
    authorize_tool_call,
    before_llm_request,
    on_stream_chunk,
)

__all__ = [
    "PolicyGateway",
    "PolicyGatewaySettings",
    "PolicyGatewayError",
    "PolicyConfigurationError",
    "PolicyEvaluationError",
    "PolicyAction",
    "PolicyStage",
    "PolicySeverity",
    "PolicyContext",
    "PolicyRule",
    "PolicyInput",
    "PolicyFinding",
    "PolicyDecision",
    "PolicyDetectorPort",
    "TextSanitizerPort",
    "RegexPolicyDetectorAdapter",
    "RedactionSanitizerAdapter",
    "build_llm_policy_hooks",
    "before_llm_request",
    "after_llm_response",
    "on_stream_chunk",
    "authorize_tool_call",
]
