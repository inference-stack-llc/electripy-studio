"""Integration helpers between policy gateway and LLM gateway hooks."""

from __future__ import annotations

from electripy.ai.llm_gateway import LlmMessage, LlmRequest, LlmResponse
from electripy.ai.llm_gateway.config import LlmRequestHook, LlmResponseHook
from electripy.ai.llm_gateway.errors import PolicyViolationError

from .domain import PolicyAction
from .services import PolicyGateway


def build_llm_policy_hooks(
    gateway: PolicyGateway,
    *,
    sanitize_preflight: bool = True,
    sanitize_postflight: bool = True,
) -> tuple[LlmRequestHook, LlmResponseHook]:
    """Build request/response hooks for :class:`LlmGatewaySettings`.

    Args:
        gateway: Policy gateway instance used for pre/post checks.
        sanitize_preflight: If True, apply preflight sanitization to messages.
        sanitize_postflight: If True, apply postflight sanitization to response text.

    Returns:
        A tuple of ``(request_hook, response_hook)`` callables.
    """

    def request_hook(request: LlmRequest) -> LlmRequest:
        updated_messages: list[LlmMessage] = []
        changed = False

        for message in request.messages:
            decision = gateway.evaluate_preflight(message.content)
            if decision.action in (PolicyAction.DENY, PolicyAction.REQUIRE_APPROVAL):
                reasons = tuple(decision.reason_codes or (decision.action.value,))
                raise PolicyViolationError(stage="preflight", reasons=reasons)

            content = message.content
            if (
                sanitize_preflight
                and decision.action == PolicyAction.SANITIZE
                and decision.sanitized_text is not None
            ):
                content = decision.sanitized_text
                changed = changed or (content != message.content)

            updated_messages.append(LlmMessage(role=message.role, content=content))

        if not changed:
            return request
        return request.clone_with_messages(updated_messages)

    def response_hook(request: LlmRequest, response: LlmResponse) -> LlmResponse:
        del request  # Request is available for future policy enrichments.

        decision = gateway.evaluate_postflight(response.text)
        if decision.action in (PolicyAction.DENY, PolicyAction.REQUIRE_APPROVAL):
            reasons = tuple(decision.reason_codes or (decision.action.value,))
            raise PolicyViolationError(stage="postflight", reasons=reasons)

        if (
            sanitize_postflight
            and decision.action == PolicyAction.SANITIZE
            and decision.sanitized_text is not None
        ):
            response.text = decision.sanitized_text
            response.metadata["policy_postflight"] = {
                "action": decision.action.value,
                "reason_codes": list(decision.reason_codes),
            }
        return response

    return request_hook, response_hook
