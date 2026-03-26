"""Tests for prompt_fingerprint."""

from __future__ import annotations

from electripy.ai.llm_gateway.domain import LlmMessage, LlmRequest, LlmRole
from electripy.ai.prompt_fingerprint import prompt_fingerprint, prompt_fingerprint_short


def _req(
    content: str = "hello", model: str = "gpt-4o-mini", temperature: float = 0.2
) -> LlmRequest:
    return LlmRequest(
        model=model,
        messages=[LlmMessage(role=LlmRole.USER, content=content)],
        temperature=temperature,
    )


class TestPromptFingerprint:
    def test_deterministic(self) -> None:
        r = _req()
        assert prompt_fingerprint(r) == prompt_fingerprint(r)

    def test_same_content_same_hash(self) -> None:
        r1 = _req("test prompt")
        r2 = _req("test prompt")
        assert prompt_fingerprint(r1) == prompt_fingerprint(r2)

    def test_different_content_different_hash(self) -> None:
        r1 = _req("hello")
        r2 = _req("goodbye")
        assert prompt_fingerprint(r1) != prompt_fingerprint(r2)

    def test_different_model_different_hash(self) -> None:
        r1 = _req(model="gpt-4o-mini")
        r2 = _req(model="gpt-4o")
        assert prompt_fingerprint(r1) != prompt_fingerprint(r2)

    def test_different_temperature_different_hash(self) -> None:
        r1 = _req(temperature=0.0)
        r2 = _req(temperature=0.7)
        assert prompt_fingerprint(r1) != prompt_fingerprint(r2)

    def test_sha256_length(self) -> None:
        fp = prompt_fingerprint(_req())
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_short_default_length(self) -> None:
        fp = prompt_fingerprint_short(_req())
        assert len(fp) == 12

    def test_short_custom_length(self) -> None:
        fp = prompt_fingerprint_short(_req(), length=8)
        assert len(fp) == 8

    def test_short_is_prefix_of_full(self) -> None:
        r = _req()
        full = prompt_fingerprint(r)
        short = prompt_fingerprint_short(r, length=16)
        assert full.startswith(short)

    def test_multi_message_order_matters(self) -> None:
        r1 = LlmRequest(
            model="test",
            messages=[
                LlmMessage(role=LlmRole.SYSTEM, content="be helpful"),
                LlmMessage(role=LlmRole.USER, content="hi"),
            ],
        )
        r2 = LlmRequest(
            model="test",
            messages=[
                LlmMessage(role=LlmRole.USER, content="hi"),
                LlmMessage(role=LlmRole.SYSTEM, content="be helpful"),
            ],
        )
        assert prompt_fingerprint(r1) != prompt_fingerprint(r2)

    def test_compatible_with_cache_key(self) -> None:
        """Verify fingerprint matches the llm_cache compute_cache_key."""
        from electripy.ai.llm_cache.services import compute_cache_key

        r = _req("compatibility check")
        assert prompt_fingerprint(r) == compute_cache_key(r)
