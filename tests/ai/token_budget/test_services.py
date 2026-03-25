from __future__ import annotations

import pytest

from electripy.ai.token_budget import (
    BudgetExceededError,
    CharEstimatorTokenizer,
    TokenCount,
    TruncationStrategy,
    count_tokens,
    fits_budget,
    truncate_to_budget,
)


class _ExactTokenizer:
    """Test tokenizer that counts words (space-separated)."""

    def count(self, text: str) -> int:
        return len(text.split()) if text else 0


class TestCharEstimatorTokenizer:
    def test_default_ratio(self) -> None:
        tok = CharEstimatorTokenizer()
        # 12 chars / 4.0 = 3.0, rounded = 3
        assert tok.count("Hello World!") == 3

    def test_empty_string(self) -> None:
        assert CharEstimatorTokenizer().count("") == 0

    def test_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CharEstimatorTokenizer(chars_per_token=0)


class TestCountTokens:
    def test_basic_count(self) -> None:
        tok = _ExactTokenizer()
        result = count_tokens("one two three", tok)
        assert isinstance(result, TokenCount)
        assert result.token_count == 3
        assert result.text == "one two three"


class TestFitsBudget:
    def test_fits(self) -> None:
        tok = _ExactTokenizer()
        assert fits_budget("one two", 5, tok) is True

    def test_exceeds(self) -> None:
        tok = _ExactTokenizer()
        assert fits_budget("one two three four five six", 3, tok) is False


class TestTruncateToBudget:
    def test_no_truncation_needed(self) -> None:
        tok = _ExactTokenizer()
        result = truncate_to_budget("short", 10, tok)
        assert result.was_truncated is False
        assert result.text == "short"

    def test_tail_truncation(self) -> None:
        tok = _ExactTokenizer()
        text = "one two three four five"
        result = truncate_to_budget(text, 3, tok, strategy=TruncationStrategy.TAIL)
        assert result.was_truncated is True
        assert result.final_tokens <= 3

    def test_head_truncation(self) -> None:
        tok = _ExactTokenizer()
        text = "one two three four five"
        result = truncate_to_budget(text, 3, tok, strategy=TruncationStrategy.HEAD)
        assert result.was_truncated is True
        assert result.final_tokens <= 3

    def test_middle_truncation(self) -> None:
        tok = _ExactTokenizer()
        text = "one two three four five six"
        result = truncate_to_budget(text, 3, tok, strategy=TruncationStrategy.MIDDLE)
        assert result.was_truncated is True
        assert "..." in result.text

    def test_strict_raises(self) -> None:
        tok = _ExactTokenizer()
        with pytest.raises(BudgetExceededError, match="exceeds budget"):
            truncate_to_budget("one two three", 1, tok, strict=True)

    def test_budget_exceeded_error_attrs(self) -> None:
        err = BudgetExceededError(100, 50)
        assert err.token_count == 100
        assert err.budget == 50
