"""Services for token counting, budget checking, and truncation."""

from __future__ import annotations

from .domain import TokenCount, TruncationResult, TruncationStrategy
from .errors import BudgetExceededError
from .ports import TokenizerPort


class CharEstimatorTokenizer:
    """Fallback tokenizer that estimates tokens from character count.

    Uses a configurable chars-per-token ratio (default 4, which is a
    reasonable average for English text with GPT-class models).

    Args:
        chars_per_token: Average characters per token.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._ratio = chars_per_token

    def count(self, text: str) -> int:
        """Estimate token count from character length."""
        return max(1, int(len(text) / self._ratio + 0.5)) if text else 0


def count_tokens(text: str, tokenizer: TokenizerPort) -> TokenCount:
    """Count tokens in a text string.

    Args:
        text: The text to count.
        tokenizer: Tokenizer implementation.

    Returns:
        A TokenCount with the text and its token count.

    Example::

        tc = count_tokens("Hello world", CharEstimatorTokenizer())
    """
    return TokenCount(text=text, token_count=tokenizer.count(text))


def fits_budget(text: str, budget: int, tokenizer: TokenizerPort) -> bool:
    """Check whether text fits within a token budget.

    Args:
        text: The text to check.
        budget: Maximum allowed tokens.
        tokenizer: Tokenizer implementation.

    Returns:
        True if the text fits within the budget.
    """
    return tokenizer.count(text) <= budget


def truncate_to_budget(
    text: str,
    budget: int,
    tokenizer: TokenizerPort,
    *,
    strategy: TruncationStrategy = TruncationStrategy.TAIL,
    strict: bool = False,
) -> TruncationResult:
    """Truncate text to fit within a token budget.

    Uses binary search on character position to find the longest prefix/suffix
    that fits within the budget.

    Args:
        text: The text to truncate.
        budget: Maximum allowed tokens.
        tokenizer: Tokenizer implementation.
        strategy: Where to cut (TAIL keeps start, HEAD keeps end, MIDDLE keeps edges).
        strict: If True, raise BudgetExceededError instead of truncating.

    Returns:
        A TruncationResult describing what happened.

    Raises:
        BudgetExceededError: If strict=True and text exceeds budget.
    """
    original_count = tokenizer.count(text)

    if original_count <= budget:
        return TruncationResult(
            text=text,
            original_tokens=original_count,
            final_tokens=original_count,
            was_truncated=False,
        )

    if strict:
        raise BudgetExceededError(original_count, budget)

    truncated = _truncate_by_strategy(text, budget, tokenizer, strategy)
    final_count = tokenizer.count(truncated)

    return TruncationResult(
        text=truncated,
        original_tokens=original_count,
        final_tokens=final_count,
        was_truncated=True,
    )


def _truncate_by_strategy(
    text: str,
    budget: int,
    tokenizer: TokenizerPort,
    strategy: TruncationStrategy,
) -> str:
    """Binary search for the longest substring that fits the budget."""
    if strategy == TruncationStrategy.TAIL:
        return _binary_search_prefix(text, budget, tokenizer)
    elif strategy == TruncationStrategy.HEAD:
        return _binary_search_suffix(text, budget, tokenizer)
    else:
        half_budget = budget // 2
        prefix = _binary_search_prefix(text, half_budget, tokenizer)
        suffix = _binary_search_suffix(text, budget - half_budget, tokenizer)
        return prefix + "..." + suffix


def _binary_search_prefix(text: str, budget: int, tokenizer: TokenizerPort) -> str:
    """Find the longest prefix that fits within the budget."""
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if tokenizer.count(candidate) <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _binary_search_suffix(text: str, budget: int, tokenizer: TokenizerPort) -> str:
    """Find the longest suffix that fits within the budget."""
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[len(text) - mid :]
        if tokenizer.count(candidate) <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best
