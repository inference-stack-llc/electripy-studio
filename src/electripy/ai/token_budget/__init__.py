"""Token counting, budget tracking, and truncation utilities.

Purpose:
  - Provide a pluggable tokenizer port for counting tokens across LLM providers.
  - Track and enforce context window budgets with automatic truncation.

Guarantees:
  - Tokenizer implementation is swappable via the TokenizerPort protocol.
  - Ships with a zero-dependency character-estimate fallback.
"""

from __future__ import annotations

from .domain import TokenCount, TruncationResult, TruncationStrategy
from .errors import BudgetExceededError, TokenBudgetError
from .ports import TokenizerPort
from .services import (
    CharEstimatorTokenizer,
    count_tokens,
    fits_budget,
    truncate_to_budget,
)

__all__ = [
    "TokenCount",
    "TruncationResult",
    "TruncationStrategy",
    "TokenBudgetError",
    "BudgetExceededError",
    "TokenizerPort",
    "CharEstimatorTokenizer",
    "count_tokens",
    "fits_budget",
    "truncate_to_budget",
]
