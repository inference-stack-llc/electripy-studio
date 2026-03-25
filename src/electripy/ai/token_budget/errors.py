"""Exception hierarchy for token budget utilities."""

from __future__ import annotations


class TokenBudgetError(Exception):
    """Base exception for token budget errors."""


class BudgetExceededError(TokenBudgetError):
    """Raised when text exceeds the allowed token budget."""

    def __init__(self, token_count: int, budget: int) -> None:
        self.token_count = token_count
        self.budget = budget
        super().__init__(f"Token count {token_count} exceeds budget {budget}")
