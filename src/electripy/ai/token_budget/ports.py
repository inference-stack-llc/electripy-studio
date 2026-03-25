"""Port for pluggable tokenizer implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerPort(Protocol):
    """Protocol for token counting implementations.

    Implement this to plug in a real tokenizer (e.g. tiktoken).
    """

    def count(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        ...
