"""Services for conversation memory management."""

from __future__ import annotations

from collections.abc import Sequence

from electripy.ai.token_budget.ports import TokenizerPort

from .domain import ConversationWindow, Turn, TurnRole


def append_turn(
    window: ConversationWindow,
    role: TurnRole,
    content: str,
    tokenizer: TokenizerPort,
) -> ConversationWindow:
    """Create a new window with an additional turn appended.

    Args:
        window: The current conversation window.
        role: Role of the new turn.
        content: Message text.
        tokenizer: Tokenizer for counting tokens.

    Returns:
        A new ConversationWindow with the turn added.

    Example::

        from electripy.ai.token_budget import CharEstimatorTokenizer
        from electripy.ai.conversation_memory import (
            ConversationWindow, TurnRole, append_turn,
        )

        window = ConversationWindow()
        window = append_turn(window, TurnRole.USER, "Hello!", CharEstimatorTokenizer())
    """
    tokens = tokenizer.count(content)
    turn = Turn(role=role, content=content, token_count=tokens)
    new_turns = [*window.turns, turn]
    return ConversationWindow(
        turns=new_turns,
        total_tokens=window.total_tokens + tokens,
    )


def recent_turns(
    window: ConversationWindow,
    n: int,
) -> ConversationWindow:
    """Return only the most recent n turns.

    Args:
        window: The conversation window to trim.
        n: Maximum number of turns to keep.

    Returns:
        A new ConversationWindow with at most n recent turns.
    """
    if n >= len(window.turns):
        return window
    kept = window.turns[-n:]
    total = sum(t.token_count for t in kept)
    return ConversationWindow(turns=kept, total_tokens=total)


def sliding_window(
    window: ConversationWindow,
    max_turns: int,
    tokenizer: TokenizerPort,
) -> ConversationWindow:
    """Apply a sliding window that keeps the most recent turns.

    Recounts tokens for accuracy when window is trimmed.

    Args:
        window: The conversation window.
        max_turns: Maximum number of turns.
        tokenizer: Tokenizer for recounting.

    Returns:
        A trimmed ConversationWindow.
    """
    if len(window.turns) <= max_turns:
        return window

    kept = window.turns[-max_turns:]
    total = 0
    for turn in kept:
        if turn.token_count == 0:
            turn.token_count = tokenizer.count(turn.content)
        total += turn.token_count

    return ConversationWindow(turns=kept, total_tokens=total)


def trim_to_budget(
    window: ConversationWindow,
    budget: int,
    tokenizer: TokenizerPort,
    *,
    preserve_system: bool = True,
) -> ConversationWindow:
    """Trim conversation to fit within a token budget.

    Drops the oldest non-system turns first. If ``preserve_system`` is True,
    system turns are never dropped.

    Args:
        window: The conversation window.
        budget: Maximum token budget.
        tokenizer: Tokenizer for counting.
        preserve_system: If True, system turns are always kept.

    Returns:
        A trimmed ConversationWindow that fits within the budget.
    """
    for turn in window.turns:
        if turn.token_count == 0:
            turn.token_count = tokenizer.count(turn.content)

    system_turns: list[Turn] = []
    other_turns: list[Turn] = []

    for turn in window.turns:
        if preserve_system and turn.role == TurnRole.SYSTEM:
            system_turns.append(turn)
        else:
            other_turns.append(turn)

    system_cost = sum(t.token_count for t in system_turns)
    remaining_budget = budget - system_cost

    kept_others: list[Turn] = []
    for turn in reversed(other_turns):
        if remaining_budget >= turn.token_count:
            kept_others.append(turn)
            remaining_budget -= turn.token_count
        # else skip this turn (oldest are dropped first since we iterate reversed)

    kept_others.reverse()

    final_turns = _merge_preserving_order(window.turns, system_turns, kept_others)
    total = sum(t.token_count for t in final_turns)
    return ConversationWindow(turns=final_turns, total_tokens=total)


def _merge_preserving_order(
    original: Sequence[Turn],
    system: Sequence[Turn],
    others: Sequence[Turn],
) -> list[Turn]:
    """Merge system and other turns back in their original order."""
    system_set = {id(t) for t in system}
    others_set = {id(t) for t in others}
    return [t for t in original if id(t) in system_set or id(t) in others_set]
