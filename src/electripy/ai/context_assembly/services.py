"""Services for priority-based context window assembly."""

from __future__ import annotations

from collections.abc import Sequence

from electripy.ai.token_budget.ports import TokenizerPort

from .domain import AssembledContext, ContextBlock


def assemble_context(
    blocks: Sequence[ContextBlock],
    budget: int,
    tokenizer: TokenizerPort,
) -> AssembledContext:
    """Assemble context blocks into a token-limited window.

    Blocks are sorted by priority (highest first). The assembler greedily
    includes blocks until the budget is exhausted, dropping the lowest-priority
    blocks first. Within equal priority, insertion order is preserved.

    Args:
        blocks: Context blocks to assemble.
        budget: Maximum total tokens.
        tokenizer: Tokenizer for counting tokens.

    Returns:
        An AssembledContext with surviving blocks and metadata.

    Example::

        from electripy.ai.token_budget import CharEstimatorTokenizer
        from electripy.ai.context_assembly import (
            ContextBlock, ContextPriority, assemble_context,
        )

        blocks = [
            ContextBlock(label="system", content="You are helpful.", priority=ContextPriority.CRITICAL),
            ContextBlock(label="docs", content="Long document...", priority=ContextPriority.LOW),
            ContextBlock(label="query", content="What is X?", priority=ContextPriority.HIGH),
        ]
        result = assemble_context(blocks, budget=100, tokenizer=CharEstimatorTokenizer())
    """
    counted: list[tuple[int, int, ContextBlock]] = []
    for idx, block in enumerate(blocks):
        tokens = tokenizer.count(block.content)
        block.token_count = tokens
        counted.append((block.priority, idx, block))

    counted.sort(key=lambda t: (-t[0], t[1]))

    included: list[tuple[int, ContextBlock]] = []
    dropped: list[str] = []
    total = 0

    for _priority, original_idx, block in counted:
        if total + block.token_count <= budget:
            total += block.token_count
            included.append((original_idx, block))
        else:
            dropped.append(block.label)

    included.sort(key=lambda t: t[0])

    return AssembledContext(
        blocks=[block for _, block in included],
        total_tokens=total,
        dropped_labels=dropped,
        budget=budget,
    )
