from __future__ import annotations

from electripy.ai.context_assembly import (
    ContextBlock,
    ContextPriority,
    assemble_context,
)


class _WordTokenizer:
    """Test tokenizer that counts words."""

    def count(self, text: str) -> int:
        return len(text.split()) if text else 0


class TestAssembleContext:
    def test_all_blocks_fit(self) -> None:
        blocks = [
            ContextBlock(label="a", content="one two", priority=ContextPriority.HIGH),
            ContextBlock(label="b", content="three", priority=ContextPriority.LOW),
        ]
        result = assemble_context(blocks, budget=10, tokenizer=_WordTokenizer())
        assert len(result.blocks) == 2
        assert result.dropped_labels == []
        assert result.total_tokens == 3

    def test_low_priority_dropped_first(self) -> None:
        blocks = [
            ContextBlock(
                label="critical",
                content="must keep",
                priority=ContextPriority.CRITICAL,
            ),
            ContextBlock(
                label="low",
                content="can drop this one easily",
                priority=ContextPriority.LOW,
            ),
        ]
        result = assemble_context(blocks, budget=3, tokenizer=_WordTokenizer())
        assert len(result.blocks) == 1
        assert result.blocks[0].label == "critical"
        assert "low" in result.dropped_labels

    def test_preserves_insertion_order(self) -> None:
        blocks = [
            ContextBlock(label="first", content="a", priority=ContextPriority.MEDIUM),
            ContextBlock(label="second", content="b", priority=ContextPriority.HIGH),
            ContextBlock(label="third", content="c", priority=ContextPriority.MEDIUM),
        ]
        result = assemble_context(blocks, budget=100, tokenizer=_WordTokenizer())
        labels = [b.label for b in result.blocks]
        assert labels == ["first", "second", "third"]

    def test_text_property(self) -> None:
        blocks = [
            ContextBlock(label="a", content="Hello", priority=ContextPriority.HIGH),
            ContextBlock(label="b", content="World", priority=ContextPriority.HIGH),
        ]
        result = assemble_context(blocks, budget=100, tokenizer=_WordTokenizer())
        assert result.text == "Hello\n\nWorld"

    def test_empty_blocks(self) -> None:
        result = assemble_context([], budget=100, tokenizer=_WordTokenizer())
        assert result.blocks == []
        assert result.total_tokens == 0

    def test_budget_zero_drops_all(self) -> None:
        blocks = [
            ContextBlock(label="a", content="text", priority=ContextPriority.HIGH),
        ]
        result = assemble_context(blocks, budget=0, tokenizer=_WordTokenizer())
        assert result.blocks == []
        assert "a" in result.dropped_labels
