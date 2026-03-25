from __future__ import annotations

from electripy.ai.conversation_memory import (
    ConversationWindow,
    Turn,
    TurnRole,
    append_turn,
    recent_turns,
    sliding_window,
    trim_to_budget,
)


class _WordTokenizer:
    """Test tokenizer that counts words."""

    def count(self, text: str) -> int:
        return len(text.split()) if text else 0


class TestAppendTurn:
    def test_appends_and_counts(self) -> None:
        tok = _WordTokenizer()
        window = ConversationWindow()
        window = append_turn(window, TurnRole.USER, "hello world", tok)
        assert len(window.turns) == 1
        assert window.turns[0].token_count == 2
        assert window.total_tokens == 2

    def test_accumulates_tokens(self) -> None:
        tok = _WordTokenizer()
        window = ConversationWindow()
        window = append_turn(window, TurnRole.USER, "hello", tok)
        window = append_turn(window, TurnRole.ASSISTANT, "hi there friend", tok)
        assert window.total_tokens == 4  # 1 + 3


class TestRecentTurns:
    def test_keeps_most_recent(self) -> None:
        turns = [Turn(role=TurnRole.USER, content=f"msg{i}", token_count=1) for i in range(5)]
        window = ConversationWindow(turns=turns, total_tokens=5)
        trimmed = recent_turns(window, 2)
        assert len(trimmed.turns) == 2
        assert trimmed.turns[0].content == "msg3"
        assert trimmed.turns[1].content == "msg4"

    def test_noop_when_under_limit(self) -> None:
        turns = [Turn(role=TurnRole.USER, content="a", token_count=1)]
        window = ConversationWindow(turns=turns, total_tokens=1)
        assert recent_turns(window, 10) is window


class TestSlidingWindow:
    def test_trims_oldest(self) -> None:
        tok = _WordTokenizer()
        turns = [Turn(role=TurnRole.USER, content=f"word{i}", token_count=1) for i in range(10)]
        window = ConversationWindow(turns=turns, total_tokens=10)
        trimmed = sliding_window(window, max_turns=3, tokenizer=tok)
        assert len(trimmed.turns) == 3
        assert trimmed.turns[0].content == "word7"


class TestTrimToBudget:
    def test_preserves_system_drops_oldest(self) -> None:
        tok = _WordTokenizer()
        turns = [
            Turn(role=TurnRole.SYSTEM, content="system prompt here", token_count=0),
            Turn(role=TurnRole.USER, content="old message one", token_count=0),
            Turn(role=TurnRole.ASSISTANT, content="old reply one", token_count=0),
            Turn(role=TurnRole.USER, content="recent question", token_count=0),
        ]
        window = ConversationWindow(turns=turns, total_tokens=0)
        trimmed = trim_to_budget(window, budget=5, tokenizer=tok)

        roles = [t.role for t in trimmed.turns]
        assert TurnRole.SYSTEM in roles
        # System "system prompt here" = 3 tokens, budget remaining = 2
        # "recent question" = 2 tokens -> should fit
        contents = [t.content for t in trimmed.turns]
        assert "system prompt here" in contents
        assert "recent question" in contents

    def test_all_fit(self) -> None:
        tok = _WordTokenizer()
        turns = [Turn(role=TurnRole.USER, content="hi", token_count=0)]
        window = ConversationWindow(turns=turns, total_tokens=0)
        trimmed = trim_to_budget(window, budget=100, tokenizer=tok)
        assert len(trimmed.turns) == 1

    def test_to_dicts(self) -> None:
        window = ConversationWindow(
            turns=[
                Turn(role=TurnRole.USER, content="Hello"),
                Turn(role=TurnRole.ASSISTANT, content="Hi"),
            ]
        )
        assert window.to_dicts() == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
