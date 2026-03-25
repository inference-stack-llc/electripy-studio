from __future__ import annotations

import pytest

from electripy.ai.prompt_engine import (
    FewShotExample,
    MissingVariableError,
    PromptRole,
    RenderedMessage,
    build_few_shot_block,
    compose_messages,
    render_template,
)


class TestRenderTemplate:
    def test_simple_substitution(self) -> None:
        result = render_template("Hello {{name}}", {"name": "World"})
        assert result == "Hello World"

    def test_multiple_variables(self) -> None:
        result = render_template("{{greeting}} {{name}}!", {"greeting": "Hi", "name": "Alice"})
        assert result == "Hi Alice!"

    def test_missing_variable_raises(self) -> None:
        with pytest.raises(MissingVariableError, match="name"):
            render_template("Hello {{name}}", {})

    def test_no_placeholders_returned_as_is(self) -> None:
        assert render_template("plain text", {}) == "plain text"

    def test_repeated_variable(self) -> None:
        result = render_template("{{x}} and {{x}}", {"x": "A"})
        assert result == "A and A"

    def test_empty_template(self) -> None:
        assert render_template("", {}) == ""


class TestBuildFewShotBlock:
    def test_single_example(self) -> None:
        examples = [FewShotExample(user="2+2?", assistant="4")]
        messages = build_few_shot_block(examples)
        assert len(messages) == 2
        assert messages[0].role == PromptRole.USER
        assert messages[0].content == "2+2?"
        assert messages[1].role == PromptRole.ASSISTANT
        assert messages[1].content == "4"

    def test_max_examples_cap(self) -> None:
        examples = [FewShotExample(user=f"q{i}", assistant=f"a{i}") for i in range(5)]
        messages = build_few_shot_block(examples, max_examples=2)
        assert len(messages) == 4  # 2 examples x 2 messages each

    def test_empty_examples(self) -> None:
        assert build_few_shot_block([]) == []


class TestComposeMessages:
    def test_basic_composition(self) -> None:
        prompt = compose_messages(
            system="You are a {{persona}}.",
            user="Summarize: {{text}}",
            variables={"persona": "helper", "text": "Hello"},
        )
        assert len(prompt.messages) == 2
        assert prompt.messages[0].role == PromptRole.SYSTEM
        assert prompt.messages[0].content == "You are a helper."
        assert prompt.messages[1].role == PromptRole.USER
        assert prompt.messages[1].content == "Summarize: Hello"

    def test_with_few_shot(self) -> None:
        prompt = compose_messages(
            system="System",
            few_shot=[FewShotExample(user="q", assistant="a")],
            user="Real question",
        )
        assert len(prompt.messages) == 4  # system + 2 few-shot + user

    def test_no_system(self) -> None:
        prompt = compose_messages(user="Just user")
        assert len(prompt.messages) == 1
        assert prompt.messages[0].role == PromptRole.USER

    def test_to_dicts(self) -> None:
        prompt = compose_messages(system="Sys", user="Usr")
        dicts = prompt.to_dicts()
        assert dicts == [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Usr"},
        ]


class TestDomainModels:
    def test_rendered_message_to_dict(self) -> None:
        msg = RenderedMessage(role=PromptRole.ASSISTANT, content="Hi")
        assert msg.to_dict() == {"role": "assistant", "content": "Hi"}

    def test_few_shot_example_metadata(self) -> None:
        ex = FewShotExample(user="q", assistant="a", metadata={"source": "train"})
        assert ex.metadata["source"] == "train"
