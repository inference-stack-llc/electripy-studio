"""Services for prompt rendering and composition."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .domain import FewShotExample, PromptRole, RenderedMessage, RenderedPrompt
from .errors import MissingVariableError, TemplateSyntaxError

_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")
_MALFORMED_PATTERN = re.compile(r"\{\{[^}]*$|\{\w+\}(?!\})")


def render_template(template: str, variables: dict[str, str]) -> str:
    """Render a template string by replacing ``{{var}}`` placeholders.

    Args:
        template: Template text with ``{{variable}}`` placeholders.
        variables: Mapping of variable names to their string values.

    Returns:
        The rendered string with all placeholders substituted.

    Raises:
        MissingVariableError: If a placeholder has no matching variable.
        TemplateSyntaxError: If the template has malformed placeholders.

    Example::

        render_template("Hello {{name}}", {"name": "World"})
        # => "Hello World"
    """
    if _MALFORMED_PATTERN.search(template):
        raise TemplateSyntaxError(f"Malformed placeholder in template: {template!r}")

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        if var not in variables:
            raise MissingVariableError(var)
        return variables[var]

    return _VAR_PATTERN.sub(_replace, template)


def build_few_shot_block(
    examples: Sequence[FewShotExample],
    *,
    max_examples: int | None = None,
) -> list[RenderedMessage]:
    """Convert few-shot examples into an interleaved message list.

    Args:
        examples: Ordered few-shot examples.
        max_examples: Optional cap on the number of examples to include.

    Returns:
        List of alternating user/assistant messages.

    Example::

        msgs = build_few_shot_block([
            FewShotExample(user="2+2?", assistant="4"),
        ])
    """
    selected = examples[:max_examples] if max_examples is not None else examples
    messages: list[RenderedMessage] = []
    for ex in selected:
        messages.append(RenderedMessage(role=PromptRole.USER, content=ex.user))
        messages.append(RenderedMessage(role=PromptRole.ASSISTANT, content=ex.assistant))
    return messages


def compose_messages(
    *,
    system: str | None = None,
    few_shot: Sequence[FewShotExample] | None = None,
    max_few_shot: int | None = None,
    user: str,
    variables: dict[str, str] | None = None,
) -> RenderedPrompt:
    """Compose a full chat prompt from building blocks.

    Renders templates in ``system`` and ``user`` if ``variables`` is provided,
    inserts optional few-shot examples between system and user messages.

    Args:
        system: Optional system prompt template.
        few_shot: Optional few-shot examples.
        max_few_shot: Cap on few-shot examples.
        user: User message template.
        variables: Variables for template rendering.

    Returns:
        A fully rendered prompt ready for LLM submission.

    Example::

        prompt = compose_messages(
            system="You are a {{persona}}.",
            user="Summarize: {{text}}",
            variables={"persona": "helpful assistant", "text": "Hello world"},
        )
    """
    vars_ = variables or {}
    messages: list[RenderedMessage] = []

    if system is not None:
        rendered_system = render_template(system, vars_) if vars_ else system
        messages.append(RenderedMessage(role=PromptRole.SYSTEM, content=rendered_system))

    if few_shot:
        messages.extend(build_few_shot_block(few_shot, max_examples=max_few_shot))

    rendered_user = render_template(user, vars_) if vars_ else user
    messages.append(RenderedMessage(role=PromptRole.USER, content=rendered_user))

    return RenderedPrompt(messages=messages)
