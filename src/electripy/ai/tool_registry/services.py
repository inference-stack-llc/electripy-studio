"""Services for tool registration, schema generation, and argument validation."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from .domain import ToolDefinition, ToolParameter, ToolSchema
from .errors import ToolRegistryError, ToolValidationError

_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _resolve_json_type(annotation: object) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    if annotation is inspect.Parameter.empty:
        return "string"

    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        annotation = origin

    if isinstance(annotation, type):
        return _PYTHON_TO_JSON_TYPE.get(annotation, "string")
    return "string"


def generate_schema(func: Callable[..., Any]) -> ToolSchema:
    """Generate a ToolSchema from a function's signature and type hints.

    Inspects the function's parameters and type annotations to build
    a schema. Parameters named ``self`` or ``cls`` are skipped.

    Args:
        func: The function to generate a schema for.

    Returns:
        A ToolSchema describing the function's parameters.

    Example::

        def greet(name: str, times: int = 1) -> str:
            return f"Hello {name}!" * times

        schema = generate_schema(greet)
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    params: list[ToolParameter] = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, param.annotation)
        type_str = _resolve_json_type(annotation)
        has_default = param.default is not inspect.Parameter.empty
        desc = ""

        params.append(
            ToolParameter(
                name=param_name,
                type_str=type_str,
                description=desc,
                required=not has_default,
                default=param.default if has_default else None,
            )
        )

    return ToolSchema(parameters=params)


def tool_from_function(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolDefinition:
    """Create a ToolDefinition from a Python function.

    Uses the function name and docstring as defaults for the tool name
    and description.

    Args:
        func: The function to wrap.
        name: Override tool name (defaults to function name).
        description: Override description (defaults to first line of docstring).

    Returns:
        A complete ToolDefinition.

    Example::

        def search(query: str, limit: int = 10) -> list[str]:
            \"\"\"Search the knowledge base.\"\"\"
            ...

        tool = tool_from_function(search)
    """
    tool_name = name or func.__name__
    tool_desc = description or _extract_docstring_summary(func) or tool_name

    return ToolDefinition(
        name=tool_name,
        description=tool_desc,
        schema=generate_schema(func),
    )


def _extract_docstring_summary(func: Callable[..., Any]) -> str:
    """Extract the first non-empty line from a function's docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def validate_arguments(
    tool: ToolDefinition,
    arguments: dict[str, object],
) -> dict[str, object]:
    """Validate arguments against a tool's schema.

    Checks that all required parameters are present and that provided
    values have acceptable types.

    Args:
        tool: The tool definition to validate against.
        arguments: The arguments to validate.

    Returns:
        The validated arguments (with defaults filled in).

    Raises:
        ToolValidationError: If required arguments are missing.
    """
    validated = dict(arguments)

    for param in tool.schema.parameters:
        if param.required and param.name not in validated:
            raise ToolValidationError(tool.name, f"Missing required argument: {param.name!r}")
        if not param.required and param.name not in validated:
            validated[param.name] = param.default

    return validated


class ToolRegistry:
    """A registry of tool definitions for agent frameworks.

    Provides registration, lookup, and bulk export of tool definitions.

    Example::

        registry = ToolRegistry()
        registry.register(tool_from_function(my_func))
        tools = registry.to_openai_tools()
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition.

        Args:
            tool: The tool to register.

        Raises:
            ToolRegistryError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ToolRegistryError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        """Look up a tool by name.

        Args:
            name: Tool name.

        Returns:
            The matching tool definition.

        Raises:
            ToolRegistryError: If no tool with that name exists.
        """
        if name not in self._tools:
            raise ToolRegistryError(f"Unknown tool: {name!r}")
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools)

    def to_openai_tools(self) -> list[dict[str, object]]:
        """Export all tools in OpenAI function-calling format.

        Returns:
            A list of tool dicts suitable for the OpenAI API.
        """
        return [t.to_openai_tool() for t in self._tools.values()]
