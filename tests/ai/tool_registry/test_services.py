from __future__ import annotations

import pytest

from electripy.ai.tool_registry import (
    ToolParameter,
    ToolRegistry,
    ToolRegistryError,
    ToolSchema,
    ToolValidationError,
    generate_schema,
    tool_from_function,
    validate_arguments,
)


def _sample_func(query: str, limit: int = 10) -> list[str]:
    """Search the knowledge base."""
    return []


class TestGenerateSchema:
    def test_basic_function(self) -> None:
        schema = generate_schema(_sample_func)
        assert len(schema.parameters) == 2

        query_param = schema.parameters[0]
        assert query_param.name == "query"
        assert query_param.type_str == "string"
        assert query_param.required is True

        limit_param = schema.parameters[1]
        assert limit_param.name == "limit"
        assert limit_param.type_str == "integer"
        assert limit_param.required is False
        assert limit_param.default == 10

    def test_no_params(self) -> None:
        def noop() -> None:
            pass

        schema = generate_schema(noop)
        assert schema.parameters == []


class TestToolFromFunction:
    def test_infers_name_and_description(self) -> None:
        tool = tool_from_function(_sample_func)
        assert tool.name == "_sample_func"
        assert tool.description == "Search the knowledge base."

    def test_override_name_and_description(self) -> None:
        tool = tool_from_function(_sample_func, name="search", description="Custom desc")
        assert tool.name == "search"
        assert tool.description == "Custom desc"


class TestValidateArguments:
    def test_valid_args(self) -> None:
        tool = tool_from_function(_sample_func)
        result = validate_arguments(tool, {"query": "test"})
        assert result["query"] == "test"
        assert result["limit"] == 10  # default filled in

    def test_missing_required(self) -> None:
        tool = tool_from_function(_sample_func)
        with pytest.raises(ToolValidationError, match="query"):
            validate_arguments(tool, {})


class TestToolSchema:
    def test_to_json_schema(self) -> None:
        schema = ToolSchema(
            parameters=[
                ToolParameter(name="q", type_str="string", required=True),
                ToolParameter(name="n", type_str="integer", required=False, default=5),
            ]
        )
        js = schema.to_json_schema()
        assert js["type"] == "object"
        assert "q" in js["properties"]  # type: ignore[operator]
        assert js["required"] == ["q"]


class TestToolDefinition:
    def test_to_openai_tool(self) -> None:
        tool = tool_from_function(_sample_func, name="search")
        openai_fmt = tool.to_openai_tool()
        assert openai_fmt["type"] == "function"
        func_obj = openai_fmt["function"]
        assert isinstance(func_obj, dict)
        assert func_obj["name"] == "search"


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = tool_from_function(_sample_func, name="search")
        registry.register(tool)
        assert registry.get("search").name == "search"
        assert "search" in registry.names

    def test_duplicate_registration_raises(self) -> None:
        registry = ToolRegistry()
        tool = tool_from_function(_sample_func, name="search")
        registry.register(tool)
        with pytest.raises(ToolRegistryError, match="already registered"):
            registry.register(tool)

    def test_unknown_tool_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError, match="Unknown"):
            registry.get("nope")

    def test_to_openai_tools(self) -> None:
        registry = ToolRegistry()
        tool = tool_from_function(_sample_func, name="search")
        registry.register(tool)
        tools = registry.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
