"""Domain models for the tool registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ToolParameter:
    """Describes a single parameter of a tool.

    Attributes:
        name: Parameter name.
        type_str: JSON Schema type string (e.g. "string", "integer").
        description: Human-readable parameter description.
        required: Whether this parameter is required.
        default: Default value, if any.
        enum: Allowed values, if constrained.
    """

    name: str
    type_str: str
    description: str = ""
    required: bool = True
    default: object = None
    enum: list[str] | None = None


@dataclass(slots=True)
class ToolSchema:
    """JSON Schema representation of a tool's parameters.

    Attributes:
        parameters: List of parameter definitions.
    """

    parameters: list[ToolParameter] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, object]:
        """Convert to an OpenAI-compatible JSON Schema dict.

        Returns:
            A dict with ``type``, ``properties``, and ``required`` keys.
        """
        properties: dict[str, object] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, object] = {"type": param.type_str}
            if param.description:
                prop["description"] = param.description
            if param.enum is not None:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        result: dict[str, object] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            result["required"] = required
        return result


@dataclass(slots=True)
class ToolDefinition:
    """Complete definition of an agent tool.

    Attributes:
        name: Tool name (used in function-calling APIs).
        description: Human-readable description of what the tool does.
        schema: Parameter schema.
    """

    name: str
    description: str
    schema: ToolSchema

    def to_openai_tool(self) -> dict[str, object]:
        """Convert to OpenAI function-calling tool format.

        Returns:
            A dict matching the OpenAI ``tools`` array element format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema.to_json_schema(),
            },
        }
