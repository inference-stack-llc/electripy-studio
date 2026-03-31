"""Domain models for the Model Context Protocol toolkit.

This module defines the core data structures for MCP communication:
tool definitions, requests, responses, and configuration.  All models
are intentionally free of third-party dependencies.

Example:
    from electripy.ai.mcp.domain import MCPToolDefinition, MCPToolCall

    tool = MCPToolDefinition(
        name="get_weather",
        description="Get weather data for a city",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

__all__ = [
    "JsonObject",
    "MCPClientSettings",
    "MCPContent",
    "MCPContentType",
    "MCPErrorData",
    "MCPProtocolVersion",
    "MCPRequest",
    "MCPResponse",
    "MCPServerCapabilities",
    "MCPServerConfig",
    "MCPToolCall",
    "MCPToolDefinition",
    "MCPToolResult",
]

JsonObject: TypeAlias = dict[str, object]


class MCPProtocolVersion(StrEnum):
    """Supported MCP protocol versions."""

    V_2024_11_05 = "2024-11-05"


class MCPContentType(StrEnum):
    """Content types returned by MCP tool invocations."""

    TEXT = "text"
    IMAGE = "image"
    RESOURCE = "resource"


@dataclass(frozen=True, slots=True)
class MCPContent:
    """A single content block in an MCP tool result.

    Attributes:
        type: Content type discriminator.
        text: Text payload when ``type`` is ``TEXT``.
        data: Base64-encoded binary payload when ``type`` is ``IMAGE``.
        mime_type: MIME type for image or resource content.
        uri: Resource URI when ``type`` is ``RESOURCE``.
    """

    type: MCPContentType
    text: str | None = None
    data: str | None = None
    mime_type: str | None = None
    uri: str | None = None

    def to_dict(self) -> JsonObject:
        """Serialize to a JSON-compatible dictionary."""
        d: JsonObject = {"type": self.type.value}
        if self.text is not None:
            d["text"] = self.text
        if self.data is not None:
            d["data"] = self.data
        if self.mime_type is not None:
            d["mimeType"] = self.mime_type
        if self.uri is not None:
            d["uri"] = self.uri
        return d


@dataclass(frozen=True, slots=True)
class MCPToolDefinition:
    """Definition of an MCP-exposed tool.

    Attributes:
        name: Unique tool name.
        description: Human-readable description.
        input_schema: JSON Schema describing the tool's input parameters.
    """

    name: str
    description: str
    input_schema: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """Serialize to MCP wire format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema or {"type": "object", "properties": {}},
        }

    @classmethod
    def from_dict(cls, data: JsonObject) -> MCPToolDefinition:
        """Deserialize from MCP wire format."""
        raw_schema = data.get("inputSchema", {})
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            input_schema=dict(raw_schema) if isinstance(raw_schema, dict) else {},
        )


@dataclass(frozen=True, slots=True)
class MCPToolCall:
    """An MCP tool invocation request.

    Attributes:
        name: Name of the tool to invoke.
        arguments: Input arguments matching the tool's input schema.
        call_id: Optional caller-assigned identifier for correlation.
    """

    name: str
    arguments: JsonObject = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class MCPToolResult:
    """Result of an MCP tool invocation.

    Attributes:
        content: List of content blocks produced by the tool.
        is_error: Whether the tool reported a logical error.
        call_id: Correlation identifier echoed from the call.
    """

    content: list[MCPContent] = field(default_factory=list)
    is_error: bool = False
    call_id: str | None = None

    def to_dict(self) -> JsonObject:
        """Serialize to MCP wire format."""
        return {
            "content": [c.to_dict() for c in self.content],
            "isError": self.is_error,
        }


@dataclass(frozen=True, slots=True)
class MCPServerCapabilities:
    """Capabilities advertised by an MCP server.

    Attributes:
        tools: Whether the server exposes tools.
        resources: Whether the server exposes resources.
        prompts: Whether the server exposes prompts.
    """

    tools: bool = False
    resources: bool = False
    prompts: bool = False

    def to_dict(self) -> JsonObject:
        """Serialize to MCP wire format."""
        caps: JsonObject = {}
        if self.tools:
            caps["tools"] = {}
        if self.resources:
            caps["resources"] = {}
        if self.prompts:
            caps["prompts"] = {}
        return caps

    @classmethod
    def from_dict(cls, data: JsonObject) -> MCPServerCapabilities:
        """Deserialize from MCP wire format."""
        return cls(
            tools="tools" in data,
            resources="resources" in data,
            prompts="prompts" in data,
        )


@dataclass(frozen=True, slots=True)
class MCPErrorData:
    """Structured error payload in an MCP response.

    Attributes:
        code: JSON-RPC error code.
        message: Human-readable error message.
        data: Optional additional error data.
    """

    code: int
    message: str
    data: object | None = None

    def to_dict(self) -> JsonObject:
        """Serialize to JSON-RPC format."""
        d: JsonObject = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


@dataclass(frozen=True, slots=True)
class MCPRequest:
    """JSON-RPC 2.0 request used by the MCP protocol.

    Attributes:
        method: RPC method name.
        params: Optional method parameters.
        id: Request identifier; ``None`` for notifications.
        jsonrpc: JSON-RPC version string.
    """

    method: str
    params: JsonObject | None = None
    id: str | int | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> JsonObject:
        """Serialize to JSON-compatible dictionary."""
        d: JsonObject = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            d["params"] = self.params
        if self.id is not None:
            d["id"] = self.id
        return d


@dataclass(frozen=True, slots=True)
class MCPResponse:
    """JSON-RPC 2.0 response received from an MCP server.

    Attributes:
        id: Correlated request identifier.
        result: Successful result payload.
        error: Error payload when the request failed.
        jsonrpc: JSON-RPC version string.
    """

    id: str | int | None = None
    result: JsonObject | None = None
    error: MCPErrorData | None = None
    jsonrpc: str = "2.0"

    @property
    def is_error(self) -> bool:
        """Return ``True`` when the response carries an error."""
        return self.error is not None

    def to_dict(self) -> JsonObject:
        """Serialize to JSON-compatible dictionary."""
        d: JsonObject = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            d["id"] = self.id
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: JsonObject) -> MCPResponse:
        """Deserialize from a JSON-compatible dictionary."""
        error_data = data.get("error")
        error: MCPErrorData | None = None
        if isinstance(error_data, dict):
            raw_code = error_data.get("code")
            code = int(raw_code) if isinstance(raw_code, (int, float)) else -1
            error = MCPErrorData(
                code=code,
                message=str(error_data.get("message", "")),
                data=error_data.get("data"),
            )
        raw_result = data.get("result")
        result = dict(raw_result) if isinstance(raw_result, dict) else None
        raw_id = data.get("id")
        parsed_id: str | int | None = None
        if isinstance(raw_id, (str, int)):
            parsed_id = raw_id
        return cls(
            id=parsed_id,
            result=result,
            error=error,
            jsonrpc=str(data.get("jsonrpc", "2.0")),
        )


@dataclass(slots=True)
class MCPServerConfig:
    """Configuration for an MCP tool server.

    Attributes:
        name: Server display name.
        version: Server version string.
        protocol_version: MCP protocol version to advertise.
        capabilities: Server capabilities to advertise.
    """

    name: str
    version: str = "1.0.0"
    protocol_version: str = MCPProtocolVersion.V_2024_11_05
    capabilities: MCPServerCapabilities = field(
        default_factory=lambda: MCPServerCapabilities(tools=True)
    )


@dataclass(slots=True)
class MCPClientSettings:
    """Settings for the MCP client.

    Attributes:
        name: Client display name sent during initialization.
        version: Client version string.
        protocol_version: MCP protocol version requested.
        timeout_s: Default request timeout in seconds.
    """

    name: str = "electripy-mcp-client"
    version: str = "1.0.0"
    protocol_version: str = MCPProtocolVersion.V_2024_11_05
    timeout_s: float = 30.0
