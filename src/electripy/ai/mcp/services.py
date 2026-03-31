"""Services for MCP client and tool server orchestration.

This module provides the main entry points:

- :class:`MCPClient` — connects to an MCP server, discovers tools,
  and invokes them.
- :class:`AsyncMCPClient` — asynchronous equivalent.
- :class:`MCPToolServer` — in-process server that registers tool
  handlers and dispatches incoming MCP requests.

Example:
    from electripy.ai.mcp import (
        MCPClient,
        MCPClientSettings,
        MCPToolServer,
        MCPServerConfig,
        MCPToolDefinition,
        InMemoryTransportAdapter,
    )

    server = MCPToolServer(MCPServerConfig(name="my-server"))
    server.register_tool(
        MCPToolDefinition(name="echo", description="Echo input"),
        handler=my_echo_handler,
    )

    client = MCPClient(
        transport=InMemoryTransportAdapter(handler=server.handle_request),
    )
    capabilities = client.initialize()
    tools = client.list_tools()
    result = client.call_tool("echo", {"text": "hello"})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from electripy.core.logging import get_logger

from .domain import (
    JsonObject,
    MCPClientSettings,
    MCPContent,
    MCPContentType,
    MCPErrorData,
    MCPRequest,
    MCPResponse,
    MCPServerCapabilities,
    MCPServerConfig,
    MCPToolCall,
    MCPToolDefinition,
    MCPToolResult,
)
from .errors import MCPError, MCPProtocolError, MCPToolExecutionError
from .ports import AsyncMCPTransportPort, MCPToolHandlerPort, MCPTransportPort

__all__ = [
    "AsyncMCPClient",
    "MCPClient",
    "MCPToolServer",
]

logger = get_logger(__name__)


# ── MCPClient (synchronous) ─────────────────────────────────────────


@dataclass(slots=True)
class MCPClient:
    """Synchronous MCP client.

    Orchestrates the MCP protocol lifecycle: initialization, tool
    discovery, and tool invocation over a pluggable transport.

    Attributes:
        transport: Transport adapter for server communication.
        settings: Client identity and protocol settings.
    """

    transport: MCPTransportPort
    settings: MCPClientSettings = field(default_factory=MCPClientSettings)
    _request_id: int = field(default=0, init=False, repr=False)
    _capabilities: MCPServerCapabilities | None = field(default=None, init=False, repr=False)
    _tools: list[MCPToolDefinition] = field(default_factory=list, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    # ── internal helpers ─────────────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise MCPProtocolError("Client has not been initialized; call initialize() first")

    def _check_response(self, response: MCPResponse, context: str) -> JsonObject:
        if response.is_error:
            assert response.error is not None  # noqa: S101
            raise MCPProtocolError(f"{context}: {response.error.message}")
        return response.result or {}

    # ── public API ───────────────────────────────────────────────────

    def initialize(self) -> MCPServerCapabilities:
        """Perform the MCP initialize handshake.

        Sends an ``initialize`` request followed by an
        ``notifications/initialized`` notification.  Returns the server's
        advertised capabilities.

        Returns:
            The server's capabilities.

        Raises:
            MCPProtocolError: If the server rejects initialization.
        """
        request = MCPRequest(
            method="initialize",
            params={
                "protocolVersion": self.settings.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": self.settings.name,
                    "version": self.settings.version,
                },
            },
            id=self._next_id(),
        )
        response = self.transport.send(request)
        result = self._check_response(response, "initialize")

        raw_caps = result.get("capabilities", {})
        self._capabilities = (
            MCPServerCapabilities.from_dict(raw_caps)
            if isinstance(raw_caps, dict)
            else MCPServerCapabilities()
        )

        # Send initialized notification (no response expected).
        self.transport.send(MCPRequest(method="notifications/initialized"))
        self._initialized = True
        logger.info("MCP client initialized", extra={"server_caps": self._capabilities})
        return self._capabilities

    @property
    def capabilities(self) -> MCPServerCapabilities | None:
        """Server capabilities discovered during initialization."""
        return self._capabilities

    @property
    def tools(self) -> list[MCPToolDefinition]:
        """Tools discovered by the most recent :meth:`list_tools` call."""
        return list(self._tools)

    def list_tools(self) -> list[MCPToolDefinition]:
        """Fetch the list of tools available on the server.

        Returns:
            List of tool definitions.

        Raises:
            MCPProtocolError: If the client is not initialized or the
                server returns an error.
        """
        self._ensure_initialized()
        request = MCPRequest(method="tools/list", id=self._next_id())
        response = self.transport.send(request)
        result = self._check_response(response, "tools/list")

        raw_tools = result.get("tools", [])
        if not isinstance(raw_tools, list):
            raw_tools = []
        self._tools = [
            MCPToolDefinition.from_dict(t) for t in raw_tools if isinstance(t, dict)
        ]
        return list(self._tools)

    def call_tool(
        self,
        name: str,
        arguments: JsonObject | None = None,
    ) -> MCPToolResult:
        """Invoke a tool on the MCP server.

        Args:
            name: Tool name.
            arguments: Input arguments for the tool.

        Returns:
            The tool result.

        Raises:
            MCPProtocolError: If the client is not initialized.
            MCPToolExecutionError: If the server reports a tool error.
        """
        self._ensure_initialized()
        start = time.monotonic()
        request = MCPRequest(
            method="tools/call",
            params={"name": name, "arguments": arguments or {}},
            id=self._next_id(),
        )
        response = self.transport.send(request)
        elapsed_ms = (time.monotonic() - start) * 1000

        if response.is_error:
            assert response.error is not None  # noqa: S101
            logger.warning(
                "MCP tool call failed",
                extra={"tool": name, "elapsed_ms": elapsed_ms},
            )
            raise MCPToolExecutionError(name, response.error.message)

        result = response.result or {}
        content = _parse_content_list(result.get("content", []))
        is_error = bool(result.get("isError", False))

        if is_error:
            logger.warning(
                "MCP tool returned error content",
                extra={"tool": name, "elapsed_ms": elapsed_ms},
            )
        logger.debug(
            "MCP tool call completed",
            extra={"tool": name, "elapsed_ms": elapsed_ms, "is_error": is_error},
        )
        return MCPToolResult(content=content, is_error=is_error)

    def ping(self) -> bool:
        """Send a ping to the MCP server.

        Returns:
            ``True`` if the server responds successfully.
        """
        try:
            response = self.transport.send(MCPRequest(method="ping", id=self._next_id()))
            return not response.is_error
        except MCPError:
            return False

    def close(self) -> None:
        """Close the transport connection."""
        self.transport.close()


# ── AsyncMCPClient ───────────────────────────────────────────────────


@dataclass(slots=True)
class AsyncMCPClient:
    """Asynchronous MCP client.

    Mirrors :class:`MCPClient` but uses an async transport port.

    Attributes:
        transport: Async transport adapter for server communication.
        settings: Client identity and protocol settings.
    """

    transport: AsyncMCPTransportPort
    settings: MCPClientSettings = field(default_factory=MCPClientSettings)
    _request_id: int = field(default=0, init=False, repr=False)
    _capabilities: MCPServerCapabilities | None = field(default=None, init=False, repr=False)
    _tools: list[MCPToolDefinition] = field(default_factory=list, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise MCPProtocolError("Client has not been initialized; call initialize() first")

    def _check_response(self, response: MCPResponse, context: str) -> JsonObject:
        if response.is_error:
            assert response.error is not None  # noqa: S101
            raise MCPProtocolError(f"{context}: {response.error.message}")
        return response.result or {}

    async def initialize(self) -> MCPServerCapabilities:
        """Perform the MCP initialize handshake asynchronously."""
        request = MCPRequest(
            method="initialize",
            params={
                "protocolVersion": self.settings.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": self.settings.name,
                    "version": self.settings.version,
                },
            },
            id=self._next_id(),
        )
        response = await self.transport.send(request)
        result = self._check_response(response, "initialize")

        raw_caps = result.get("capabilities", {})
        self._capabilities = (
            MCPServerCapabilities.from_dict(raw_caps)
            if isinstance(raw_caps, dict)
            else MCPServerCapabilities()
        )

        await self.transport.send(MCPRequest(method="notifications/initialized"))
        self._initialized = True
        return self._capabilities

    @property
    def capabilities(self) -> MCPServerCapabilities | None:
        """Server capabilities discovered during initialization."""
        return self._capabilities

    @property
    def tools(self) -> list[MCPToolDefinition]:
        """Tools discovered by the most recent :meth:`list_tools` call."""
        return list(self._tools)

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Fetch tool list asynchronously."""
        self._ensure_initialized()
        request = MCPRequest(method="tools/list", id=self._next_id())
        response = await self.transport.send(request)
        result = self._check_response(response, "tools/list")

        raw_tools = result.get("tools", [])
        if not isinstance(raw_tools, list):
            raw_tools = []
        self._tools = [
            MCPToolDefinition.from_dict(t) for t in raw_tools if isinstance(t, dict)
        ]
        return list(self._tools)

    async def call_tool(
        self,
        name: str,
        arguments: JsonObject | None = None,
    ) -> MCPToolResult:
        """Invoke a tool asynchronously."""
        self._ensure_initialized()
        start = time.monotonic()
        request = MCPRequest(
            method="tools/call",
            params={"name": name, "arguments": arguments or {}},
            id=self._next_id(),
        )
        response = await self.transport.send(request)
        elapsed_ms = (time.monotonic() - start) * 1000

        if response.is_error:
            assert response.error is not None  # noqa: S101
            raise MCPToolExecutionError(name, response.error.message)

        result = response.result or {}
        content = _parse_content_list(result.get("content", []))
        is_error = bool(result.get("isError", False))
        logger.debug(
            "MCP async tool call completed",
            extra={"tool": name, "elapsed_ms": elapsed_ms, "is_error": is_error},
        )
        return MCPToolResult(content=content, is_error=is_error)

    async def ping(self) -> bool:
        """Send a ping asynchronously."""
        try:
            response = await self.transport.send(
                MCPRequest(method="ping", id=self._next_id())
            )
            return not response.is_error
        except MCPError:
            return False

    async def close(self) -> None:
        """Close the async transport."""
        await self.transport.close()


# ── MCPToolServer ────────────────────────────────────────────────────


class MCPToolServer:
    """In-process MCP tool server.

    Registers tool definitions paired with handlers and dispatches
    incoming :class:`MCPRequest` objects according to the MCP protocol.

    Args:
        config: Server identity and capability configuration.

    Example:
        server = MCPToolServer(MCPServerConfig(name="demo"))
        server.register_tool(tool_def, handler)
        response = server.handle_request(request)
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._tools: dict[str, tuple[MCPToolDefinition, MCPToolHandlerPort]] = {}

    @property
    def config(self) -> MCPServerConfig:
        """Server configuration."""
        return self._config

    @property
    def tool_names(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools)

    def register_tool(
        self,
        definition: MCPToolDefinition,
        handler: MCPToolHandlerPort,
    ) -> None:
        """Register a tool with its handler.

        Args:
            definition: Tool metadata and input schema.
            handler: Handler that executes when the tool is called.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name!r}")
        self._tools[definition.name] = (definition, handler)
        logger.debug("Registered MCP tool", extra={"tool": definition.name})

    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Dispatch an incoming MCP request.

        Routes the request to the appropriate internal handler based on
        the JSON-RPC method name.

        Args:
            request: Incoming MCP request.

        Returns:
            MCP response.
        """
        method = request.method
        if method == "initialize":
            return self._handle_initialize(request)
        if method == "notifications/initialized":
            return MCPResponse(id=request.id)
        if method == "tools/list":
            return self._handle_tools_list(request)
        if method == "tools/call":
            return self._handle_tools_call(request)
        if method == "ping":
            return MCPResponse(id=request.id, result={})
        return MCPResponse(
            id=request.id,
            error=MCPErrorData(code=-32601, message=f"Method not found: {method}"),
        )

    # ── private handlers ─────────────────────────────────────────────

    def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        return MCPResponse(
            id=request.id,
            result={
                "protocolVersion": self._config.protocol_version,
                "capabilities": self._config.capabilities.to_dict(),
                "serverInfo": {
                    "name": self._config.name,
                    "version": self._config.version,
                },
            },
        )

    def _handle_tools_list(self, request: MCPRequest) -> MCPResponse:
        tools_list = [defn.to_dict() for defn, _ in self._tools.values()]
        return MCPResponse(id=request.id, result={"tools": tools_list})

    def _handle_tools_call(self, request: MCPRequest) -> MCPResponse:
        params = request.params or {}
        tool_name = str(params.get("name", ""))
        arguments = params.get("arguments", {})

        if tool_name not in self._tools:
            return MCPResponse(
                id=request.id,
                error=MCPErrorData(code=-32602, message=f"Unknown tool: {tool_name!r}"),
            )

        _, handler = self._tools[tool_name]
        call = MCPToolCall(
            name=tool_name,
            arguments=dict(arguments) if isinstance(arguments, dict) else {},
        )

        try:
            tool_result = handler.handle(call)
        except Exception as exc:
            logger.error(
                "MCP tool handler raised", extra={"tool": tool_name, "error": str(exc)}
            )
            return MCPResponse(
                id=request.id,
                result={
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            )

        return MCPResponse(id=request.id, result=tool_result.to_dict())


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_content_list(raw: object) -> list[MCPContent]:
    """Parse a list of content dicts into MCPContent objects."""
    if not isinstance(raw, list):
        return []
    result: list[MCPContent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text_val = item.get("text")
        data_val = item.get("data")
        mime_val = item.get("mimeType")
        uri_val = item.get("uri")
        result.append(
            MCPContent(
                type=MCPContentType(str(item.get("type", "text"))),
                text=str(text_val) if text_val is not None else None,
                data=str(data_val) if data_val is not None else None,
                mime_type=str(mime_val) if mime_val is not None else None,
                uri=str(uri_val) if uri_val is not None else None,
            )
        )
    return result
