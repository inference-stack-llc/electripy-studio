"""Model Context Protocol toolkit for MCP clients and server adapters.

Purpose:
  - Provide a provider-neutral, strongly typed MCP interface.
  - Enable teams to connect to MCP servers and expose tools via MCP.
  - Integrate with ElectriPy observability and tool-registry ecosystems.

Guarantees:
  - All MCP interactions go through Protocol ports, allowing any transport.
  - Domain models are free of third-party dependencies.

Usage:
  Client example::

    from electripy.ai.mcp import (
        MCPClient,
        MCPToolServer,
        MCPServerConfig,
        MCPToolDefinition,
        InMemoryTransportAdapter,
    )

    server = MCPToolServer(MCPServerConfig(name="my-server"))
    client = MCPClient(
        transport=InMemoryTransportAdapter(handler=server.handle_request),
    )
    caps = client.initialize()
    tools = client.list_tools()
"""

from __future__ import annotations

from .adapters import (
    AsyncHttpTransportAdapter,
    BearerTokenAuthAdapter,
    HttpTransportAdapter,
    InMemoryTransportAdapter,
    NoOpAuthAdapter,
    StdioTransportAdapter,
)
from .domain import (
    JsonObject,
    MCPClientSettings,
    MCPContent,
    MCPContentType,
    MCPErrorData,
    MCPProtocolVersion,
    MCPRequest,
    MCPResponse,
    MCPServerCapabilities,
    MCPServerConfig,
    MCPToolCall,
    MCPToolDefinition,
    MCPToolResult,
)
from .errors import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTimeoutError,
    MCPToolExecutionError,
)
from .ports import (
    AsyncMCPTransportPort,
    MCPAuthPort,
    MCPToolHandlerPort,
    MCPTransportPort,
)
from .services import (
    AsyncMCPClient,
    MCPClient,
    MCPToolServer,
)

__all__ = [
    # Domain
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
    # Errors
    "MCPAuthenticationError",
    "MCPConnectionError",
    "MCPError",
    "MCPProtocolError",
    "MCPTimeoutError",
    "MCPToolExecutionError",
    # Ports
    "AsyncMCPTransportPort",
    "MCPAuthPort",
    "MCPToolHandlerPort",
    "MCPTransportPort",
    # Adapters
    "AsyncHttpTransportAdapter",
    "BearerTokenAuthAdapter",
    "HttpTransportAdapter",
    "InMemoryTransportAdapter",
    "NoOpAuthAdapter",
    "StdioTransportAdapter",
    # Services
    "AsyncMCPClient",
    "MCPClient",
    "MCPToolServer",
]
