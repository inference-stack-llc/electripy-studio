"""Ports (Protocol interfaces) for the MCP toolkit.

These runtime-checkable protocols define the pluggable boundaries of the
MCP package:

- **Transport** — how JSON-RPC messages reach the server.
- **Auth** — how authentication headers are provided.
- **Tool handler** — how incoming tool calls are dispatched.

All concrete implementations live in :mod:`electripy.ai.mcp.adapters`.

Example:
    from electripy.ai.mcp.ports import MCPTransportPort

    class MyTransport(MCPTransportPort):
        def send(self, request: MCPRequest) -> MCPResponse:
            ...
        def close(self) -> None:
            ...
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .domain import MCPRequest, MCPResponse, MCPToolCall, MCPToolResult

__all__ = [
    "AsyncMCPTransportPort",
    "MCPAuthPort",
    "MCPToolHandlerPort",
    "MCPTransportPort",
]


@runtime_checkable
class MCPTransportPort(Protocol):
    """Synchronous transport for MCP JSON-RPC communication.

    Implementations must serialize the request, deliver it to the MCP
    server over the chosen transport mechanism, and return the
    deserialized response.
    """

    def send(self, request: MCPRequest) -> MCPResponse:
        """Send a request and return the response.

        Args:
            request: The MCP request to send.

        Returns:
            The server's response.

        Raises:
            MCPConnectionError: If the server is unreachable.
            MCPTimeoutError: If the request times out.
        """
        ...

    def close(self) -> None:
        """Release transport resources."""
        ...


@runtime_checkable
class AsyncMCPTransportPort(Protocol):
    """Asynchronous transport for MCP JSON-RPC communication."""

    async def send(self, request: MCPRequest) -> MCPResponse:
        """Send a request and return the response asynchronously.

        Args:
            request: The MCP request to send.

        Returns:
            The server's response.
        """
        ...

    async def close(self) -> None:
        """Release transport resources."""
        ...


@runtime_checkable
class MCPAuthPort(Protocol):
    """Authentication provider for MCP connections.

    Implementations return headers (e.g. ``Authorization``) that the
    transport layer should include in outgoing requests.
    """

    def get_headers(self) -> dict[str, str]:
        """Return authentication headers.

        Returns:
            Mapping of header names to values.
        """
        ...


@runtime_checkable
class MCPToolHandlerPort(Protocol):
    """Handler for an individual MCP tool invocation.

    Server-side code registers one handler per tool.  The handler
    receives a :class:`MCPToolCall` and returns a :class:`MCPToolResult`.
    """

    def handle(self, call: MCPToolCall) -> MCPToolResult:
        """Execute the tool and return a result.

        Args:
            call: The incoming tool call.

        Returns:
            Tool execution result.
        """
        ...
