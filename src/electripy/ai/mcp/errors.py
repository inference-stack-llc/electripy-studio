"""Errors for the Model Context Protocol toolkit.

All MCP exceptions derive from :class:`MCPError` which itself extends
:class:`ElectriPyError`, keeping the error hierarchy consistent with
the rest of the ElectriPy codebase.

Example:
    from electripy.ai.mcp.errors import MCPToolExecutionError

    try:
        client.call_tool("missing_tool", {})
    except MCPToolExecutionError as exc:
        print(f"Tool {exc.tool_name} failed: {exc}")
"""

from __future__ import annotations

from electripy.core.errors import ElectriPyError

__all__ = [
    "MCPAuthenticationError",
    "MCPConnectionError",
    "MCPError",
    "MCPProtocolError",
    "MCPTimeoutError",
    "MCPToolExecutionError",
]


class MCPError(ElectriPyError):
    """Base exception for all MCP-related failures."""


class MCPConnectionError(MCPError):
    """Raised when the transport cannot reach the MCP server."""


class MCPTimeoutError(MCPError):
    """Raised when an MCP request exceeds the allowed timeout."""


class MCPAuthenticationError(MCPError):
    """Raised when MCP server authentication fails."""


class MCPProtocolError(MCPError):
    """Raised for JSON-RPC or MCP protocol violations."""


class MCPToolExecutionError(MCPError):
    """Raised when a tool call fails on the server side.

    Attributes:
        tool_name: Name of the tool that failed.
        details: Optional additional context.
    """

    def __init__(
        self,
        tool_name: str,
        message: str | None = None,
        *,
        details: str | None = None,
    ) -> None:
        full = message or f"Tool execution failed: {tool_name}"
        super().__init__(full)
        self.tool_name = tool_name
        self.details = details
