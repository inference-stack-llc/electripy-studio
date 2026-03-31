"""Tests for electripy.ai.mcp.errors."""

from __future__ import annotations

import pytest

from electripy.ai.mcp.errors import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTimeoutError,
    MCPToolExecutionError,
)
from electripy.core.errors import ElectriPyError


class TestMCPErrorHierarchy:
    def test_mcp_error_extends_electripy(self) -> None:
        assert issubclass(MCPError, ElectriPyError)

    def test_connection_error(self) -> None:
        assert issubclass(MCPConnectionError, MCPError)

    def test_timeout_error(self) -> None:
        assert issubclass(MCPTimeoutError, MCPError)

    def test_auth_error(self) -> None:
        assert issubclass(MCPAuthenticationError, MCPError)

    def test_protocol_error(self) -> None:
        assert issubclass(MCPProtocolError, MCPError)

    def test_tool_execution_error(self) -> None:
        assert issubclass(MCPToolExecutionError, MCPError)


class TestMCPToolExecutionError:
    def test_tool_name(self) -> None:
        exc = MCPToolExecutionError("echo", "failed")
        assert exc.tool_name == "echo"
        assert str(exc) == "failed"

    def test_default_message(self) -> None:
        exc = MCPToolExecutionError("echo")
        assert "echo" in str(exc)

    def test_details(self) -> None:
        exc = MCPToolExecutionError("echo", details="timeout")
        assert exc.details == "timeout"

    def test_catch_as_mcp_error(self) -> None:
        with pytest.raises(MCPError):
            raise MCPToolExecutionError("echo", "boom")

    def test_catch_as_electripy_error(self) -> None:
        with pytest.raises(ElectriPyError):
            raise MCPToolExecutionError("echo", "boom")


class TestErrorInstances:
    def test_connection_error_str(self) -> None:
        exc = MCPConnectionError("server unreachable")
        assert "server unreachable" in str(exc)

    def test_timeout_error_str(self) -> None:
        exc = MCPTimeoutError("30s exceeded")
        assert "30s exceeded" in str(exc)

    def test_auth_error_str(self) -> None:
        exc = MCPAuthenticationError("bad token")
        assert "bad token" in str(exc)

    def test_protocol_error_str(self) -> None:
        exc = MCPProtocolError("invalid jsonrpc")
        assert "invalid jsonrpc" in str(exc)
