"""Tests for electripy.ai.mcp.services."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from electripy.ai.mcp.adapters import InMemoryTransportAdapter
from electripy.ai.mcp.domain import (
    MCPClientSettings,
    MCPContent,
    MCPContentType,
    MCPRequest,
    MCPResponse,
    MCPServerCapabilities,
    MCPServerConfig,
    MCPToolCall,
    MCPToolDefinition,
    MCPToolResult,
)
from electripy.ai.mcp.errors import MCPConnectionError, MCPProtocolError, MCPToolExecutionError
from electripy.ai.mcp.services import AsyncMCPClient, MCPClient, MCPToolServer

# ── Test helpers ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EchoHandler:
    """Test tool handler that echoes input."""

    def handle(self, call: MCPToolCall) -> MCPToolResult:
        text = str(call.arguments.get("text", ""))
        return MCPToolResult(
            content=[MCPContent(type=MCPContentType.TEXT, text=f"echo: {text}")],
        )


@dataclass(frozen=True, slots=True)
class FailHandler:
    """Test tool handler that always raises."""

    def handle(self, call: MCPToolCall) -> MCPToolResult:
        raise RuntimeError("boom")


@dataclass(frozen=True, slots=True)
class ErrorContentHandler:
    """Test tool handler that returns error content."""

    def handle(self, call: MCPToolCall) -> MCPToolResult:
        return MCPToolResult(
            content=[MCPContent(type=MCPContentType.TEXT, text="something went wrong")],
            is_error=True,
        )


@dataclass(frozen=True, slots=True)
class UpperHandler:
    """Test tool handler that upper-cases input text."""

    def handle(self, call: MCPToolCall) -> MCPToolResult:
        text = str(call.arguments.get("text", ""))
        return MCPToolResult(
            content=[MCPContent(type=MCPContentType.TEXT, text=text.upper())],
        )


class _AsyncInMemoryTransport:
    """Async wrapper around MCPToolServer for testing."""

    def __init__(self, server: MCPToolServer) -> None:
        self._server = server

    async def send(self, request: MCPRequest) -> MCPResponse:
        return self._server.handle_request(request)

    async def close(self) -> None:
        pass


def _make_wired_pair(
    *,
    server_name: str = "test-server",
) -> tuple[MCPClient, MCPToolServer]:
    """Create a client/server pair wired with in-memory transport."""
    server = MCPToolServer(MCPServerConfig(name=server_name))
    client = MCPClient(
        transport=InMemoryTransportAdapter(handler=server.handle_request),
    )
    return client, server


# ── MCPToolServer Tests ──────────────────────────────────────────────


class TestMCPToolServer:
    def test_register_tool(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        defn = MCPToolDefinition(name="echo", description="Echo")
        server.register_tool(defn, EchoHandler())
        assert "echo" in server.tool_names

    def test_register_duplicate_raises(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        defn = MCPToolDefinition(name="echo", description="Echo")
        server.register_tool(defn, EchoHandler())
        with pytest.raises(ValueError, match="already registered"):
            server.register_tool(defn, EchoHandler())

    def test_handle_initialize(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="my-server", version="2.0"))
        resp = server.handle_request(MCPRequest(method="initialize", id=1))
        assert not resp.is_error
        assert resp.result is not None
        assert resp.result["serverInfo"]["name"] == "my-server"
        assert resp.result["serverInfo"]["version"] == "2.0"

    def test_handle_initialize_capabilities(self) -> None:
        server = MCPToolServer(
            MCPServerConfig(
                name="t",
                capabilities=MCPServerCapabilities(tools=True, resources=True),
            )
        )
        resp = server.handle_request(MCPRequest(method="initialize", id=1))
        caps = resp.result["capabilities"]
        assert "tools" in caps
        assert "resources" in caps

    def test_handle_ping(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        resp = server.handle_request(MCPRequest(method="ping", id=1))
        assert not resp.is_error
        assert resp.result == {}

    def test_handle_tools_list_empty(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        resp = server.handle_request(MCPRequest(method="tools/list", id=1))
        assert not resp.is_error
        assert resp.result["tools"] == []

    def test_handle_tools_list(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        defn = MCPToolDefinition(name="echo", description="Echo input")
        server.register_tool(defn, EchoHandler())
        resp = server.handle_request(MCPRequest(method="tools/list", id=1))
        assert not resp.is_error
        tools = resp.result["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"

    def test_handle_tools_list_multiple(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        server.register_tool(
            MCPToolDefinition(name="echo", description="Echo"), EchoHandler()
        )
        server.register_tool(
            MCPToolDefinition(name="upper", description="Upper"), UpperHandler()
        )
        resp = server.handle_request(MCPRequest(method="tools/list", id=1))
        names = [t["name"] for t in resp.result["tools"]]
        assert "echo" in names
        assert "upper" in names

    def test_handle_tools_call(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        defn = MCPToolDefinition(name="echo", description="Echo")
        server.register_tool(defn, EchoHandler())
        resp = server.handle_request(
            MCPRequest(
                method="tools/call",
                params={"name": "echo", "arguments": {"text": "hello"}},
                id=1,
            )
        )
        assert not resp.is_error
        assert resp.result["content"][0]["text"] == "echo: hello"

    def test_handle_tools_call_unknown(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        resp = server.handle_request(
            MCPRequest(
                method="tools/call",
                params={"name": "missing"},
                id=1,
            )
        )
        assert resp.is_error
        assert "Unknown tool" in resp.error.message

    def test_handle_tools_call_handler_exception(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        defn = MCPToolDefinition(name="fail", description="Fails")
        server.register_tool(defn, FailHandler())
        resp = server.handle_request(
            MCPRequest(
                method="tools/call",
                params={"name": "fail", "arguments": {}},
                id=1,
            )
        )
        assert not resp.is_error  # Wrapped as error content, not a protocol error
        assert resp.result["isError"] is True
        assert "boom" in resp.result["content"][0]["text"]

    def test_handle_tools_call_error_content(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        defn = MCPToolDefinition(name="err", description="Err")
        server.register_tool(defn, ErrorContentHandler())
        resp = server.handle_request(
            MCPRequest(
                method="tools/call",
                params={"name": "err", "arguments": {}},
                id=1,
            )
        )
        assert resp.result["isError"] is True

    def test_handle_unknown_method(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        resp = server.handle_request(MCPRequest(method="unknown/method", id=1))
        assert resp.is_error
        assert "Method not found" in resp.error.message

    def test_handle_notification(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        resp = server.handle_request(MCPRequest(method="notifications/initialized"))
        assert not resp.is_error

    def test_config_property(self) -> None:
        cfg = MCPServerConfig(name="x")
        server = MCPToolServer(cfg)
        assert server.config is cfg

    def test_tool_names_empty(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        assert server.tool_names == []

    def test_tool_names_order(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        server.register_tool(MCPToolDefinition(name="a", description="A"), EchoHandler())
        server.register_tool(MCPToolDefinition(name="b", description="B"), EchoHandler())
        assert server.tool_names == ["a", "b"]

    def test_tools_call_no_arguments(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        server.register_tool(
            MCPToolDefinition(name="echo", description="Echo"), EchoHandler()
        )
        resp = server.handle_request(
            MCPRequest(method="tools/call", params={"name": "echo"}, id=1)
        )
        assert not resp.is_error
        assert resp.result["content"][0]["text"] == "echo: "


# ── MCPClient Tests ──────────────────────────────────────────────────


class TestMCPClient:
    def test_initialize(self) -> None:
        client, _ = _make_wired_pair()
        caps = client.initialize()
        assert isinstance(caps, MCPServerCapabilities)
        assert caps.tools

    def test_not_initialized_list_tools_raises(self) -> None:
        client, _ = _make_wired_pair()
        with pytest.raises(MCPProtocolError, match="not been initialized"):
            client.list_tools()

    def test_not_initialized_call_tool_raises(self) -> None:
        client, _ = _make_wired_pair()
        with pytest.raises(MCPProtocolError, match="not been initialized"):
            client.call_tool("echo")

    def test_list_tools(self) -> None:
        client, server = _make_wired_pair()
        defn = MCPToolDefinition(name="echo", description="Echo input")
        server.register_tool(defn, EchoHandler())
        client.initialize()
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_list_tools_multiple(self) -> None:
        client, server = _make_wired_pair()
        server.register_tool(
            MCPToolDefinition(name="echo", description="Echo"), EchoHandler()
        )
        server.register_tool(
            MCPToolDefinition(name="upper", description="Upper"), UpperHandler()
        )
        client.initialize()
        tools = client.list_tools()
        assert len(tools) == 2

    def test_tools_property_cached(self) -> None:
        client, server = _make_wired_pair()
        defn = MCPToolDefinition(name="echo", description="Echo")
        server.register_tool(defn, EchoHandler())
        client.initialize()
        client.list_tools()
        assert len(client.tools) == 1

    def test_tools_property_returns_copy(self) -> None:
        client, server = _make_wired_pair()
        server.register_tool(
            MCPToolDefinition(name="echo", description="Echo"), EchoHandler()
        )
        client.initialize()
        client.list_tools()
        tools = client.tools
        tools.clear()
        assert len(client.tools) == 1  # Original not affected

    def test_call_tool(self) -> None:
        client, server = _make_wired_pair()
        defn = MCPToolDefinition(name="echo", description="Echo")
        server.register_tool(defn, EchoHandler())
        client.initialize()
        result = client.call_tool("echo", {"text": "world"})
        assert not result.is_error
        assert result.content[0].text == "echo: world"

    def test_call_tool_unknown_raises(self) -> None:
        client, _ = _make_wired_pair()
        client.initialize()
        with pytest.raises(MCPToolExecutionError) as exc_info:
            client.call_tool("nonexistent")
        assert exc_info.value.tool_name == "nonexistent"

    def test_call_tool_handler_exception(self) -> None:
        client, server = _make_wired_pair()
        defn = MCPToolDefinition(name="fail", description="Fails")
        server.register_tool(defn, FailHandler())
        client.initialize()
        result = client.call_tool("fail")
        assert result.is_error
        assert "boom" in result.content[0].text

    def test_call_tool_error_content(self) -> None:
        client, server = _make_wired_pair()
        defn = MCPToolDefinition(name="err", description="Error content")
        server.register_tool(defn, ErrorContentHandler())
        client.initialize()
        result = client.call_tool("err")
        assert result.is_error

    def test_call_tool_no_arguments(self) -> None:
        client, server = _make_wired_pair()
        server.register_tool(
            MCPToolDefinition(name="echo", description="Echo"), EchoHandler()
        )
        client.initialize()
        result = client.call_tool("echo")
        assert result.content[0].text == "echo: "

    def test_ping_success(self) -> None:
        client, _ = _make_wired_pair()
        assert client.ping()

    def test_ping_failure(self) -> None:
        def broken_handler(req: MCPRequest) -> MCPResponse:
            raise MCPConnectionError("server down")

        client = MCPClient(
            transport=InMemoryTransportAdapter(handler=broken_handler),
        )
        assert not client.ping()

    def test_capabilities_before_init(self) -> None:
        client, _ = _make_wired_pair()
        assert client.capabilities is None

    def test_capabilities_after_init(self) -> None:
        client, _ = _make_wired_pair()
        client.initialize()
        assert client.capabilities is not None
        assert client.capabilities.tools

    def test_close(self) -> None:
        client, _ = _make_wired_pair()
        client.close()  # should not raise

    def test_settings_defaults(self) -> None:
        client, _ = _make_wired_pair()
        assert client.settings.name == "electripy-mcp-client"

    def test_custom_settings(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="test"))
        client = MCPClient(
            transport=InMemoryTransportAdapter(handler=server.handle_request),
            settings=MCPClientSettings(name="custom-client", version="3.0"),
        )
        assert client.settings.name == "custom-client"

    def test_protocol_error_on_init_failure(self) -> None:
        def error_handler(req: MCPRequest) -> MCPResponse:
            from electripy.ai.mcp.domain import MCPErrorData

            return MCPResponse(
                id=req.id,
                error=MCPErrorData(code=-1, message="init rejected"),
            )

        client = MCPClient(
            transport=InMemoryTransportAdapter(handler=error_handler),
        )
        with pytest.raises(MCPProtocolError, match="init rejected"):
            client.initialize()


# ── AsyncMCPClient Tests ────────────────────────────────────────────


class TestAsyncMCPClient:
    async def test_initialize(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        caps = await client.initialize()
        assert caps.tools

    async def test_list_tools(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        defn = MCPToolDefinition(name="echo", description="Echo")
        server.register_tool(defn, EchoHandler())
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        await client.initialize()
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

    async def test_call_tool(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        defn = MCPToolDefinition(name="echo", description="Echo")
        server.register_tool(defn, EchoHandler())
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        await client.initialize()
        result = await client.call_tool("echo", {"text": "async"})
        assert result.content[0].text == "echo: async"

    async def test_call_tool_error(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        await client.initialize()
        with pytest.raises(MCPToolExecutionError):
            await client.call_tool("missing")

    async def test_ping(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        assert await client.ping()

    async def test_not_initialized_raises(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        with pytest.raises(MCPProtocolError):
            await client.list_tools()

    async def test_capabilities(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        assert client.capabilities is None
        await client.initialize()
        assert client.capabilities is not None

    async def test_close(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        await client.close()  # should not raise

    async def test_tools_property(self) -> None:
        server = MCPToolServer(MCPServerConfig(name="async-test"))
        server.register_tool(
            MCPToolDefinition(name="echo", description="Echo"), EchoHandler()
        )
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))
        await client.initialize()
        await client.list_tools()
        assert len(client.tools) == 1


# ── End-to-end integration tests ────────────────────────────────────


class TestEndToEnd:
    def test_full_lifecycle(self) -> None:
        """Complete client-server lifecycle: init → list → call → close."""
        server = MCPToolServer(MCPServerConfig(name="e2e"))
        server.register_tool(
            MCPToolDefinition(
                name="greet",
                description="Greet someone",
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            ),
            _GreetHandler(),
        )
        client = MCPClient(
            transport=InMemoryTransportAdapter(handler=server.handle_request),
        )
        caps = client.initialize()
        assert caps.tools

        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "greet"

        result = client.call_tool("greet", {"name": "Alice"})
        assert not result.is_error
        assert result.content[0].text == "Hello, Alice!"

        client.close()

    def test_multiple_tools(self) -> None:
        """Server with multiple tools, client calls each."""
        server = MCPToolServer(MCPServerConfig(name="multi"))
        server.register_tool(
            MCPToolDefinition(name="echo", description="Echo"), EchoHandler()
        )
        server.register_tool(
            MCPToolDefinition(name="upper", description="Upper"), UpperHandler()
        )
        client = MCPClient(
            transport=InMemoryTransportAdapter(handler=server.handle_request),
        )
        client.initialize()

        echo_result = client.call_tool("echo", {"text": "hi"})
        assert echo_result.content[0].text == "echo: hi"

        upper_result = client.call_tool("upper", {"text": "hello"})
        assert upper_result.content[0].text == "HELLO"

    async def test_async_full_lifecycle(self) -> None:
        """Complete async client lifecycle."""
        server = MCPToolServer(MCPServerConfig(name="async-e2e"))
        server.register_tool(
            MCPToolDefinition(name="echo", description="Echo"), EchoHandler()
        )
        client = AsyncMCPClient(transport=_AsyncInMemoryTransport(server))

        caps = await client.initialize()
        assert caps.tools

        tools = await client.list_tools()
        assert len(tools) == 1

        result = await client.call_tool("echo", {"text": "async-e2e"})
        assert result.content[0].text == "echo: async-e2e"

        await client.close()


@dataclass(frozen=True, slots=True)
class _GreetHandler:
    def handle(self, call: MCPToolCall) -> MCPToolResult:
        name = str(call.arguments.get("name", "World"))
        return MCPToolResult(
            content=[MCPContent(type=MCPContentType.TEXT, text=f"Hello, {name}!")],
        )
