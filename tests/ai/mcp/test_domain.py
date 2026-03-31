"""Tests for electripy.ai.mcp.domain."""

from __future__ import annotations

import pytest

from electripy.ai.mcp.domain import (
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


class TestMCPProtocolVersion:
    def test_values(self) -> None:
        assert MCPProtocolVersion.V_2024_11_05 == "2024-11-05"

    def test_is_str(self) -> None:
        assert isinstance(MCPProtocolVersion.V_2024_11_05, str)


class TestMCPContentType:
    def test_text(self) -> None:
        assert MCPContentType.TEXT == "text"

    def test_image(self) -> None:
        assert MCPContentType.IMAGE == "image"

    def test_resource(self) -> None:
        assert MCPContentType.RESOURCE == "resource"


class TestMCPContent:
    def test_text_content(self) -> None:
        c = MCPContent(type=MCPContentType.TEXT, text="hello")
        assert c.text == "hello"
        assert c.type == MCPContentType.TEXT

    def test_to_dict_text(self) -> None:
        c = MCPContent(type=MCPContentType.TEXT, text="hello")
        d = c.to_dict()
        assert d == {"type": "text", "text": "hello"}

    def test_to_dict_image(self) -> None:
        c = MCPContent(type=MCPContentType.IMAGE, data="base64data", mime_type="image/png")
        d = c.to_dict()
        assert d == {"type": "image", "data": "base64data", "mimeType": "image/png"}

    def test_to_dict_resource(self) -> None:
        c = MCPContent(type=MCPContentType.RESOURCE, uri="file:///tmp/x.txt")
        d = c.to_dict()
        assert d == {"type": "resource", "uri": "file:///tmp/x.txt"}

    def test_to_dict_omits_none(self) -> None:
        c = MCPContent(type=MCPContentType.TEXT, text="x")
        d = c.to_dict()
        assert "data" not in d
        assert "mimeType" not in d
        assert "uri" not in d

    def test_frozen(self) -> None:
        c = MCPContent(type=MCPContentType.TEXT, text="x")
        with pytest.raises(AttributeError):
            c.text = "y"  # type: ignore[misc]


class TestMCPToolDefinition:
    def test_basic(self) -> None:
        t = MCPToolDefinition(name="echo", description="Echo input")
        assert t.name == "echo"
        assert t.description == "Echo input"

    def test_to_dict(self) -> None:
        t = MCPToolDefinition(
            name="get_weather",
            description="Get weather",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        d = t.to_dict()
        assert d["name"] == "get_weather"
        assert d["description"] == "Get weather"
        assert "inputSchema" in d

    def test_to_dict_default_schema(self) -> None:
        t = MCPToolDefinition(name="echo", description="Echo")
        d = t.to_dict()
        assert d["inputSchema"] == {"type": "object", "properties": {}}

    def test_from_dict(self) -> None:
        data: dict[str, object] = {
            "name": "echo",
            "description": "Echo input",
            "inputSchema": {"type": "object"},
        }
        t = MCPToolDefinition.from_dict(data)
        assert t.name == "echo"
        assert t.description == "Echo input"
        assert t.input_schema == {"type": "object"}

    def test_from_dict_missing_description(self) -> None:
        data: dict[str, object] = {"name": "echo"}
        t = MCPToolDefinition.from_dict(data)
        assert t.description == ""

    def test_roundtrip(self) -> None:
        original = MCPToolDefinition(
            name="tool",
            description="desc",
            input_schema={"type": "object"},
        )
        restored = MCPToolDefinition.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.description == original.description

    def test_from_dict_non_dict_schema(self) -> None:
        data: dict[str, object] = {"name": "t", "inputSchema": "bad"}
        t = MCPToolDefinition.from_dict(data)
        assert t.input_schema == {}


class TestMCPToolCall:
    def test_basic(self) -> None:
        call = MCPToolCall(name="echo", arguments={"text": "hi"})
        assert call.name == "echo"
        assert call.arguments == {"text": "hi"}

    def test_defaults(self) -> None:
        call = MCPToolCall(name="ping")
        assert call.arguments == {}
        assert call.call_id is None


class TestMCPToolResult:
    def test_success(self) -> None:
        r = MCPToolResult(
            content=[MCPContent(type=MCPContentType.TEXT, text="ok")],
        )
        assert not r.is_error
        assert len(r.content) == 1

    def test_error(self) -> None:
        r = MCPToolResult(is_error=True)
        assert r.is_error

    def test_to_dict(self) -> None:
        r = MCPToolResult(
            content=[MCPContent(type=MCPContentType.TEXT, text="ok")],
            is_error=False,
        )
        d = r.to_dict()
        assert d["isError"] is False
        assert len(d["content"]) == 1
        assert d["content"][0]["type"] == "text"


class TestMCPServerCapabilities:
    def test_defaults(self) -> None:
        caps = MCPServerCapabilities()
        assert not caps.tools
        assert not caps.resources
        assert not caps.prompts

    def test_to_dict_empty(self) -> None:
        caps = MCPServerCapabilities()
        assert caps.to_dict() == {}

    def test_to_dict_with_tools(self) -> None:
        caps = MCPServerCapabilities(tools=True)
        assert caps.to_dict() == {"tools": {}}

    def test_to_dict_full(self) -> None:
        caps = MCPServerCapabilities(tools=True, resources=True, prompts=True)
        d = caps.to_dict()
        assert "tools" in d
        assert "resources" in d
        assert "prompts" in d

    def test_from_dict(self) -> None:
        caps = MCPServerCapabilities.from_dict({"tools": {}, "resources": {}})
        assert caps.tools
        assert caps.resources
        assert not caps.prompts

    def test_roundtrip(self) -> None:
        original = MCPServerCapabilities(tools=True, prompts=True)
        restored = MCPServerCapabilities.from_dict(original.to_dict())
        assert restored.tools == original.tools
        assert restored.prompts == original.prompts
        assert restored.resources == original.resources


class TestMCPErrorData:
    def test_basic(self) -> None:
        e = MCPErrorData(code=-32600, message="Invalid request")
        assert e.code == -32600
        assert e.message == "Invalid request"
        assert e.data is None

    def test_to_dict(self) -> None:
        e = MCPErrorData(code=-32601, message="Not found", data={"detail": "x"})
        d = e.to_dict()
        assert d["code"] == -32601
        assert d["message"] == "Not found"
        assert d["data"] == {"detail": "x"}

    def test_to_dict_no_data(self) -> None:
        e = MCPErrorData(code=1, message="err")
        d = e.to_dict()
        assert "data" not in d


class TestMCPRequest:
    def test_basic(self) -> None:
        r = MCPRequest(method="tools/list", id=1)
        assert r.method == "tools/list"
        assert r.jsonrpc == "2.0"

    def test_to_dict(self) -> None:
        r = MCPRequest(method="tools/call", params={"name": "echo"}, id=42)
        d = r.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["method"] == "tools/call"
        assert d["params"] == {"name": "echo"}
        assert d["id"] == 42

    def test_notification_no_id(self) -> None:
        r = MCPRequest(method="notifications/initialized")
        d = r.to_dict()
        assert "id" not in d

    def test_no_params(self) -> None:
        r = MCPRequest(method="ping", id=1)
        d = r.to_dict()
        assert "params" not in d


class TestMCPResponse:
    def test_success(self) -> None:
        r = MCPResponse(id=1, result={"tools": []})
        assert not r.is_error
        assert r.result == {"tools": []}

    def test_error(self) -> None:
        r = MCPResponse(id=1, error=MCPErrorData(code=-1, message="fail"))
        assert r.is_error

    def test_to_dict(self) -> None:
        r = MCPResponse(id=1, result={"ok": True})
        d = r.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["result"] == {"ok": True}

    def test_to_dict_error(self) -> None:
        r = MCPResponse(id=1, error=MCPErrorData(code=-1, message="err"))
        d = r.to_dict()
        assert "error" in d
        assert d["error"]["code"] == -1

    def test_from_dict_success(self) -> None:
        data: dict[str, object] = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
        r = MCPResponse.from_dict(data)
        assert r.id == 1
        assert r.result == {"tools": []}
        assert not r.is_error

    def test_from_dict_error(self) -> None:
        data: dict[str, object] = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Not found"},
        }
        r = MCPResponse.from_dict(data)
        assert r.is_error
        assert r.error is not None
        assert r.error.code == -32601

    def test_from_dict_string_id(self) -> None:
        data: dict[str, object] = {"jsonrpc": "2.0", "id": "abc", "result": {}}
        r = MCPResponse.from_dict(data)
        assert r.id == "abc"

    def test_from_dict_no_id(self) -> None:
        data: dict[str, object] = {"jsonrpc": "2.0", "result": {}}
        r = MCPResponse.from_dict(data)
        assert r.id is None

    def test_roundtrip_success(self) -> None:
        original = MCPResponse(id=5, result={"key": "value"})
        restored = MCPResponse.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.result == original.result
        assert not restored.is_error

    def test_roundtrip_error(self) -> None:
        original = MCPResponse(id=5, error=MCPErrorData(code=-1, message="err"))
        restored = MCPResponse.from_dict(original.to_dict())
        assert restored.is_error
        assert restored.error is not None
        assert restored.error.code == -1


class TestMCPServerConfig:
    def test_defaults(self) -> None:
        cfg = MCPServerConfig(name="test")
        assert cfg.name == "test"
        assert cfg.version == "1.0.0"
        assert cfg.capabilities.tools
        assert cfg.protocol_version == MCPProtocolVersion.V_2024_11_05

    def test_custom(self) -> None:
        cfg = MCPServerConfig(
            name="custom",
            version="2.0",
            capabilities=MCPServerCapabilities(tools=True, resources=True),
        )
        assert cfg.name == "custom"
        assert cfg.version == "2.0"
        assert cfg.capabilities.resources


class TestMCPClientSettings:
    def test_defaults(self) -> None:
        s = MCPClientSettings()
        assert s.name == "electripy-mcp-client"
        assert s.version == "1.0.0"
        assert s.timeout_s == 30.0

    def test_custom(self) -> None:
        s = MCPClientSettings(name="my-client", timeout_s=10.0)
        assert s.name == "my-client"
        assert s.timeout_s == 10.0
