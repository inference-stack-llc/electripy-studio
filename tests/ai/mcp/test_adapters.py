"""Tests for electripy.ai.mcp.adapters."""

from __future__ import annotations

import pytest

from electripy.ai.mcp.adapters import (
    BearerTokenAuthAdapter,
    InMemoryTransportAdapter,
    NoOpAuthAdapter,
)
from electripy.ai.mcp.domain import MCPErrorData, MCPRequest, MCPResponse
from electripy.ai.mcp.ports import MCPAuthPort, MCPTransportPort


class TestNoOpAuthAdapter:
    def test_get_headers_empty(self) -> None:
        adapter = NoOpAuthAdapter()
        assert adapter.get_headers() == {}

    def test_satisfies_port(self) -> None:
        assert isinstance(NoOpAuthAdapter(), MCPAuthPort)

    def test_frozen(self) -> None:
        adapter = NoOpAuthAdapter()
        with pytest.raises((AttributeError, TypeError)):
            adapter.x = 1  # type: ignore[attr-defined]


class TestBearerTokenAuthAdapter:
    def test_get_headers(self) -> None:
        adapter = BearerTokenAuthAdapter(token="secret123")
        headers = adapter.get_headers()
        assert headers == {"Authorization": "Bearer secret123"}

    def test_satisfies_port(self) -> None:
        assert isinstance(BearerTokenAuthAdapter(token="x"), MCPAuthPort)

    def test_frozen(self) -> None:
        adapter = BearerTokenAuthAdapter(token="x")
        with pytest.raises(AttributeError):
            adapter.token = "y"  # type: ignore[misc]

    def test_different_tokens(self) -> None:
        a = BearerTokenAuthAdapter(token="abc")
        b = BearerTokenAuthAdapter(token="xyz")
        assert a.get_headers() != b.get_headers()


class TestInMemoryTransportAdapter:
    def test_send_delegates_to_handler(self) -> None:
        expected = MCPResponse(id=1, result={"ok": True})
        adapter = InMemoryTransportAdapter(handler=lambda req: expected)
        result = adapter.send(MCPRequest(method="ping", id=1))
        assert result is expected

    def test_close_is_noop(self) -> None:
        adapter = InMemoryTransportAdapter(handler=lambda r: MCPResponse())
        adapter.close()  # should not raise

    def test_satisfies_transport_port(self) -> None:
        adapter = InMemoryTransportAdapter(handler=lambda r: MCPResponse())
        assert isinstance(adapter, MCPTransportPort)

    def test_error_response(self) -> None:
        err_resp = MCPResponse(id=1, error=MCPErrorData(code=-1, message="fail"))
        adapter = InMemoryTransportAdapter(handler=lambda r: err_resp)
        resp = adapter.send(MCPRequest(method="test", id=1))
        assert resp.is_error

    def test_handler_receives_request(self) -> None:
        received: list[MCPRequest] = []

        def capture(req: MCPRequest) -> MCPResponse:
            received.append(req)
            return MCPResponse(id=req.id)

        adapter = InMemoryTransportAdapter(handler=capture)
        adapter.send(MCPRequest(method="tools/list", id=7))
        assert len(received) == 1
        assert received[0].method == "tools/list"
        assert received[0].id == 7

    def test_multiple_sends(self) -> None:
        counter = {"n": 0}

        def counting_handler(req: MCPRequest) -> MCPResponse:
            counter["n"] += 1
            return MCPResponse(id=req.id, result={"call": counter["n"]})

        adapter = InMemoryTransportAdapter(handler=counting_handler)
        r1 = adapter.send(MCPRequest(method="a", id=1))
        r2 = adapter.send(MCPRequest(method="b", id=2))
        assert r1.result == {"call": 1}
        assert r2.result == {"call": 2}
