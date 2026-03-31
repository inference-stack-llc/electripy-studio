"""Adapters for MCP transports and authentication.

This module provides concrete implementations of the ports defined in
:mod:`electripy.ai.mcp.ports`:

- **InMemoryTransportAdapter** — routes requests to a callable handler
  (useful for testing with :class:`MCPToolServer`).
- **HttpTransportAdapter** — sends JSON-RPC over HTTP using *httpx*.
- **AsyncHttpTransportAdapter** — async version of the HTTP transport.
- **StdioTransportAdapter** — manages an MCP server subprocess and
  communicates via newline-delimited JSON on stdin/stdout.
- **NoOpAuthAdapter** / **BearerTokenAuthAdapter** — authentication.

Example:
    from electripy.ai.mcp.adapters import HttpTransportAdapter

    transport = HttpTransportAdapter(endpoint="http://localhost:8080/mcp")
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from electripy.core.logging import get_logger

from .domain import MCPRequest, MCPResponse
from .errors import MCPAuthenticationError, MCPConnectionError, MCPTimeoutError

__all__ = [
    "AsyncHttpTransportAdapter",
    "BearerTokenAuthAdapter",
    "HttpTransportAdapter",
    "InMemoryTransportAdapter",
    "NoOpAuthAdapter",
    "StdioTransportAdapter",
]

logger = get_logger(__name__)


# ── Auth adapters ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NoOpAuthAdapter:
    """Auth adapter that provides no credentials."""

    def get_headers(self) -> dict[str, str]:
        """Return an empty header mapping."""
        return {}


@dataclass(frozen=True, slots=True)
class BearerTokenAuthAdapter:
    """Auth adapter that injects a Bearer token.

    Attributes:
        token: The bearer token value.
    """

    token: str

    def get_headers(self) -> dict[str, str]:
        """Return an ``Authorization: Bearer`` header."""
        return {"Authorization": f"Bearer {self.token}"}


# ── Transport adapters ───────────────────────────────────────────────


@dataclass(slots=True)
class InMemoryTransportAdapter:
    """Transport that routes requests directly to a handler callable.

    This is the preferred adapter for unit tests.  Pass
    ``server.handle_request`` as the handler to wire up an
    :class:`MCPToolServer` without network I/O.

    Attributes:
        handler: Callable that accepts an :class:`MCPRequest` and
            returns an :class:`MCPResponse`.
    """

    handler: Callable[[MCPRequest], MCPResponse]

    def send(self, request: MCPRequest) -> MCPResponse:
        """Dispatch request to the in-memory handler."""
        return self.handler(request)

    def close(self) -> None:
        """No resources to release."""


@dataclass(slots=True)
class HttpTransportAdapter:
    """HTTP-based transport using *httpx* synchronous client.

    Sends each :class:`MCPRequest` as an HTTP POST with a JSON body and
    parses the JSON response into an :class:`MCPResponse`.

    Attributes:
        endpoint: Full URL of the MCP server HTTP endpoint.
        auth_headers: Optional static headers for authentication.
        timeout_s: Per-request timeout in seconds.
    """

    endpoint: str
    auth_headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 30.0
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def send(self, request: MCPRequest) -> MCPResponse:
        """Send request over HTTP POST."""
        client = self._get_client()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self.auth_headers)

        try:
            http_resp = client.post(
                self.endpoint,
                json=request.to_dict(),
                headers=headers,
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise MCPTimeoutError("MCP HTTP request timed out") from exc
        except httpx.ConnectError as exc:
            raise MCPConnectionError(f"MCP HTTP connection failed: {exc}") from exc

        if http_resp.status_code == 401:
            raise MCPAuthenticationError("MCP server returned 401 Unauthorized")

        if request.id is None:
            return MCPResponse()

        return MCPResponse.from_dict(http_resp.json())

    def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            self._client.close()
            self._client = None


@dataclass(slots=True)
class AsyncHttpTransportAdapter:
    """HTTP-based transport using *httpx* asynchronous client.

    Attributes:
        endpoint: Full URL of the MCP server HTTP endpoint.
        auth_headers: Optional static headers for authentication.
        timeout_s: Per-request timeout in seconds.
    """

    endpoint: str
    auth_headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 30.0
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def send(self, request: MCPRequest) -> MCPResponse:
        """Send request over HTTP POST asynchronously."""
        client = self._get_client()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self.auth_headers)

        try:
            http_resp = await client.post(
                self.endpoint,
                json=request.to_dict(),
                headers=headers,
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise MCPTimeoutError("MCP HTTP request timed out") from exc
        except httpx.ConnectError as exc:
            raise MCPConnectionError(f"MCP HTTP connection failed: {exc}") from exc

        if http_resp.status_code == 401:
            raise MCPAuthenticationError("MCP server returned 401 Unauthorized")

        if request.id is None:
            return MCPResponse()

        return MCPResponse.from_dict(http_resp.json())

    async def close(self) -> None:
        """Close the underlying httpx async client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@dataclass(slots=True)
class StdioTransportAdapter:
    """Transport that manages an MCP server as a subprocess.

    Messages are exchanged as newline-delimited JSON over the child
    process's stdin (outgoing) and stdout (incoming).

    Attributes:
        command: Executable to launch (e.g. ``"npx"``).
        args: Command-line arguments for the subprocess.
        env: Optional environment variable overrides.
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        if self._process is None or self._process.poll() is not None:
            logger.info(
                "Starting MCP server subprocess",
                extra={"command": self.command, "args": self.args},
            )
            self._process = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
            )
        return self._process

    def send(self, request: MCPRequest) -> MCPResponse:
        """Write request to subprocess stdin and read response from stdout."""
        process = self._ensure_process()
        assert process.stdin is not None  # noqa: S101
        assert process.stdout is not None  # noqa: S101

        payload = json.dumps(request.to_dict()) + "\n"
        try:
            process.stdin.write(payload.encode())
            process.stdin.flush()
        except OSError as exc:
            raise MCPConnectionError("Failed to write to MCP server process") from exc

        if request.id is None:
            return MCPResponse()

        line = process.stdout.readline()
        if not line:
            raise MCPConnectionError("MCP server process closed stdout unexpectedly")

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MCPConnectionError(f"Invalid JSON from MCP server: {line!r}") from exc

        return MCPResponse.from_dict(data)

    def close(self) -> None:
        """Terminate the managed subprocess."""
        if self._process is not None:
            logger.info("Terminating MCP server subprocess")
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
            finally:
                self._process = None
