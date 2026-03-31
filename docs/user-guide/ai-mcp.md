# AI / Model Context Protocol (MCP)

The `electripy.ai.mcp` package provides a **provider-neutral, strongly typed
Model Context Protocol toolkit** that enables teams to build MCP clients and
MCP server/tool adapters in a clean, production-friendly way.

## Architecture

The package follows ElectriPy's hexagonal architecture:

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Domain** | `domain.py` | MCP data models — tool definitions, requests, responses, capabilities |
| **Errors** | `errors.py` | Typed exception hierarchy rooted at `MCPError` |
| **Ports** | `ports.py` | Protocol interfaces for transports, auth, and tool handlers |
| **Adapters** | `adapters.py` | Concrete transport and auth implementations |
| **Services** | `services.py` | `MCPClient`, `AsyncMCPClient`, and `MCPToolServer` |

## Quick Start

### Client — connect to an MCP tool server

```python
from electripy.ai.mcp import (
    MCPClient,
    MCPToolServer,
    MCPServerConfig,
    MCPToolDefinition,
    InMemoryTransportAdapter,
)

# Wire up an in-process server for demonstration.
server = MCPToolServer(MCPServerConfig(name="demo"))
server.register_tool(
    MCPToolDefinition(name="echo", description="Echo input"),
    handler=echo_handler,
)

client = MCPClient(
    transport=InMemoryTransportAdapter(handler=server.handle_request),
)
caps = client.initialize()
tools = client.list_tools()
result = client.call_tool("echo", {"text": "hello"})
print(result.content[0].text)  # "echo: hello"
client.close()
```

### Server — expose tools via MCP

```python
from dataclasses import dataclass
from electripy.ai.mcp import (
    MCPToolServer,
    MCPServerConfig,
    MCPToolDefinition,
    MCPToolCall,
    MCPToolResult,
    MCPContent,
    MCPContentType,
)

@dataclass(frozen=True, slots=True)
class GreetHandler:
    def handle(self, call: MCPToolCall) -> MCPToolResult:
        name = str(call.arguments.get("name", "World"))
        return MCPToolResult(
            content=[MCPContent(type=MCPContentType.TEXT, text=f"Hello, {name}!")],
        )

server = MCPToolServer(MCPServerConfig(name="greet-server"))
server.register_tool(
    MCPToolDefinition(
        name="greet",
        description="Greet someone by name",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
    ),
    GreetHandler(),
)

# Handle an incoming request (e.g. from a transport layer).
response = server.handle_request(request)
```

## Transports

| Adapter | Use Case |
|---------|----------|
| `InMemoryTransportAdapter` | Unit tests — routes to a handler callable |
| `HttpTransportAdapter` | Remote MCP servers over HTTP POST |
| `AsyncHttpTransportAdapter` | Async HTTP transport via *httpx* |
| `StdioTransportAdapter` | MCP servers launched as subprocesses (stdin/stdout) |

### HTTP transport

```python
from electripy.ai.mcp import MCPClient, HttpTransportAdapter, BearerTokenAuthAdapter

auth = BearerTokenAuthAdapter(token="sk-...")
transport = HttpTransportAdapter(
    endpoint="https://mcp.example.com/rpc",
    auth_headers=auth.get_headers(),
    timeout_s=15.0,
)
client = MCPClient(transport=transport)
caps = client.initialize()
```

### Stdio transport (subprocess)

```python
from electripy.ai.mcp import MCPClient, StdioTransportAdapter

transport = StdioTransportAdapter(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)
client = MCPClient(transport=transport)
try:
    caps = client.initialize()
    tools = client.list_tools()
finally:
    client.close()
```

## Async Client

```python
from electripy.ai.mcp import AsyncMCPClient, AsyncHttpTransportAdapter

transport = AsyncHttpTransportAdapter(endpoint="https://mcp.example.com/rpc")
client = AsyncMCPClient(transport=transport)

caps = await client.initialize()
tools = await client.list_tools()
result = await client.call_tool("search", {"query": "hello"})
await client.close()
```

## Authentication

| Adapter | Description |
|---------|-------------|
| `NoOpAuthAdapter` | No credentials (default) |
| `BearerTokenAuthAdapter` | Injects `Authorization: Bearer <token>` |
| Custom | Implement `MCPAuthPort.get_headers()` |

## Error Handling

All errors extend `MCPError` (which extends `ElectriPyError`):

```python
from electripy.ai.mcp import MCPError, MCPToolExecutionError

try:
    result = client.call_tool("missing_tool")
except MCPToolExecutionError as exc:
    print(f"Tool {exc.tool_name} failed: {exc}")
except MCPError as exc:
    print(f"MCP error: {exc}")
```

| Exception | When |
|-----------|------|
| `MCPConnectionError` | Transport cannot reach the server |
| `MCPTimeoutError` | Request exceeds timeout |
| `MCPAuthenticationError` | Server returns 401 |
| `MCPProtocolError` | JSON-RPC / MCP protocol violation |
| `MCPToolExecutionError` | Server-side tool call failure |

## Observability Integration

The MCP package integrates with `electripy.observability.observe` via the
existing `MCPMetadata` and `SpanKind.MCP` types:

```python
from electripy.observability.observe import ObservabilityService, MCPMetadata

obs = ObservabilityService(tracer=my_tracer)
with obs.start_mcp_span("my-server", meta=MCPMetadata(
    server_name="my-server",
    tool_name="echo",
    protocol_version="2024-11-05",
)):
    result = client.call_tool("echo", {"text": "traced"})
```

## Domain Models

| Model | Purpose |
|-------|---------|
| `MCPToolDefinition` | Tool name, description, and JSON Schema |
| `MCPToolCall` | Tool invocation (name + arguments) |
| `MCPToolResult` | Tool output (content blocks + error flag) |
| `MCPContent` | Single content block (text, image, resource) |
| `MCPRequest` / `MCPResponse` | JSON-RPC 2.0 wire format |
| `MCPServerCapabilities` | Server capability advertisement |
| `MCPServerConfig` | Server identity and configuration |
| `MCPClientSettings` | Client identity and timeout settings |
