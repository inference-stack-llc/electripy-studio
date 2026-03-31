# AI Realtime Session Orchestration

The realtime package provides a **provider-neutral runtime substrate** for managing streaming AI sessions, events, tool calls, interruptions, and lifecycle transitions.

## Why it exists

Voice, multimodal, and streaming AI systems all share the same operational substrate: sessions that stream input and output, invoke tools, handle interruptions, and transition through well-defined lifecycle states. This package replaces fragile WebSocket/event-loop glue code with reusable, strongly typed orchestration primitives.

## Architecture

The package follows ElectriPy's Ports & Adapters pattern:

| Layer | Module | Responsibility |
|-------|--------|---------------|
| Domain | `domain.py` | Session, events, state machine, config |
| Ports | `ports.py` | Protocol interfaces for transport, storage, tools, observability |
| Adapters | `adapters.py` | In-memory transport, session store, observer, echo tool executor |
| Services | `services.py` | Session lifecycle orchestration and streaming helpers |

## Core concepts

- `RealtimeSession`: Mutable session with state, config, and event log.
- `SessionState`: Explicit lifecycle states (initialized → active → completed/failed → closed).
- `EventEnvelope`: Typed, sequenced wrapper around any event payload.
- `EventKind`: Discriminator for routing (input_text, output_text, tool_call, interrupt, etc.).
- `RealtimeSessionService`: Primary orchestration facade.

## Session lifecycle

```
initialized ──→ active ──→ completed ──→ closed
                  │  ↑           │
                  │  └─ resumed  │
                  ↓              │
             interrupted ──→ closed
                  │
                  ↓
              waiting_on_tool ──→ active
                  │
                  ↓
                failed ──→ closed
```

Any state (except `closed`) can transition to `failed` or `closed`.

## Quick example

```python
from electripy.ai.realtime import (
    RealtimeSessionService,
    RealtimeConfig,
    EventKind,
    InputStreamChunk,
    OutputStreamChunk,
    ChunkStatus,
)

svc = RealtimeSessionService()
session = svc.create_session(config=RealtimeConfig(model="gpt-4o"))
svc.start_session(session.session_id)

# Ingest user input
svc.ingest_event(
    session.session_id,
    EventKind.INPUT_TEXT,
    InputStreamChunk(index=0, text="Hello, what's the weather?"),
)

# Emit streamed output
for i, word in enumerate(["The", " weather", " is", " sunny."]):
    svc.emit_output(
        session.session_id,
        OutputStreamChunk(
            index=i,
            text=word,
            status=ChunkStatus.FINAL if i == 3 else ChunkStatus.PARTIAL,
        ),
    )

svc.complete_session(session.session_id)
svc.close_session(session.session_id)
```

## Streaming text collection

```python
from electripy.ai.realtime import OutputStreamChunk, collect_output_text, iter_output_text

chunks = [
    OutputStreamChunk(index=0, text="Hello"),
    OutputStreamChunk(index=1, text=" world"),
]

# Iterate deltas
for delta in iter_output_text(chunks):
    print(delta, end="")

# Or collect all at once
full_text = collect_output_text(chunks)
```

Async variants are also available:

```python
from electripy.ai.realtime import async_collect_output_text

text = await async_collect_output_text(async_chunk_stream)
```

## Interruption

```python
# Interrupt current generation
svc.interrupt(session.session_id, reason="user cancelled", hard=True)

# Resume when ready
svc.resume(session.session_id)
```

The session transitions to `interrupted` and records an `InterruptEvent`. Use `hard=True` to signal that buffered output should be discarded.

## Tool calls

```python
from electripy.ai.realtime import ToolCallEvent

call = ToolCallEvent(
    call_id="call-1",
    tool_name="web_search",
    arguments={"query": "weather today"},
)

# Async tool execution
result = await svc.handle_tool_call(session.session_id, call)
print(result.result)  # tool output
```

The session transitions to `waiting_on_tool` during execution, then back to `active` on success (or `failed` on error). Both the tool call and result are recorded in the event log.

### Custom tool executor

Implement `ToolExecutionPort`:

```python
from electripy.ai.realtime import ToolCallEvent, ToolResultEvent

class MyToolExecutor:
    async def execute(self, event: ToolCallEvent) -> ToolResultEvent:
        # Your tool logic here
        return ToolResultEvent(
            call_id=event.call_id,
            tool_name=event.tool_name,
            result={"answer": "sunny, 72°F"},
        )

svc = RealtimeSessionService(tool_executor=MyToolExecutor())
```

## Observability

Implement `RealtimeObserverPort` to hook into lifecycle events:

```python
from electripy.ai.realtime import InMemoryObserver, RealtimeSessionService

observer = InMemoryObserver()
svc = RealtimeSessionService(observer=observer)

session = svc.create_session()
svc.start_session(session.session_id)

# Inspect captured events
print(observer.state_changes)  # [(session_id, previous, current), ...]
print(observer.events)         # [EventEnvelope, ...]
```

For production, implement `RealtimeObserverPort` to forward events to your telemetry backend.

## Event replay

```python
from electripy.ai.realtime import EventKind

# Replay all events
all_events = svc.replay_events(session.session_id)

# Replay only output events
output_events = svc.replay_events(
    session.session_id,
    kinds=frozenset({EventKind.OUTPUT_TEXT, EventKind.OUTPUT_AUDIO}),
)
```

## Transport integration

```python
from electripy.ai.realtime import InMemoryTransport, RealtimeSessionService

transport = InMemoryTransport()
svc = RealtimeSessionService(transport=transport)

session = svc.create_session()
svc.start_session(session.session_id)

chunk = OutputStreamChunk(index=0, text="hi")
env = svc.emit_output(session.session_id, chunk)
await svc.send_to_transport(session.session_id, env)
```

For production WebSocket or message-queue transports, implement `RealtimeTransportPort`.

## Extension points

| Extension | How |
|-----------|-----|
| Custom transport (WebSocket, gRPC) | Implement `RealtimeTransportPort` |
| Persistent session store (Redis, DB) | Implement `SessionStorePort` |
| Custom tool execution | Implement `ToolExecutionPort` |
| Telemetry / tracing | Implement `RealtimeObserverPort` |
| Audio/video processing | Use `InputStreamChunk.audio_bytes` / `OutputStreamChunk.audio_bytes` |
| Backpressure control | Use `emit_backpressure()` with `BackpressureDirective` |
