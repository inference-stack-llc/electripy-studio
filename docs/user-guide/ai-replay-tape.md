# LLM Replay Tape

The **LLM Replay Tape** captures every LLM request/response pair as an
immutable tape.  Replay tapes in tests for deterministic, offline
execution, and diff tapes across model versions or prompt changes to
detect regressions.

## When to use it

- You need **deterministic LLM tests** that run without network access.
- You want to detect output drift after changing a prompt or upgrading a
  model.
- You need a reproducible audit trail of LLM interactions.

## Core concepts

- **Domain models**:
    - `TapeEntry` — frozen record of one request/response pair with a
      sequence index.
    - `DiffResult` — the result of comparing two tape entries.
    - `DiffStatus` — enum: `identical`, `changed`, `added`, `removed`.
- **Services**:
    - `RecordingLlmPort` — transparent wrapper: forwards calls to an
      inner `SyncLlmPort` and records every exchange.
    - `ReplayLlmPort` — plays back pre-recorded tape entries in order.
      No network, fully deterministic.
    - `TapeDiff` — compares two tapes entry-by-entry and produces
      structured diff reports.
    - `TapeSerializer` — reads/writes tapes as JSONL for persistence.

## Record and replay

```python
from electripy.ai.replay_tape import RecordingLlmPort, ReplayLlmPort

# --- Record phase (e.g. in a staging environment) ---
recorder = RecordingLlmPort(inner=real_llm_port)
response = recorder.complete(request)
tape = recorder.tape()  # list[TapeEntry]

# --- Replay phase (e.g. in CI) ---
replay = ReplayLlmPort(tape=tape)
response = replay.complete(request)  # returns recorded response, no network
```

## Diff across versions

After upgrading a model or editing a prompt, diff the old and new tapes
to see exactly what changed:

```python
from electripy.ai.replay_tape import TapeDiff

diff = TapeDiff.compare(tape_a=old_tape, tape_b=new_tape)

for result in diff:
    print(result.status, result.index)  # e.g. "changed 3"

print(TapeDiff.summary(diff))
# "identical: 8, changed: 2, added: 0, removed: 0"
```

## Persist tapes as JSONL

Serialize tapes to disk for version control or artefact storage:

```python
from electripy.ai.replay_tape import TapeSerializer

# Write
TapeSerializer.write(tape, "fixtures/baseline.jsonl")

# Read
tape = TapeSerializer.read("fixtures/baseline.jsonl")
```

Each line in the JSONL file is one `TapeEntry`, making tapes
git-diff-friendly.

## Integration with other components

- **LLM Caching** — use recording in staging, caching in production.
- **Eval Assertions** — replay a tape and assert each response with
  `assert_llm_output()` for regression testing.
- **Structured Output** — record structured extraction calls and replay
  them to test parsing logic without an LLM.
