"""LLM Replay Tape — record, replay, and diff LLM interactions.

Purpose:
  - Record every LLM request/response as an immutable tape entry.
  - Replay tapes as deterministic test fixtures (no network needed).
  - Diff outputs across model versions or prompt changes.

Guarantees:
  - Tape entries are fully serialisable to JSONL.
  - Replay port is a drop-in ``SyncLlmPort`` replacement.
  - Side-effect-free diffing with structured change reports.
  - All domain models are frozen and immutable.

Usage::

    from electripy.ai.replay_tape import RecordingLlmPort, ReplayLlmPort, TapeDiff

    # Record
    recorder = RecordingLlmPort(inner=my_llm_port)
    recorder.complete(request)
    tape = recorder.tape()  # list[TapeEntry]

    # Replay
    replay_port = ReplayLlmPort(tape=tape)
    response = replay_port.complete(request)  # deterministic, offline

    # Diff
    diff = TapeDiff.compare(tape_a=old_tape, tape_b=new_tape)
"""

from __future__ import annotations

from .domain import DiffResult, DiffStatus, TapeEntry
from .services import RecordingLlmPort, ReplayLlmPort, TapeDiff, TapeSerializer

__all__ = [
    # Domain models
    "TapeEntry",
    "DiffResult",
    "DiffStatus",
    # Services
    "RecordingLlmPort",
    "ReplayLlmPort",
    "TapeDiff",
    "TapeSerializer",
]
