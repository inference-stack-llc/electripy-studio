"""Domain models for streaming chat output."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StreamChunk:
    """Single chunk emitted by a chat stream.

    Attributes:
      index: Position in the stream sequence.
      delta_text: Text appended by this chunk.
      done: True if this chunk marks stream completion.
      metadata: Optional non-sensitive metadata.
    """

    index: int
    delta_text: str
    done: bool = False
    metadata: dict[str, str] = field(default_factory=dict)
