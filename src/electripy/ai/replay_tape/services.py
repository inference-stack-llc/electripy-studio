"""Services for the LLM Replay Tape."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from electripy.ai.llm_gateway.domain import LlmRequest, LlmResponse
from electripy.ai.llm_gateway.ports import SyncLlmPort

from .domain import DiffResult, DiffStatus, TapeEntry


class RecordingLlmPort:
    """Transparent recording wrapper around a ``SyncLlmPort``.

    Every call to ``complete()`` is forwarded to the inner port and the
    request/response pair is appended to an internal tape.

    Args:
      inner: The underlying LLM port to record from.
    """

    __slots__ = ("_inner", "_entries")

    def __init__(self, *, inner: SyncLlmPort) -> None:
        self._inner = inner
        self._entries: list[TapeEntry] = []

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        """Forward to the inner port and record the interaction."""
        response = self._inner.complete(request, timeout=timeout)
        entry = TapeEntry(
            index=len(self._entries),
            model=request.model,
            messages=tuple({"role": m.role.value, "content": m.content} for m in request.messages),
            temperature=request.temperature,
            response_text=response.text,
        )
        self._entries.append(entry)
        return response

    def tape(self) -> list[TapeEntry]:
        """Return a copy of the recorded tape entries."""
        return list(self._entries)

    def reset(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()


class ReplayLlmPort:
    """Deterministic replay port that serves pre-recorded tape entries.

    Entries are served in order.  If the tape is exhausted, a
    ``StopIteration`` is raised.

    Args:
      tape: The tape entries to replay.
    """

    __slots__ = ("_tape", "_index")

    def __init__(self, *, tape: Sequence[TapeEntry]) -> None:
        self._tape = list(tape)
        self._index = 0

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        """Return the next pre-recorded response.

        Raises:
          StopIteration: If all tape entries have been consumed.
        """
        if self._index >= len(self._tape):
            raise StopIteration("replay tape exhausted")
        entry = self._tape[self._index]
        self._index += 1
        return LlmResponse(text=entry.response_text, model=entry.model)

    def reset(self) -> None:
        """Rewind to the beginning of the tape."""
        self._index = 0


class TapeDiff:
    """Compare two tapes and produce structured diff results."""

    __slots__ = ()

    @staticmethod
    def compare(
        *,
        tape_a: Sequence[TapeEntry],
        tape_b: Sequence[TapeEntry],
    ) -> list[DiffResult]:
        """Compare two tapes entry-by-entry.

        Entries are compared by index.  If one tape is shorter,
        extra entries in the longer tape are marked as added/removed.

        Args:
          tape_a: The baseline tape.
          tape_b: The comparison tape.

        Returns:
          A list of :class:`DiffResult` objects, one per index.
        """
        results: list[DiffResult] = []
        max_len = max(len(tape_a), len(tape_b))

        for i in range(max_len):
            a = tape_a[i] if i < len(tape_a) else None
            b = tape_b[i] if i < len(tape_b) else None

            if a is not None and b is not None:
                if a.response_text == b.response_text:
                    status = DiffStatus.IDENTICAL
                else:
                    status = DiffStatus.CHANGED
                results.append(
                    DiffResult(
                        index=i,
                        status=status,
                        text_a=a.response_text,
                        text_b=b.response_text,
                    )
                )
            elif a is not None:
                results.append(
                    DiffResult(
                        index=i,
                        status=DiffStatus.REMOVED,
                        text_a=a.response_text,
                    )
                )
            else:
                assert b is not None
                results.append(
                    DiffResult(
                        index=i,
                        status=DiffStatus.ADDED,
                        text_b=b.response_text,
                    )
                )

        return results

    @staticmethod
    def summary(diffs: Sequence[DiffResult]) -> dict[str, int]:
        """Return a count of each diff status.

        Args:
          diffs: The diff results to summarise.

        Returns:
          A dict mapping status names to counts.
        """
        counts: dict[str, int] = {s.value: 0 for s in DiffStatus}
        for d in diffs:
            counts[d.status.value] += 1
        return counts


class TapeSerializer:
    """Serialise and deserialise tapes to/from JSONL."""

    __slots__ = ()

    @staticmethod
    def to_jsonl(tape: Sequence[TapeEntry]) -> str:
        """Serialise a tape to a JSONL string.

        Args:
          tape: Tape entries to serialise.

        Returns:
          A newline-delimited JSON string.
        """
        lines: list[str] = []
        for entry in tape:
            obj: dict[str, Any] = {
                "index": entry.index,
                "model": entry.model,
                "messages": list(entry.messages),
                "temperature": entry.temperature,
                "response_text": entry.response_text,
            }
            if entry.metadata:
                obj["metadata"] = entry.metadata
            lines.append(json.dumps(obj, ensure_ascii=False))
        return "\n".join(lines)

    @staticmethod
    def from_jsonl(text: str) -> list[TapeEntry]:
        """Deserialise a JSONL string into tape entries.

        Args:
          text: The JSONL string to parse.

        Returns:
          A list of :class:`TapeEntry` objects.

        Raises:
          json.JSONDecodeError: If a line is not valid JSON.
          KeyError: If required fields are missing.
        """
        entries: list[TapeEntry] = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            entries.append(
                TapeEntry(
                    index=obj["index"],
                    model=obj["model"],
                    messages=tuple(obj["messages"]),
                    temperature=obj["temperature"],
                    response_text=obj["response_text"],
                    metadata=obj.get("metadata", {}),
                )
            )
        return entries
