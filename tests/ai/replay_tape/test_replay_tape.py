"""Tests for the LLM Replay Tape."""

from __future__ import annotations

import json

import pytest

from electripy.ai.llm_gateway.domain import LlmMessage, LlmRequest, LlmResponse, LlmRole
from electripy.ai.replay_tape import (
    DiffResult,
    DiffStatus,
    RecordingLlmPort,
    ReplayLlmPort,
    TapeDiff,
    TapeEntry,
    TapeSerializer,
)

# ---------------------------------------------------------------------------
# Fake LLM port
# ---------------------------------------------------------------------------


class _FakeLlmPort:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        text = self._responses[self._index]
        self._index += 1
        return LlmResponse(text=text, model=request.model)


def _req(content: str = "hello") -> LlmRequest:
    return LlmRequest(
        model="gpt-4o-mini",
        messages=[LlmMessage(role=LlmRole.USER, content=content)],
    )


# ---------------------------------------------------------------------------
# RecordingLlmPort tests
# ---------------------------------------------------------------------------


class TestRecordingLlmPort:
    def test_records_interactions(self) -> None:
        inner = _FakeLlmPort(["resp1", "resp2"])
        recorder = RecordingLlmPort(inner=inner)

        recorder.complete(_req("a"))
        recorder.complete(_req("b"))

        tape = recorder.tape()
        assert len(tape) == 2
        assert tape[0].index == 0
        assert tape[0].response_text == "resp1"
        assert tape[1].index == 1
        assert tape[1].response_text == "resp2"

    def test_forwards_response(self) -> None:
        inner = _FakeLlmPort(["hello"])
        recorder = RecordingLlmPort(inner=inner)

        resp = recorder.complete(_req())
        assert resp.text == "hello"

    def test_records_messages(self) -> None:
        inner = _FakeLlmPort(["resp"])
        recorder = RecordingLlmPort(inner=inner)

        recorder.complete(_req("test prompt"))
        tape = recorder.tape()

        assert tape[0].messages == ({"role": "user", "content": "test prompt"},)

    def test_reset_clears_tape(self) -> None:
        inner = _FakeLlmPort(["resp"])
        recorder = RecordingLlmPort(inner=inner)

        recorder.complete(_req())
        recorder.reset()

        assert recorder.tape() == []

    def test_tape_is_a_copy(self) -> None:
        inner = _FakeLlmPort(["resp"])
        recorder = RecordingLlmPort(inner=inner)

        recorder.complete(_req())
        tape = recorder.tape()
        tape.clear()

        assert len(recorder.tape()) == 1


# ---------------------------------------------------------------------------
# ReplayLlmPort tests
# ---------------------------------------------------------------------------


class TestReplayLlmPort:
    def test_replay_in_order(self) -> None:
        entries = [
            TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="a"),
            TapeEntry(index=1, model="m", messages=(), temperature=0.2, response_text="b"),
        ]
        replay = ReplayLlmPort(tape=entries)

        r1 = replay.complete(_req())
        r2 = replay.complete(_req())

        assert r1.text == "a"
        assert r2.text == "b"

    def test_exhausted_raises_stop_iteration(self) -> None:
        replay = ReplayLlmPort(
            tape=[
                TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="only"),
            ]
        )
        replay.complete(_req())

        with pytest.raises(StopIteration, match="exhausted"):
            replay.complete(_req())

    def test_reset_rewinds(self) -> None:
        entry = TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="x")
        replay = ReplayLlmPort(tape=[entry])

        replay.complete(_req())
        replay.reset()
        resp = replay.complete(_req())

        assert resp.text == "x"


# ---------------------------------------------------------------------------
# TapeDiff tests
# ---------------------------------------------------------------------------


class TestTapeDiff:
    def test_identical_tapes(self) -> None:
        tape = [
            TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="same"),
        ]
        diffs = TapeDiff.compare(tape_a=tape, tape_b=tape)
        assert len(diffs) == 1
        assert diffs[0].status == DiffStatus.IDENTICAL

    def test_changed_entries(self) -> None:
        a = [TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="old")]
        b = [TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="new")]
        diffs = TapeDiff.compare(tape_a=a, tape_b=b)
        assert diffs[0].status == DiffStatus.CHANGED
        assert diffs[0].text_a == "old"
        assert diffs[0].text_b == "new"

    def test_added_entries(self) -> None:
        a: list[TapeEntry] = []
        b = [TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="new")]
        diffs = TapeDiff.compare(tape_a=a, tape_b=b)
        assert diffs[0].status == DiffStatus.ADDED

    def test_removed_entries(self) -> None:
        a = [TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="old")]
        b: list[TapeEntry] = []
        diffs = TapeDiff.compare(tape_a=a, tape_b=b)
        assert diffs[0].status == DiffStatus.REMOVED

    def test_mixed_diff(self) -> None:
        a = [
            TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="same"),
            TapeEntry(index=1, model="m", messages=(), temperature=0.2, response_text="old"),
        ]
        b = [
            TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="same"),
            TapeEntry(index=1, model="m", messages=(), temperature=0.2, response_text="new"),
            TapeEntry(index=2, model="m", messages=(), temperature=0.2, response_text="extra"),
        ]
        diffs = TapeDiff.compare(tape_a=a, tape_b=b)
        assert diffs[0].status == DiffStatus.IDENTICAL
        assert diffs[1].status == DiffStatus.CHANGED
        assert diffs[2].status == DiffStatus.ADDED

    def test_summary(self) -> None:
        diffs = [
            DiffResult(index=0, status=DiffStatus.IDENTICAL),
            DiffResult(index=1, status=DiffStatus.CHANGED, text_a="a", text_b="b"),
            DiffResult(index=2, status=DiffStatus.ADDED, text_b="c"),
        ]
        summary = TapeDiff.summary(diffs)
        assert summary == {"identical": 1, "changed": 1, "added": 1, "removed": 0}


# ---------------------------------------------------------------------------
# TapeSerializer tests
# ---------------------------------------------------------------------------


class TestTapeSerializer:
    def test_round_trip(self) -> None:
        tape = [
            TapeEntry(
                index=0,
                model="gpt-4o",
                messages=({"role": "user", "content": "hi"},),
                temperature=0.5,
                response_text="hello",
            ),
            TapeEntry(
                index=1,
                model="gpt-4o",
                messages=({"role": "user", "content": "bye"},),
                temperature=0.2,
                response_text="goodbye",
            ),
        ]
        jsonl = TapeSerializer.to_jsonl(tape)
        restored = TapeSerializer.from_jsonl(jsonl)

        assert len(restored) == 2
        assert restored[0].response_text == "hello"
        assert restored[1].response_text == "goodbye"
        assert restored[0].messages == ({"role": "user", "content": "hi"},)

    def test_to_jsonl_format(self) -> None:
        tape = [
            TapeEntry(index=0, model="m", messages=(), temperature=0.2, response_text="r"),
        ]
        jsonl = TapeSerializer.to_jsonl(tape)
        lines = jsonl.strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["index"] == 0
        assert obj["response_text"] == "r"

    def test_metadata_preserved(self) -> None:
        tape = [
            TapeEntry(
                index=0,
                model="m",
                messages=(),
                temperature=0.2,
                response_text="r",
                metadata={"tag": "test"},
            ),
        ]
        jsonl = TapeSerializer.to_jsonl(tape)
        restored = TapeSerializer.from_jsonl(jsonl)
        assert restored[0].metadata == {"tag": "test"}

    def test_empty_tape(self) -> None:
        assert TapeSerializer.to_jsonl([]) == ""
        assert TapeSerializer.from_jsonl("") == []
