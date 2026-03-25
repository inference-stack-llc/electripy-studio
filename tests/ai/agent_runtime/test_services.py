from __future__ import annotations

from electripy.ai.agent_runtime import AgentExecutor, ToolInvocation


class _FakeToolPort:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, name: str, args: dict[str, object]) -> str:
        self.calls.append(name)
        if name == "fail":
            raise RuntimeError("boom")
        return f"ok:{name}:{args.get('id', '')}"


def test_agent_executor_stops_on_failure() -> None:
    port = _FakeToolPort()
    executor = AgentExecutor(tool_port=port, max_attempts=2)

    result = executor.run(
        [
            ToolInvocation(name="first", args={"id": 1}),
            ToolInvocation(name="fail", args={}),
            ToolInvocation(name="after", args={}),
        ]
    )

    assert result.all_successful is False
    assert len(result.steps) == 2
    assert result.steps[0].success is True
    assert result.steps[1].success is False
    assert port.calls == ["first", "fail", "fail"]
