from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json

from electripy.observability.ai_telemetry import (
    InMemoryTelemetryAdapter,
    JsonlTelemetrySinkAdapter,
    Severity,
    TableCostEstimator,
    create_telemetry_context,
    current_telemetry_context,
    inject_context_headers,
    scoped_telemetry_context,
)
from electripy.observability.ai_telemetry.services import record_llm_call


def test_scoped_telemetry_context_sets_and_restores() -> None:
    ctx = create_telemetry_context(environment="test")

    assert current_telemetry_context() is None
    with scoped_telemetry_context(ctx):
        assert current_telemetry_context() == ctx
    assert current_telemetry_context() is None


def test_inject_context_headers_uses_current_context() -> None:
    ctx = create_telemetry_context(environment="test")
    headers: dict[str, str] = {}

    with scoped_telemetry_context(ctx):
        inject_context_headers(headers)

    assert "X-Request-Id" in headers
    assert headers["X-Request-Id"] == ctx.request_id


def test_inmemory_adapter_sanitises_sensitive_attributes() -> None:
    telemetry = InMemoryTelemetryAdapter()
    ctx = create_telemetry_context(environment="test")

    telemetry.increment("llm.tokens", attrs={"prompt": "secret prompt", "other": "ok"}, ctx=ctx)

    assert len(telemetry.counters) == 1
    attrs = telemetry.counters[0].attributes or {}
    # sensitive key is removed and replaced with a hash key
    assert "prompt" not in attrs
    assert any(k.endswith("_hash") for k in attrs)
    assert attrs["other"] == "ok"


def test_jsonl_adapter_writes_events(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    sink = JsonlTelemetrySinkAdapter(path=path)
    ctx = create_telemetry_context(environment="test")

    from electripy.observability.ai_telemetry import TelemetryEvent

    event = TelemetryEvent(
        name="example",
        timestamp=datetime.now(tz=timezone.utc),
        context=ctx,
        attributes={"provider": "fake"},
        severity=Severity.INFO,
    )
    sink.emit_event(event)

    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["type"] == "event"
    assert payload["name"] == "example"
    assert payload["context"]["trace_id"] == ctx.trace_id


def test_span_context_manager_records_start_and_finish(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    sink = JsonlTelemetrySinkAdapter(path=path)
    ctx = create_telemetry_context(environment="test")

    with scoped_telemetry_context(ctx):
        with sink.span("work", attrs={"k": "v"}):
            pass

    lines = path.read_text(encoding="utf-8").splitlines()
    names = [json.loads(line)["name"] for line in lines]
    assert "span_started" in names
    assert "span_finished" in names


def test_table_cost_estimator_missing_rate_returns_none() -> None:
    estimator = TableCostEstimator(rates={})
    cost = estimator.estimate_cost(
        provider="openai",
        model="gpt-4.1",
        input_tokens=1000,
        output_tokens=500,
    )
    assert cost.estimated_cost_usd is None


def test_table_cost_estimator_computes_cost() -> None:
    estimator = TableCostEstimator(rates={("openai", "gpt-4.1"): (0.01, 0.03)})
    cost = estimator.estimate_cost(
        provider="openai",
        model="gpt-4.1",
        input_tokens=1000,
        output_tokens=500,
    )
    # 1000/1000 * 0.01 + 500/1000 * 0.03 = 0.01 + 0.015 = 0.025
    assert cost.estimated_cost_usd == 0.025


def test_record_llm_call_emits_metrics_and_event() -> None:
    telemetry = InMemoryTelemetryAdapter()
    ctx = create_telemetry_context(environment="test")

    record_llm_call(
        telemetry,
        provider="openai",
        model="gpt-4.1",
        latency_ms=123.0,
        input_tokens=1000,
        output_tokens=250,
        finish_reason="stop",
        structured_output_valid=True,
        ctx=ctx,
    )

    assert telemetry.counters
    assert telemetry.histograms
    assert telemetry.events
    names = {e.name for e in telemetry.events}
    assert "llm.call" in names
