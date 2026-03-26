"""Tests for CostLedger."""

from __future__ import annotations

import threading

import pytest

from electripy.ai.cost_ledger import CostLedger, LedgerEntry, LedgerTotal


class TestCostLedger:
    def test_empty_ledger(self) -> None:
        ledger = CostLedger()
        t = ledger.total()
        assert t.tokens == 0
        assert t.estimated_cost == 0.0
        assert t.call_count == 0

    def test_record_and_total(self) -> None:
        ledger = CostLedger(cost_per_1k_tokens=0.002)
        ledger.record(tokens=1000)
        ledger.record(tokens=500)

        t = ledger.total()
        assert t.tokens == 1500
        assert t.estimated_cost == pytest.approx(0.003)
        assert t.call_count == 2

    def test_by_label(self) -> None:
        ledger = CostLedger(cost_per_1k_tokens=0.01)
        ledger.record(tokens=100, labels={"tenant": "acme"})
        ledger.record(tokens=200, labels={"tenant": "acme"})
        ledger.record(tokens=300, labels={"tenant": "globex"})

        by_tenant = ledger.by_label("tenant")

        assert "acme" in by_tenant
        assert by_tenant["acme"].tokens == 300
        assert by_tenant["acme"].call_count == 2
        assert "globex" in by_tenant
        assert by_tenant["globex"].tokens == 300

    def test_by_label_missing_key(self) -> None:
        ledger = CostLedger()
        ledger.record(tokens=100, labels={"model": "gpt-4"})
        ledger.record(tokens=200)  # no labels

        by_tenant = ledger.by_label("tenant")
        assert by_tenant == {}

    def test_multiple_label_dimensions(self) -> None:
        ledger = CostLedger()
        ledger.record(tokens=100, labels={"tenant": "acme", "model": "gpt-4"})
        ledger.record(tokens=200, labels={"tenant": "acme", "model": "claude"})

        by_model = ledger.by_label("model")
        assert by_model["gpt-4"].tokens == 100
        assert by_model["claude"].tokens == 200

    def test_entries_snapshot(self) -> None:
        ledger = CostLedger()
        ledger.record(tokens=100)
        ledger.record(tokens=200)

        entries = ledger.entries()
        assert len(entries) == 2
        assert entries[0].tokens == 100
        assert entries[1].tokens == 200

    def test_reset(self) -> None:
        ledger = CostLedger()
        ledger.record(tokens=100)
        ledger.reset()

        assert ledger.total().tokens == 0
        assert ledger.entries() == []

    def test_cost_estimation(self) -> None:
        ledger = CostLedger(cost_per_1k_tokens=0.06)
        ledger.record(tokens=5000, labels={"model": "gpt-4"})

        t = ledger.total()
        assert t.estimated_cost == pytest.approx(0.30)

    def test_zero_cost_rate(self) -> None:
        ledger = CostLedger()  # default 0.0
        ledger.record(tokens=10_000)

        assert ledger.total().estimated_cost == 0.0

    def test_thread_safety(self) -> None:
        ledger = CostLedger()
        barrier = threading.Barrier(10)

        def _writer() -> None:
            barrier.wait()
            for _ in range(100):
                ledger.record(tokens=1)

        threads = [threading.Thread(target=_writer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ledger.total().tokens == 1000
        assert ledger.total().call_count == 1000

    def test_ledger_entry_immutable(self) -> None:
        entry = LedgerEntry(tokens=100, labels={"k": "v"})
        assert entry.tokens == 100

    def test_ledger_total_immutable(self) -> None:
        total = LedgerTotal(tokens=100, estimated_cost=0.01, call_count=1)
        assert total.tokens == 100
