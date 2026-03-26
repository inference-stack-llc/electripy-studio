"""Cost Ledger — track LLM token spend per-call, per-label, per-tenant.

Purpose:
  - Accumulate token usage and estimated cost across LLM calls.
  - Slice spend by arbitrary labels (user, tenant, feature, model).
  - Query totals and breakdowns without external services.

Guarantees:
  - Thread-safe — all mutations guarded by a lock.
  - No provider-specific code — reads ``usage_total_tokens`` from responses.
  - Immutable snapshots via frozen dataclasses.

Usage::

    from electripy.ai.cost_ledger import CostLedger

    ledger = CostLedger(cost_per_1k_tokens=0.002)
    ledger.record(tokens=1500, labels={"tenant": "acme", "model": "gpt-4o-mini"})
    ledger.record(tokens=800, labels={"tenant": "acme", "model": "gpt-4o-mini"})

    print(ledger.total())           # LedgerTotal(tokens=2300, estimated_cost=0.0046)
    print(ledger.by_label("tenant"))  # {"acme": LedgerTotal(...)}
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

__all__ = [
    "CostLedger",
    "LedgerEntry",
    "LedgerTotal",
]


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One recorded usage event."""

    tokens: int
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LedgerTotal:
    """Aggregated totals."""

    tokens: int
    estimated_cost: float
    call_count: int


class CostLedger:
    """Thread-safe, in-process token cost accumulator.

    Args:
        cost_per_1k_tokens: Dollar cost per 1 000 tokens.  Defaults to
            0.0 (tracking only, no cost estimation).
    """

    __slots__ = ("_cost_per_1k", "_entries", "_lock")

    def __init__(self, *, cost_per_1k_tokens: float = 0.0) -> None:
        self._cost_per_1k = cost_per_1k_tokens
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()

    # -- Recording ----------------------------------------------------------

    def record(
        self,
        *,
        tokens: int,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a single usage event."""
        entry = LedgerEntry(tokens=tokens, labels=labels or {})
        with self._lock:
            self._entries.append(entry)

    # -- Querying -----------------------------------------------------------

    def total(self) -> LedgerTotal:
        """Return aggregate totals across all entries."""
        with self._lock:
            entries = list(self._entries)
        total_tokens = sum(e.tokens for e in entries)
        return LedgerTotal(
            tokens=total_tokens,
            estimated_cost=total_tokens / 1000.0 * self._cost_per_1k,
            call_count=len(entries),
        )

    def by_label(self, label_key: str) -> dict[str, LedgerTotal]:
        """Return totals grouped by values of *label_key*."""
        with self._lock:
            entries = list(self._entries)
        buckets: dict[str, list[LedgerEntry]] = {}
        for entry in entries:
            value = entry.labels.get(label_key)
            if value is not None:
                buckets.setdefault(value, []).append(entry)
        return {
            key: LedgerTotal(
                tokens=sum(e.tokens for e in group),
                estimated_cost=sum(e.tokens for e in group) / 1000.0 * self._cost_per_1k,
                call_count=len(group),
            )
            for key, group in buckets.items()
        }

    def entries(self) -> list[LedgerEntry]:
        """Return a snapshot of all recorded entries."""
        with self._lock:
            return list(self._entries)

    def reset(self) -> None:
        """Clear all recorded entries."""
        with self._lock:
            self._entries.clear()
