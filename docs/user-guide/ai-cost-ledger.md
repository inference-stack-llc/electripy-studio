# Cost Ledger

The **Cost Ledger** tracks LLM token usage and estimated cost in-process
with thread-safe accumulation and label-based slicing.

## When to use it

- You want per-tenant, per-model, or per-feature cost visibility
  without shipping data to a third-party service.
- You need a running total during a batch pipeline or an agent loop.
- You want to set spend alerts or budget guards in calling code.

## Core concepts

| Symbol | Role |
|--------|------|
| `CostLedger` | Thread-safe accumulator with `record()`, `total()`, `by_label()`. |
| `LedgerEntry` | Frozen record: `tokens` + `labels`. |
| `LedgerTotal` | Frozen aggregate: `tokens`, `estimated_cost`, `call_count`. |

## Basic example

```python
from electripy.ai.cost_ledger import CostLedger

ledger = CostLedger(cost_per_1k_tokens=0.002)

# After each LLM call:
ledger.record(tokens=1_500, labels={"tenant": "acme", "model": "gpt-4o-mini"})
ledger.record(tokens=800,   labels={"tenant": "acme", "model": "gpt-4o-mini"})
ledger.record(tokens=3_200, labels={"tenant": "globex", "model": "gpt-4o"})

# Global totals
print(ledger.total())
# LedgerTotal(tokens=5500, estimated_cost=0.011, call_count=3)

# Slice by any label dimension
by_tenant = ledger.by_label("tenant")
print(by_tenant["acme"])
# LedgerTotal(tokens=2300, estimated_cost=0.0046, call_count=2)
```

## Multi-dimensional labels

Labels are arbitrary string key-value pairs.  Slice by any dimension:

```python
ledger.record(tokens=500, labels={"model": "gpt-4o", "feature": "chat", "env": "prod"})

by_model   = ledger.by_label("model")
by_feature = ledger.by_label("feature")
by_env     = ledger.by_label("env")
```

## Thread-safety

All mutations are guarded by an internal lock.  Multiple threads can
call `record()` concurrently — `total()` and `by_label()` always return
consistent snapshot aggregates.

## Resetting

Call `ledger.reset()` to clear all entries (for example, between test
runs or pipeline stages).
