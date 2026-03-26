# Eval Assertions

**Eval Assertions** gives you pytest-native helpers for validating LLM
outputs in CI.  Declare checks as composable functions, run them in a
single `assert_llm_output()` call, and get structured diagnostics on
failure — no network calls, fully deterministic.

## When to use it

- You write **regression tests** for LLM-powered features and need clear
  pass/fail signals in CI.
- You want reusable, composable assertion checks instead of ad-hoc
  string parsing.
- You need diagnostic reports that explain *why* an output failed, not
  just that it did.

## Core concepts

- **Domain models**:
    - `AssertionCheck` — a frozen, named check wrapping a callable
      `(str) -> bool`.
    - `AssertionResult` — per-check pass/fail result with a diagnostic
      message.
    - `AssertionSeverity` — `error` (fails the test) or `warning`
      (logged but does not fail).
- **Services (assertion helpers)**:
    - `assert_llm_output(output, checks)` — runs all checks and raises
      `AssertionError` with a full report if any error-severity check
      fails.
    - `contains_keywords(keywords)` — check that all keywords appear.
    - `matches_regex(pattern)` — check against a regex.
    - `matches_json_schema(schema)` — lightweight field-presence and
      type check.
    - `passes_predicate(name, fn)` — wrap any `(str) -> bool` callable.
    - `satisfies_length(min_len, max_len)` — bounds check on output
      length.

## Basic example

```python
from electripy.ai.eval_assertions import (
    assert_llm_output,
    contains_keywords,
    matches_regex,
    satisfies_length,
)

output = "The capital of France is Paris."

assert_llm_output(
    output=output,
    checks=[
        contains_keywords(["Paris", "capital"]),
        matches_regex(r"France.*Paris"),
        satisfies_length(min_len=10, max_len=200),
    ],
)
# Passes silently — all checks succeed.
```

## JSON schema validation

For structured LLM outputs, check field presence and types:

```python
from electripy.ai.eval_assertions import assert_llm_output, matches_json_schema

import json

output = json.dumps({"title": "Q3 Review", "decisions": ["Ship v2", "Hire 3 devs"]})

assert_llm_output(
    output=output,
    checks=[
        matches_json_schema({"title": str, "decisions": list}),
    ],
)
```

## Custom predicates

Wrap any logic as a named check:

```python
from electripy.ai.eval_assertions import assert_llm_output, passes_predicate

assert_llm_output(
    output="positive",
    checks=[
        passes_predicate(
            "is valid sentiment",
            lambda text: text.strip() in ("positive", "negative", "neutral"),
        ),
    ],
)
```

## Warning-severity checks

Checks with `severity="warning"` are reported but don't fail the test:

```python
from electripy.ai.eval_assertions import (
    AssertionSeverity,
    assert_llm_output,
    satisfies_length,
)

short_check = satisfies_length(min_len=100, max_len=500)
# Override severity to warning
short_check = AssertionCheck(
    name=short_check.name,
    check_fn=short_check.check_fn,
    severity=AssertionSeverity.WARNING,
)

assert_llm_output(output="Brief.", checks=[short_check])
# Does not raise — the length failure is a warning, not an error.
```

## Integration with other components

- **Replay Tape** — replay recorded LLM responses and assert each one
  for regression testing without network calls.
- **Structured Output** — validate `.parsed` objects from the extraction
  engine with `passes_predicate()`.
- **LLM Gateway** — use eval assertions in integration tests for gateway
  responses.
