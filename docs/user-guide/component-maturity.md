# Component Maturity Model

Use this model to decide when an ElectriPy component is ready for broader reuse.

## Levels

### L0 - Experimental

- API can change without notice.
- No stability guarantees.
- Basic tests may be incomplete.

### L1 - Beta

- Public API is typed and documented.
- Unit tests cover happy paths and key failures.
- No known correctness blockers for local usage.

### L2 - Production-Ready

- Deterministic behavior is verified by tests.
- Error taxonomy is explicit and documented.
- Telemetry and safe logging posture are in place.
- Migration notes exist for breaking changes.

### L3 - Hardened

- Proven in multiple real workflows.
- Performance and failure mode budgets defined.
- Backward-compatibility policy actively enforced.

## Promotion Checklist

A component must satisfy all of the following before promotion:

1. Typed public API with `__all__` exports.
2. Tests for core logic, edge cases, and error paths.
3. Documentation with basic and advanced usage.
4. Safe-by-default handling for sensitive AI inputs/outputs.
5. Deterministic outputs for the same inputs/config.
6. Changelog entry and upgrade guidance when relevant.

## Suggested Evidence Artifacts

- Test report and coverage summary.
- Example recipe demonstrating end-to-end workflow.
- API diff for the release.
- Short architecture note (ports/adapters/services).
