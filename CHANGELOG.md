# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-03-25

### Added

- **Fallback Chain** — automatic provider failover across ranked `SyncLlmPort` adapters with metadata tracking (`_fallback_provider_index`).
- **Batch Complete** — fan-out N LLM requests with bounded concurrency (`ThreadPoolExecutor`), order-preserving results, per-request error isolation, and progress callbacks.
- **Cost Ledger** — thread-safe token cost accumulation with multi-dimensional label slicing (`by_label`), estimated cost calculation, and snapshot/reset support.
- **Prompt Fingerprint** — deterministic SHA-256 request hashing (compatible with LLM Cache key algorithm) with full and short digest variants.
- **JSON Repair** — fix 7 common LLM JSON breakage patterns: markdown fences, trailing commas, single quotes, unquoted keys, mismatched brackets, and truncated JSON.
- **Circuit Breaker** — closed→open→half_open FSM for cascading failure protection with configurable thresholds, decorator support, and thread-safe state transitions.
- **Sensitive Data Scanner** — regex-based PII and secret detection with 9 built-in patterns (email, phone, SSN, credit card, API keys, AWS, IPv4), extensible via `add_pattern()`.
- User guide documentation for all seven new components.
- 82 new tests (total suite now at 351).

## [0.2.0] — 2026-03-25

### Added

- **Structured Output Engine** — extract typed Pydantic models from LLM text with auto-retry and temperature decay.
- **LLM Caching Layer** — pluggable response caching with in-memory LRU and SQLite WAL backends, deterministic cache keys, and hit-rate tracking.
- **LLM Replay Tape** — record, replay, and diff LLM interactions as JSONL tapes for deterministic offline tests and output regression detection.
- **Eval Assertions** — pytest-native assertion helpers (keyword, regex, JSON schema, predicate, length) with structured diagnostic reports.
- **Provider Adapters** — OpenAI, Anthropic (Messages API), and Ollama (HTTP) adapters with lazy imports and domain exception mapping.
- User guide documentation for all five new components.
- 105 new tests (total suite now at 269).

## [0.1.3] — 2026-03-25

### Fixed

- All README links converted to absolute GitHub URLs so they work on PyPI.
- PyPI badge cache-buster to force fresh render on GitHub.

## [0.1.2] — 2026-03-25

### Fixed

- Repository map and development workflow images now use absolute URLs for PyPI rendering.

## [0.1.1] — 2026-03-25

### Fixed

- README social-preview image now uses absolute URL so it renders correctly on PyPI.

### Added

- **Policy Gateway** — regex-based detection, sanitization, deny/allow/require-approval actions across preflight, postflight, stream, and tool-call stages.
- **Agent Collaboration Runtime** — bounded multi-agent orchestration with hop limits, deque-based message routing, and optional policy gateway integration.
- **LLM Gateway hooks** — `build_llm_policy_hooks()` bridge so policy decisions plug directly into the LLM call path.
- `electripy demo policy-collab` CLI command — run the full policy + agent collaboration pipeline offline with a Rich table report.
- Enterprise-grade code quality pass — `frozen=True` value objects, `__slots__` on services, `__all__` exports, `Sequence`/`Mapping` Protocol signatures.
- Comparison table in README — positioning against LiteLLM, Guardrails AI, CrewAI, RAGAS, Instructor, Pydantic AI, Haystack/LangChain.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`.
- GitHub issue templates (bug report, feature request).

### Fixed

- Policy gateway sanitizer: empty-string `replacement` no longer silently falls through to default.

## [0.1.0] — 2026-03-25

### Added

- **Core**: Configuration, structured logging, typed error hierarchy.
- **Concurrency**: Retry (sync/async) with exponential backoff, async token-bucket rate limiter, task groups.
- **I/O**: JSONL read/write utilities.
- **CLI**: Typer-based interface with `doctor`, `rag eval` commands.
- **AI — LLM Gateway**: Provider-agnostic sync/async clients with structured output helpers.
- **AI — RAG Evaluation Runner**: Dataset loader, evaluation runner, CLI benchmarking.
- **AI — Telemetry**: Provider-agnostic observability primitives (JSONL, optional OpenTelemetry).
- **AI — Product engineering utilities**: Streaming chat, agent runtime, RAG quality/drift, hallucination guard, response robustness, prompt engine, token budget, context assembly, model router, conversation memory, tool registry.
- Documentation site (MkDocs Material).
- GitHub Actions CI (ruff, black, mypy, pytest on 3.11 + 3.12, mkdocs build --strict).
