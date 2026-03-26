# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
