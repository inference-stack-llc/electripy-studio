# ElectriPy Studio

Production-minded Python components and recipes by Inference Stack.

## Overview

ElectriPy Studio is a curated collection of production-ready Python components and recipes designed to accelerate development while maintaining high code quality standards.

## Status

- **Last updated**: 2026-03-24
- **Maturity**: Early alpha (APIs may evolve), but core components, CLI, concurrency primitives, and a growing suite of AI product engineering utilities are in place.

## Features

- **Core Components**: Configuration, logging, error handling, and type utilities
- **Concurrency**: Retry mechanisms (sync/async) and async token bucket rate limiter
- **I/O**: JSONL read/write utilities for efficient data processing
- **CLI**: Typer-based command-line interface with health checks and evaluation commands
- **AI & LLM Gateway**: Provider-agnostic LLM clients with structured output and safety seams, plus a RAG Evaluation Runner for benchmarking retrieval quality.
- **AI Telemetry**: Provider-agnostic telemetry primitives and adapters for HTTP resilience, LLM gateway, policy decisions, and RAG evaluation, with a safe-by-default posture.
- **AI Product Engineering Utilities**: Streaming chat, deterministic agent runtime helpers, RAG quality/drift metrics, hallucination-risk grounding checks, response robustness helpers, prompt templating, token budget management, priority-based context assembly, rule-based model routing, conversation memory, and declarative tool registry.

## Documentation Map

![Documentation and recipes layout](images/docs_and_recipes.png)

## Quick Links

- [Installation](getting-started/installation.md)
- [Quickstart Guide](getting-started/quickstart.md)
- [API Reference](api.md)
 - [Core Concepts](user-guide/core.md)
 - [Concurrency & Resilience](user-guide/concurrency.md)
 - [I/O Utilities](user-guide/io.md)
 - [CLI Guide](user-guide/cli.md)
 - [LLM Gateway & AI](user-guide/ai-llm-gateway.md)
 - [AI Product Engineering Utilities](user-guide/ai-product-engineering.md)
 - [AI Telemetry](user-guide/ai-telemetry.md)
 - [RAG Evaluation Runner](user-guide/ai-rag-eval-runner.md)

## Requirements

- Python 3.11 or higher
- Modern dependency management (pip, poetry, or uv)

## License

MIT License - See [LICENSE](https://github.com/reactlabs-dev/electripy-studio/blob/main/LICENSE) for details.
