<p align="center">
  <img src="https://raw.githubusercontent.com/inference-stack-llc/electripy-studio/main/images/social-preview.png" alt="ElectriPy Studio — Production-grade Python toolkit for AI product engineering" width="100%">
</p>

# ElectriPy Studio

Production-minded Python components and recipes (cookbook) by Inference Stack.

[![CI](https://github.com/inference-stack-llc/electripy-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/inference-stack-llc/electripy-studio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/electripy-studio?color=brightgreen&v=2)](https://pypi.org/project/electripy-studio/)
[![Release](https://img.shields.io/github/v/release/inference-stack-llc/electripy-studio?label=release&color=brightgreen)](https://github.com/inference-stack-llc/electripy-studio/releases/latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ElectriPy Studio is a curated collection of production-ready Python components and recipes designed to accelerate development while maintaining high code quality standards.

## Design principles

- Ports & Adapters: swap providers (LLMs, embedders, vector stores) without rewriting business logic.
- Deterministic by default: stable IDs and reproducible evaluation runs.
- Safe logging posture: avoid leaking prompts/responses; prefer hashes + redaction seams.
- Typed, production APIs: small public surfaces, strict typing, structured outputs where it matters.
- Testability: unit tests are offline and deterministic by default (no network required).

## How ElectriPy compares

ElectriPy is **not** a framework — it's a composable toolkit of production-grade building blocks for AI-powered Python applications. Here's how it relates to popular alternatives:

| Library | Stars | Overlap | ElectriPy's edge |
| --- | --- | --- | --- |
| [LiteLLM](https://github.com/BerriAI/litellm) | 40 k | Provider-agnostic LLM gateway | Bundles policy hooks, telemetry & structured output inline — no proxy server needed |
| [Guardrails AI](https://github.com/guardrails-ai/guardrails) | 6.6 k | Input / output validation | Lighter-weight, composable policy gateway — no XML DSL or Hub dependency |
| [CrewAI](https://github.com/crewAIInc/crewAI) / [AutoGen](https://github.com/microsoft/autogen) | 50 k+ | Multi-agent orchestration | Bounded & deterministic with hop limits; building-block, not a framework |
| [RAGAS](https://github.com/explodinggradients/ragas) | 13 k | RAG evaluation metrics | Integrates eval directly into CLI & CI gating; ships with drift comparison |
| [Instructor](https://github.com/instructor-ai/instructor) | 12.6 k | Structured LLM output | Wraps the pattern alongside retry, token budget & telemetry in one toolkit |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | 10 k+ | Typed AI agents | Narrower scope; ElectriPy ships concurrency, I/O, CLI & observability too |
| [Haystack](https://github.com/deepset-ai/haystack) / [LangChain](https://github.com/langchain-ai/langchain) | 40 k+ | Full RAG / agent framework | Composable building blocks you import — not a framework you adopt wholesale |

> **TL;DR** — Use ElectriPy when you want discrete, well-typed utilities that compose into **your** architecture rather than a monolithic framework that owns it.

## Status & recent updates

- **Last updated**: 2026-03-25
- **Maturity**: Early alpha (APIs may still evolve), but core components, CLI, concurrency primitives, and a growing suite of AI product engineering utilities are in place.
- **Versioning**: SemVer begins at `v0.x` — expect breaking changes until `v1.0`.
- **Recent highlights**:
    - Added `electripy demo policy-collab` CLI command — run the full policy + agent collaboration pipeline offline with a Rich table report.
    - Added a **Policy Gateway** with regex-based detection, sanitization, deny/allow/require-approval actions across preflight, postflight, stream, and tool-call stages.
    - Added a **Agent Collaboration Runtime** for bounded multi-agent orchestration with hop limits, deque-based message routing, and optional policy gateway integration.
    - Added **LLM Gateway request/response hooks** and `build_llm_policy_hooks()` bridge so policy decisions plug directly into the LLM call path.
    - Added an LLM Gateway for provider-agnostic LLM calls with structured output and safety seams.
    - Added a RAG Evaluation Runner and `electripy rag eval` CLI for benchmarking retrieval quality over JSONL datasets.
    - Added an AI Telemetry component for safe, provider-agnostic observability across HTTP resilience, LLM gateway, policy decisions, and RAG evaluation.
    - Phase 1: Streaming chat, agent runtime, RAG quality/drift, hallucination guard, and response robustness utilities.
    - Phase 2: Prompt engine, token budget management, context assembly, model routing, conversation memory, and tool registry.
    - Expanded documentation and user guides for core, concurrency, I/O, CLI, AI, and observability components.

## Features

- 🔧 **Core Components**: Configuration, logging, error handling, and type utilities
- ⚡ **Concurrency**: Retry mechanisms (sync/async) and async token bucket rate limiter
- 📁 **I/O**: JSONL read/write utilities for efficient data processing
- 💻 **CLI**: Typer-based command-line interface with health checks, RAG eval runner, and an offline demo showcase (`electripy demo policy-collab`)
- 🤖 **AI building blocks**: Provider-agnostic LLM Gateway with sync/async clients, request/response policy hooks, structured-output helpers, and a RAG Evaluation Runner for retrieval benchmarking.
- 📊 **AI Telemetry**: Provider-agnostic telemetry primitives and adapters (JSONL, optional OpenTelemetry) for HTTP resilience, LLM gateway, policy decisions, and RAG evaluation runs.
- 🧠 **AI product engineering utilities**: Streaming chat primitives, deterministic agent runtime helpers, RAG quality/drift metrics, grounding checks for hallucination reduction, response robustness helpers for structured outputs, prompt templating and composition, token budget tracking and truncation, priority-based context window assembly, rule-based model routing, sliding-window conversation memory, and a declarative tool registry with JSON schema generation.
- 🛡️ **AI policy and collaboration runtime**: Deterministic policy gateway checks for preflight/postflight/stream/tool flows, plus bounded agent-to-agent collaboration runtime for specialist orchestration patterns.

## Quick Start

### Using ElectriPy as a library

For most users, ElectriPy is just a Python library you depend on in your own project.

Install from PyPI:

```bash
pip install electripy-studio
```

To work against a local clone in editable mode (e.g., to experiment with changes while using it in another project):

```bash
pip install -e .
```

For development on this repo itself (full tooling and test extras):

```bash
pip install -e ".[dev]"
```

### Verify Installation

```bash
electripy doctor
```

### Basic Usage

```python
from electripy import Config, get_logger
from electripy.concurrency import retry, AsyncTokenBucketRateLimiter
from electripy.io import read_jsonl, write_jsonl

# Configuration
config = Config.from_env()
logger = get_logger(__name__)

# Retry with exponential backoff
@retry(max_attempts=3, delay=1.0, backoff=2.0)
def fetch_data():
    return api_call()

# Rate limiting
limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=10)
async with limiter:
    await rate_limited_operation()

# JSONL I/O
data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
write_jsonl("output.jsonl", data)

for record in read_jsonl("output.jsonl"):
    print(record)
```

### AI quick start (LLM + RAG eval)

- Run a basic RAG evaluation over JSONL datasets:

```bash
electripy rag eval --corpus data/corpus.jsonl --queries data/queries.jsonl \
    --top-k 3,5,10 --report-json out/report.json
```

- LLM Gateway usage (offline-friendly fake provider example): see [recipes/02_llm_gateway/](https://github.com/inference-stack-llc/electripy-studio/tree/main/recipes/02_llm_gateway/).

### Demo: Policy Gateway + Agent Collaboration

Run the full pipeline offline — no API keys needed:

```bash
electripy demo policy-collab
```

Customise with `--prompt` and `--max-hops`:

```bash
electripy demo policy-collab --prompt "Alert user@corp.io about outage" --max-hops 6
```

See [recipes/03_policy_collaboration/](https://github.com/inference-stack-llc/electripy-studio/tree/main/recipes/03_policy_collaboration/) for the standalone script.

## Documentation

Full documentation is available in the [docs/](https://github.com/inference-stack-llc/electripy-studio/tree/main/docs) directory:

- [Installation Guide](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/getting-started/installation.md)
- [Quickstart](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/getting-started/quickstart.md)
- [Core Concepts](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/core.md)
- [Concurrency & Resilience](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/concurrency.md)
- [I/O Utilities](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/io.md)
- [CLI Guide](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/cli.md)
- [LLM Gateway & AI](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/ai-llm-gateway.md)
- [AI Telemetry](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/ai-telemetry.md)
- [AI Policy Gateway](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/ai-policy-gateway.md)
- [AI Agent Collaboration Runtime](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/ai-agent-collaboration.md)
- [RAG Evaluation Runner](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/ai-rag-eval-runner.md)
- [AI Product Engineering Utilities](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/ai-product-engineering.md)
- [Component Maturity Model](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/user-guide/component-maturity.md)
- [Recipes](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/recipes/cli-tool.md)
- [API Reference](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/api.md)

Build and serve docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Visual Overview

### Repository Map

![Repository map](https://raw.githubusercontent.com/inference-stack-llc/electripy-studio/main/docs/images/repo_map.png)

### Development Workflow

![Development workflow](https://raw.githubusercontent.com/inference-stack-llc/electripy-studio/main/docs/images/dev_workflow.png)

## Project Structure

```
electripy-studio/
├── src/electripy/          # Main package
│   ├── core/               # Config, logging, errors, typing
│   ├── concurrency/        # Retry & rate limiting
│   ├── io/                 # JSONL utilities
│   ├── cli/                # CLI commands
│   └── ai/                 # AI building blocks and product-engineering utilities
│       ├── llm_gateway/    # Provider-agnostic LLM client + structured output helpers
│       ├── rag_eval_runner/# Dataset + eval runner + CLI benchmarking
│       ├── streaming_chat/ # Sync/async stream chunk and collection helpers
│       ├── agent_runtime/  # Deterministic tool-plan execution primitives
│       ├── rag_quality/    # Retrieval metrics and drift comparison helpers
│       ├── hallucination_guard/ # Grounding and citation checks
│       ├── response_robustness/ # JSON extraction/repair and output guards
│       ├── prompt_engine/       # Template composition and few-shot management
│       ├── token_budget/        # Pluggable token counting and truncation
│       ├── context_assembly/    # Priority-based context window packing
│       ├── model_router/        # Rule-based model selection and routing
│       ├── conversation_memory/ # Sliding window and token-aware chat history
│       ├── policy_gateway/      # Deterministic pre/post/tool/stream policy decisions
│       ├── tool_registry/       # Declarative tool definitions and JSON schema
│       └── agent_collaboration/ # Bounded multi-agent handoff orchestration
├── tests/                  # Test suite
├── docs/                   # Documentation
├── recipes/                # Example recipes
│   ├── 01_cli_tool/        # CLI tool example
│   ├── 02_llm_gateway/     # LLM gateway examples
│   └── 03_policy_collaboration/ # End-to-end policy + multi-agent flow
├── packages/               # NPM packages
│   └── electripy-cli/      # NPM CLI wrapper
├── pyproject.toml          # Project config
├── mkdocs.yml              # Docs config
└── LICENSE                 # MIT License
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
black .

# Type checking
mypy src/
```

### Python Tooling (recommended)

These tools are **optional but recommended for contributors** working on ElectriPy Studio itself. They are installed globally (via `pipx`) and then used inside whatever project or virtualenv you prefer.

#### 1. Install global CLI tools with pipx

`pipx` lets you install Python CLIs in isolated environments, so they don't conflict with your project dependencies:

```bash
python -m pip install --upgrade pip

brew install pipx      # or see https://pipx.pypa.io for other platforms
pipx ensurepath

pipx install uv        # fast Python package/dependency manager
pipx install poetry    # project/virtualenv manager (optional; this repo uses pyproject + hatchling)
pipx install ruff      # fast linter (also available via .[dev] extra)
pipx install pre-commit  # git pre-commit hooks runner
```

#### 2. Using uv (optional)

`uv` is a fast drop-in for many `pip`/`python -m venv` workflows. For example, to create a fresh environment for hacking on ElectriPy Studio:

```bash
uv venv .venv
source .venv/bin/activate

uv pip install -e ".[dev]"
```

You can also use `uv pip install electripy-studio` in your own projects.

#### 3. Using poetry in your own projects (optional)

This repo is built with `pyproject.toml` + Hatchling, but you can happily **consume** ElectriPy from a Poetry-managed project:

```bash
poetry add electripy-studio
```

The library itself has no dependency on Poetry; it's just a convenient project manager if you already use it.

#### 4. pre-commit (for contributors)

Once `pre-commit` is installed, enable the hooks defined in [.pre-commit-config.yaml](https://github.com/inference-stack-llc/electripy-studio/blob/main/.pre-commit-config.yaml):

```bash
pre-commit install
```

This will automatically run Black, Ruff, and basic whitespace checks on changed files before each commit.

### CI/CD

GitHub Actions automatically runs tests, linting, and type checking on all pull requests. See [.github/workflows/ci.yml](https://github.com/inference-stack-llc/electripy-studio/blob/main/.github/workflows/ci.yml).

## Recipes

Check out the [recipes/](https://github.com/inference-stack-llc/electripy-studio/tree/main/recipes) directory for complete examples:

- [01_cli_tool](https://github.com/inference-stack-llc/electripy-studio/tree/main/recipes/01_cli_tool/) — Building a production-ready CLI tool
- [02_llm_gateway](https://github.com/inference-stack-llc/electripy-studio/tree/main/recipes/02_llm_gateway/) — LLM Gateway basics using a fake provider (offline-friendly)
- [03_policy_collaboration](https://github.com/inference-stack-llc/electripy-studio/tree/main/recipes/03_policy_collaboration/) — End-to-end policy gateway + LLM hooks + multi-agent collaboration demo

Additional recipe guides are available in the docs:

- [Policy Gateway recipe](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/recipes/policy-gateway.md) — standalone policy evaluation walkthrough
- [Agent Collaboration Runtime recipe](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/recipes/agent-collaboration-runtime.md) — bounded agent handoff patterns
- [Policy + Collaboration E2E recipe](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/recipes/policy-collaboration-e2e.md) — full pipeline with telemetry
- [RAG Evaluation Runner recipe](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/recipes/rag-eval-runner.md) — benchmarking retrieval quality
- [AI Telemetry recipe](https://github.com/inference-stack-llc/electripy-studio/blob/main/docs/recipes/ai-telemetry.md) — wiring observability across components

## Requirements

- Python 3.11 or higher
- Dependencies managed via `pyproject.toml`

## License

MIT License - See [LICENSE](https://github.com/inference-stack-llc/electripy-studio/blob/main/LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](https://github.com/inference-stack-llc/electripy-studio/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/inference-stack-llc/electripy-studio/blob/main/CODE_OF_CONDUCT.md) before submitting PRs. For security issues, see [SECURITY.md](https://github.com/inference-stack-llc/electripy-studio/blob/main/SECURITY.md).

## Links

- [GitHub Repository](https://github.com/inference-stack-llc/electripy-studio)
- [Documentation](https://github.com/inference-stack-llc/electripy-studio/tree/main/docs)
- [Issue Tracker](https://github.com/inference-stack-llc/electripy-studio/issues)
