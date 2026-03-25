# API Reference

Complete API reference for ElectriPy modules.

## Core Module

### Config

::: electripy.core.config.Config

### Logging

- `setup_logging(level: str = "INFO", format_type: str = "json") -> None`
- `get_logger(name: str) -> logging.Logger`

### Errors

- `ElectriPyError`: Base exception
- `ConfigError`: Configuration errors
- `ValidationError`: Validation failures
- `RetryError`: Retry exhaustion

### Types

- `JSONValue`: Union of JSON-serializable types
- `JSONDict`: Dictionary with string keys and JSON values

## Concurrency Module

### Retry

- `@retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(Exception,))`
- `@async_retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(Exception,))`

### Rate Limiter

::: electripy.concurrency.rate_limiter.AsyncTokenBucketRateLimiter

### Task Groups

- `gather_limited(coros, concurrency: int) -> list[T]`
- `map_limited(fn, items, concurrency: int) -> list[U]`

## I/O Module

### JSONL

- `read_jsonl(path, encoding="utf-8") -> Generator[JSONDict, None, None]`
- `write_jsonl(path, data, encoding="utf-8") -> None`
- `append_jsonl(path, record, encoding="utf-8") -> None`

## CLI Module

### Commands

- `electripy doctor`: Health check
- `electripy version`: Show version
- `electripy --help`: Show help

### App

::: electripy.cli.app

## AI Utilities Module

### Streaming Chat

- `StreamChunk`: typed stream chunk model
- `collect_text(chunks) -> str`
- `async_collect_text(chunks) -> str`
- `with_timeout(chunks, timeout_seconds=...) -> AsyncIterator[StreamChunk]`

### Agent Runtime

- `ToolInvocation`: tool call model
- `AgentExecutor.run(plan) -> AgentRunResult`

### RAG Quality

- `hit_rate_at_k(retrieved_ids, relevant_ids, k) -> float`
- `precision_at_k(retrieved_ids, relevant_ids, k) -> float`
- `recall_at_k(retrieved_ids, relevant_ids, k) -> float`
- `mrr_at_k(retrieved_ids, relevant_ids, k) -> float`
- `retrieval_drift(baseline, candidate, k=...) -> DriftComparison`

### Hallucination Guard

- `extract_citation_ids(text) -> list[str]`
- `evaluate_grounding(response_text=..., evidence_texts=..., min_overlap=...) -> GroundingCheckResult`

### Response Robustness

- `extract_json_object(text) -> str`
- `parse_json_with_repair(text) -> JsonRepairResult`
- `require_fields(value, fields) -> None`
- `coalesce_non_empty(candidates) -> str`

---

For more detailed examples, see the [User Guide](user-guide/core.md) and [Recipes](recipes/cli-tool.md).
