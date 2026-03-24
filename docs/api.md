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

---

For more detailed examples, see the [User Guide](user-guide/core.md) and [Recipes](recipes/cli-tool.md).
