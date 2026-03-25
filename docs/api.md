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

### Prompt Engine

- `render_template(template, variables) -> str`: Replace `{{var}}` placeholders in a template string.
- `build_few_shot_block(examples, max_examples=...) -> list[RenderedMessage]`: Convert few-shot examples into interleaved user/assistant messages.
- `compose_messages(system=..., few_shot=..., user=..., variables=...) -> RenderedPrompt`: Compose a full chat prompt from building blocks.
- `FewShotExample`: Typed few-shot example pair.
- `RenderedPrompt.to_dicts() -> list[dict]`: Export messages for LLM API payloads.

### Token Budget

- `TokenizerPort`: Protocol for pluggable token counting.
- `CharEstimatorTokenizer(chars_per_token=4.0)`: Zero-dependency character-based token estimator.
- `count_tokens(text, tokenizer) -> TokenCount`
- `fits_budget(text, budget, tokenizer) -> bool`
- `truncate_to_budget(text, budget, tokenizer, strategy=..., strict=...) -> TruncationResult`
- `TruncationStrategy`: TAIL, HEAD, or MIDDLE truncation.

### Context Assembly

- `ContextBlock(label, content, priority)`: A block of content with a priority level.
- `ContextPriority`: LOW, MEDIUM, HIGH, CRITICAL.
- `assemble_context(blocks, budget, tokenizer) -> AssembledContext`: Pack blocks into a token-limited window, dropping lowest priority first.

### Model Router

- `ModelProfile(model_id, provider, cost_tier, ...)`: Model capability/cost profile.
- `RoutingRule(name, predicate)`: Composable model selection predicate.
- `ModelRouter(models).route(rules) -> RoutingDecision`: Select cheapest model satisfying all rules.
- `CostTier`: FREE, LOW, MEDIUM, HIGH, PREMIUM.

### Conversation Memory

- `append_turn(window, role, content, tokenizer) -> ConversationWindow`
- `recent_turns(window, n) -> ConversationWindow`
- `sliding_window(window, max_turns, tokenizer) -> ConversationWindow`
- `trim_to_budget(window, budget, tokenizer, preserve_system=True) -> ConversationWindow`
- `ConversationWindow.to_dicts() -> list[dict]`: Export for LLM API payloads.

### Tool Registry

- `tool_from_function(func, name=..., description=...) -> ToolDefinition`: Create tool definitions from Python functions.
- `generate_schema(func) -> ToolSchema`: Infer JSON Schema from function signature.
- `validate_arguments(tool, arguments) -> dict`: Validate and fill defaults.
- `ToolRegistry()`: Register, look up, and export tools.
- `ToolRegistry.to_openai_tools() -> list[dict]`: Export in OpenAI function-calling format.

---

For more detailed examples, see the [User Guide](user-guide/core.md) and [Recipes](recipes/cli-tool.md).
