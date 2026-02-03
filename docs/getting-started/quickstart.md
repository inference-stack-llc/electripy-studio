# Quickstart

Get started with ElectriPy in minutes.

## Basic Usage

### Configuration

```python
from electripy import Config, get_logger

# Load config from environment
config = Config.from_env()

# Get a logger
logger = get_logger(__name__)
logger.info("Application started")
```

### Retry Mechanism

```python
from electripy.concurrency import retry, async_retry

# Synchronous retry
@retry(max_attempts=3, delay=1.0, backoff=2.0)
def fetch_data():
    # Your code here
    return data

# Asynchronous retry
@async_retry(max_attempts=3, delay=1.0)
async def fetch_data_async():
    # Your async code here
    return data
```

### Rate Limiting

```python
from electripy.concurrency import AsyncTokenBucketRateLimiter

# Create rate limiter (10 requests per second)
limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=10)

async with limiter:
    # Rate-limited operation
    await make_api_call()
```

### JSONL Operations

```python
from electripy.io import read_jsonl, write_jsonl, append_jsonl

# Write JSONL file
data = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
]
write_jsonl("output.jsonl", data)

# Read JSONL file
for record in read_jsonl("output.jsonl"):
    print(record)

# Append to JSONL file
append_jsonl("log.jsonl", {"event": "user_login", "timestamp": "2024-01-01"})
```

### CLI Commands

Check installation health:

```bash
electripy doctor
```

Check version:

```bash
electripy version
```

Get help:

```bash
electripy --help
```

## Next Steps

- Explore the [User Guide](../user-guide/core.md) for detailed information
- Check out [Recipes](../recipes/cli-tool.md) for common patterns
- Read the [API Reference](../api.md) for complete documentation
