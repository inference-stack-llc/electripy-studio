# Concurrency Module

Utilities for handling retries and rate limiting in concurrent applications.

## Retry Mechanism

Decorators for automatic retry with exponential backoff.

### Synchronous Retry

```python
from electripy.concurrency import retry

@retry(max_attempts=3, delay=1.0, backoff=2.0)
def fetch_data_from_api():
    # This will retry up to 3 times with exponential backoff
    response = requests.get("https://api.example.com/data")
    return response.json()
```

### Asynchronous Retry

```python
from electripy.concurrency import async_retry

@async_retry(max_attempts=5, delay=0.5, backoff=2.0)
async def fetch_data_async():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/data") as response:
            return await response.json()
```

### Parameters

- `max_attempts`: Maximum number of retry attempts (default: 3)
- `delay`: Initial delay between retries in seconds (default: 1.0)
- `backoff`: Multiplier for exponential backoff (default: 2.0)
- `exceptions`: Tuple of exception types to catch (default: (Exception,))

### Custom Exceptions

```python
from electripy.concurrency import retry
from requests.exceptions import RequestException

@retry(
    max_attempts=5,
    delay=2.0,
    exceptions=(RequestException,)
)
def fetch_with_custom_retry():
    # Only retries on RequestException
    return requests.get("https://api.example.com/data")
```

## Rate Limiter

Async token bucket rate limiter for controlling request rates.

### Basic Usage

```python
from electripy.concurrency import AsyncTokenBucketRateLimiter

# Create limiter: 10 requests per second
limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=10)

async def make_requests():
    for i in range(100):
        async with limiter:
            # This operation is rate-limited
            await api_call()
```

### Manual Token Acquisition

```python
limiter = AsyncTokenBucketRateLimiter(rate=5, capacity=10)

# Acquire multiple tokens at once
await limiter.acquire(tokens=3)

# Check available tokens
available = limiter.available_tokens
```

### Parameters

- `rate`: Number of tokens added per second
- `capacity`: Maximum tokens in bucket (defaults to rate)

### Use Cases

- API rate limiting
- Database connection throttling
- Resource-intensive operations
- Burst handling with sustained rate control

## Task groups and bounded worker pools

For async fan-out workloads such as RAG pipelines or batch LLM calls,
use the task-group utilities to apply backpressure-aware concurrency
limits.

```python
from electripy.concurrency.task_groups import gather_limited, map_limited


async def call_api(i: int) -> int:
    # Your async work here
    await asyncio.sleep(0.01)
    return i * 2


async def run_batch() -> None:
    items = list(range(10))

    # Map with bounded concurrency
    results = await map_limited(call_api, items, concurrency=5)

    # Or run an existing collection of coroutines
    coros = [call_api(i) for i in items]
    results_again = await gather_limited(coros, concurrency=5)
```

Both helpers preserve the original order of items while ensuring that
no more than `concurrency` tasks are in flight at any time.
