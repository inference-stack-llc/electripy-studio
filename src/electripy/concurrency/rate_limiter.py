"""Async token bucket rate limiter implementation."""

import asyncio
import time


class AsyncTokenBucketRateLimiter:
    """Async token bucket rate limiter for controlling request rates.

    The token bucket algorithm allows bursts up to the bucket capacity while
    maintaining an average rate over time.

    Example:
        limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=10)
        async with limiter:
            # Make rate-limited request
            await make_api_call()
    """

    def __init__(self, rate: float, capacity: float | None = None):
        """Initialize the rate limiter.

        Args:
            rate: Number of tokens to add per second
            capacity: Maximum number of tokens in bucket (defaults to rate)
        """
        self.rate = rate
        self.capacity = capacity or rate
        self._tokens = self.capacity
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquire tokens from the bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Raises:
            ValueError: If tokens exceeds capacity
        """
        if tokens > self.capacity:
            raise ValueError(f"Requested {tokens} tokens exceeds capacity {self.capacity}")

        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_update

                # Add tokens based on elapsed time
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last_update = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Calculate wait time for required tokens
                needed = tokens - self._tokens
                wait_time = needed / self.rate
                await asyncio.sleep(wait_time)

    async def __aenter__(self) -> "AsyncTokenBucketRateLimiter":
        """Context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Context manager exit."""
        pass

    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens (approximate).

        Returns:
            Number of tokens currently available
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        return min(self.capacity, self._tokens + elapsed * self.rate)
