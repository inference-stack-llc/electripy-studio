"""Tests for concurrency.retry module."""

import pytest

from electripy.concurrency.retry import async_retry, retry
from electripy.core.errors import RetryError


def test_retry_success_first_attempt() -> None:
    """Test retry succeeds on first attempt."""
    call_count = 0
    
    @retry(max_attempts=3)
    def func() -> str:
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = func()
    assert result == "success"
    assert call_count == 1


def test_retry_success_after_failures() -> None:
    """Test retry succeeds after some failures."""
    call_count = 0
    
    @retry(max_attempts=3, delay=0.01)
    def func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary error")
        return "success"
    
    result = func()
    assert result == "success"
    assert call_count == 3


def test_retry_exhausted() -> None:
    """Test retry raises RetryError when exhausted."""
    @retry(max_attempts=2, delay=0.01)
    def func() -> None:
        raise ValueError("Always fails")
    
    with pytest.raises(RetryError):
        func()


@pytest.mark.asyncio
async def test_async_retry_success() -> None:
    """Test async_retry succeeds on first attempt."""
    call_count = 0
    
    @async_retry(max_attempts=3)
    async def func() -> str:
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = await func()
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_async_retry_with_failures() -> None:
    """Test async_retry succeeds after failures."""
    call_count = 0
    
    @async_retry(max_attempts=3, delay=0.01)
    async def func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary error")
        return "success"
    
    result = await func()
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_async_retry_exhausted() -> None:
    """Test async_retry raises RetryError when exhausted."""
    @async_retry(max_attempts=2, delay=0.01)
    async def func() -> None:
        raise ValueError("Always fails")
    
    with pytest.raises(RetryError):
        await func()
