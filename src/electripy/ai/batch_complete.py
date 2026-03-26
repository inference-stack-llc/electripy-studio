"""Batch completion — fan-out N prompts with concurrency control.

Purpose:
  - Send many LLM requests in parallel with a configurable concurrency cap.
  - Collect results (or errors) in order with an optional progress callback.
  - Works with any ``SyncLlmPort`` via a thread pool.

Guarantees:
  - Deterministic result ordering — results[i] corresponds to requests[i].
  - Failed requests return the exception, not crash the batch.
  - Thread-safe — uses :class:`concurrent.futures.ThreadPoolExecutor`.

Usage::

    from electripy.ai.batch_complete import batch_complete

    results = batch_complete(
        port=openai_adapter,
        requests=my_requests,
        max_concurrency=5,
        on_progress=lambda done, total: print(f"{done}/{total}"),
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"Failed: {r}")
        else:
            print(r.text)
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Sequence

from electripy.ai.llm_gateway.domain import LlmRequest, LlmResponse
from electripy.ai.llm_gateway.ports import SyncLlmPort

__all__ = [
    "batch_complete",
    "BatchResult",
]

# A batch result is either a successful response or the exception.
BatchResult = LlmResponse | Exception


def batch_complete(
    *,
    port: SyncLlmPort,
    requests: Sequence[LlmRequest],
    max_concurrency: int = 5,
    timeout: float | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[BatchResult]:
    """Send *requests* through *port* with bounded parallelism.

    Args:
        port: Any ``SyncLlmPort`` adapter.
        requests: Ordered sequence of LLM requests.
        max_concurrency: Maximum number of in-flight calls.
        timeout: Per-request timeout in seconds forwarded to the port.
        on_progress: Optional ``(completed, total)`` callback invoked
            after every individual request finishes.

    Returns:
        A list of the same length as *requests*.  Each element is either
        an :class:`LlmResponse` on success or the raised :class:`Exception`.
    """
    if not requests:
        return []

    total = len(requests)
    results: list[BatchResult] = [RuntimeError("placeholder")] * total
    completed = 0

    def _run_one(index: int, request: LlmRequest) -> tuple[int, BatchResult]:
        try:
            return index, port.complete(request, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return index, exc

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_concurrency, total),
    ) as pool:
        futures = [pool.submit(_run_one, i, req) for i, req in enumerate(requests)]

        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)

    return results
