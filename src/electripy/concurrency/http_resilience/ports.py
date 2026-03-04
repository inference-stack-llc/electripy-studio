"""Ports (Protocols) for HTTP clients used by the resilience component.

These Protocols define the minimal surface required by the orchestration
layer. Concrete adapters (e.g., httpx-based clients) must implement these
interfaces, allowing the underlying HTTP library to be swapped without
changing business logic.

Example:
    from electripy.concurrency.http_resilience.domain import ResponseData
    from electripy.concurrency.http_resilience.ports import SyncHttpPort

    class MySyncAdapter(SyncHttpPort):
        def request(self, method: str, url: str, **kwargs: object) -> ResponseData:
            ...
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .domain import BodyType, Headers, QueryParams, ResponseData


@runtime_checkable
class SyncHttpPort(Protocol):
    """Synchronous HTTP client port.

    Implementations must perform an outbound HTTP request and normalize the
    result into a :class:`ResponseData` instance. They must not raise raw
    third-party exceptions; instead, they should map them to domain
    exceptions from :mod:`electripy.concurrency.http_resilience.domain`.

    Args:
        method: HTTP method string (e.g., "GET").
        url: Absolute or relative URL.
        headers: Optional request headers.
        params: Optional query parameters.
        json: Optional JSON-serializable body.
        data: Optional raw body or form data.
        timeout_s: Optional per-request timeout in seconds.

    Returns:
        ResponseData: Normalized response object.

    Raises:
        HttpClientError: For domain-level HTTP errors.

    Example:
        response = port.request("GET", "https://api.example.com")
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Headers | None = None,
        params: QueryParams | None = None,
        json: object | None = None,
        data: BodyType | None = None,
        timeout_s: float | None = None,
    ) -> ResponseData:
        """Perform a synchronous HTTP request.

        See the class docstring for parameter details.
        """

        raise NotImplementedError


@runtime_checkable
class AsyncHttpPort(Protocol):
    """Asynchronous HTTP client port.

    The async counterpart of :class:`SyncHttpPort`. Implementations must
    normalize responses into :class:`ResponseData` and map low-level errors
    into domain exceptions.

    Args:
        method: HTTP method string (e.g., "GET").
        url: Absolute or relative URL.
        headers: Optional request headers.
        params: Optional query parameters.
        json: Optional JSON-serializable body.
        data: Optional raw body or form data.
        timeout_s: Optional per-request timeout in seconds.

    Returns:
        ResponseData: Normalized response object.

    Raises:
        HttpClientError: For domain-level HTTP errors.

    Example:
        response = await port.request("GET", "https://api.example.com")
    """

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Headers | None = None,
        params: QueryParams | None = None,
        json: object | None = None,
        data: BodyType | None = None,
        timeout_s: float | None = None,
    ) -> ResponseData:
        """Perform an asynchronous HTTP request.

        See the class docstring for parameter details.
        """

        raise NotImplementedError


__all__ = ["SyncHttpPort", "AsyncHttpPort"]
