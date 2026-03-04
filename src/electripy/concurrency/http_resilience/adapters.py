"""Adapters for concrete HTTP client libraries.

This module provides httpx-based adapters that implement the HTTP ports
defined in :mod:`electripy.concurrency.http_resilience.ports`.

The rest of the system only depends on the Protocol ports and the
:class:`ResponseData` model. No other part of the code should import
``httpx`` directly.

Example:
    from electripy.concurrency.http_resilience.adapters import HttpxSyncAdapter

    adapter = HttpxSyncAdapter(base_url="https://api.example.com")
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from electripy.core.logging import get_logger

from .domain import BodyType, Headers, MutableHeaders, QueryParams, ResponseData, TransientHttpError
from .ports import AsyncHttpPort, SyncHttpPort

logger = get_logger(__name__)


def _normalize_headers(headers: Mapping[str, str] | None) -> MutableHeaders:
    """Normalize headers into a lowercase-keyed mutable mapping.

    Args:
        headers: Optional original headers mapping.

    Returns:
        MutableHeaders: Normalized header mapping.
    """

    normalized: MutableHeaders = {}
    if headers:
        for key, value in headers.items():
            normalized[key.lower()] = value
    return normalized


@dataclass(slots=True)
class HttpxSyncAdapter(SyncHttpPort):
    """httpx-based synchronous HTTP adapter.

    This adapter encapsulates an httpx synchronous client and exposes the
    minimal :class:`SyncHttpPort` surface. It ensures that all responses are
    normalized into :class:`ResponseData` and that low-level httpx exceptions
    are mapped into ElectriPy domain exceptions.

    Args:
        base_url: Optional base URL used for relative request URLs.
        default_timeout_s: Default timeout in seconds when callers do not
            provide a per-request timeout.
        client: Optional externally-managed httpx client instance.

    Example:
        adapter = HttpxSyncAdapter(base_url="https://api.example.com")
        response = adapter.request("GET", "/health")
    """

    base_url: str | None = None
    default_timeout_s: float = 5.0
    client: httpx.Client | None = field(default=None, repr=False)

    def _get_client(self) -> httpx.Client:
        """Return the underlying httpx client, creating one if needed."""

        if self.client is None:
            if self.base_url is not None:
                self.client = httpx.Client(base_url=self.base_url)
            else:
                self.client = httpx.Client()
        return self.client

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
        """Perform a synchronous HTTP request using httpx.

        See :class:`SyncHttpPort` for parameter documentation.
        """

        client = self._get_client()
        timeout = timeout_s or self.default_timeout_s
        full_url = urljoin(self.base_url, url) if self.base_url else url

        try:
            response = client.request(
                method=method,
                url=full_url,
                headers=headers,
                params=params,
                json=json,
                data=data,  # type: ignore[reportGeneralTypeIssues]
                timeout=timeout,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:  # type: ignore[attr-defined]
            logger.warning("httpx transient error", extra={"url": full_url, "error": str(exc)})
            raise TransientHttpError("Transient HTTP error") from exc
        except httpx.HTTPError as exc:  # type: ignore[catching-anything]
            logger.error("httpx HTTP error", extra={"url": full_url, "error": str(exc)})
            raise TransientHttpError("HTTP client error") from exc

        headers_normalized = _normalize_headers(response.headers)
        return ResponseData(
            status_code=response.status_code,
            headers=headers_normalized,
            body=response.content,
            url=str(response.url),
            elapsed_s=response.elapsed.total_seconds() if response.elapsed else None,
        )


@dataclass(slots=True)
class HttpxAsyncAdapter(AsyncHttpPort):
    """httpx-based asynchronous HTTP adapter.

    This adapter encapsulates an httpx asynchronous client and exposes the
    minimal :class:`AsyncHttpPort` surface.

    Args:
        base_url: Optional base URL used for relative request URLs.
        default_timeout_s: Default timeout in seconds when callers do not
            provide a per-request timeout.
        client: Optional externally-managed httpx async client instance.

    Example:
        adapter = HttpxAsyncAdapter(base_url="https://api.example.com")
        response = await adapter.request("GET", "/health")
    """

    base_url: str | None = None
    default_timeout_s: float = 5.0
    client: httpx.AsyncClient | None = field(default=None, repr=False)

    def _get_client(self) -> httpx.AsyncClient:
        """Return the underlying httpx async client, creating one if needed."""

        if self.client is None:
            if self.base_url is not None:
                self.client = httpx.AsyncClient(base_url=self.base_url)
            else:
                self.client = httpx.AsyncClient()
        return self.client

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
        """Perform an asynchronous HTTP request using httpx.

        See :class:`AsyncHttpPort` for parameter documentation.
        """

        client = self._get_client()
        timeout = timeout_s or self.default_timeout_s
        full_url = urljoin(self.base_url, url) if self.base_url else url

        try:
            response = await client.request(
                method=method,
                url=full_url,
                headers=headers,
                params=params,
                json=json,
                data=data,  # type: ignore[reportGeneralTypeIssues]
                timeout=timeout,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:  # type: ignore[attr-defined]
            logger.warning("httpx transient error", extra={"url": full_url, "error": str(exc)})
            raise TransientHttpError("Transient HTTP error") from exc
        except httpx.HTTPError as exc:  # type: ignore[catching-anything]
            logger.error("httpx HTTP error", extra={"url": full_url, "error": str(exc)})
            raise TransientHttpError("HTTP client error") from exc

        headers_normalized = _normalize_headers(response.headers)
        return ResponseData(
            status_code=response.status_code,
            headers=headers_normalized,
            body=response.content,
            url=str(response.url),
            elapsed_s=response.elapsed.total_seconds() if response.elapsed else None,
        )


__all__ = ["HttpxSyncAdapter", "HttpxAsyncAdapter"]
