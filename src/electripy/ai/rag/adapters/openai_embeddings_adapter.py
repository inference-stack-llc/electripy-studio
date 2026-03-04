"""OpenAI embeddings adapter for the RAG kit.

Purpose:
  - Implement :class:`EmbeddingPort` using the official OpenAI Python
    SDK while keeping the rest of the RAG kit provider-agnostic.

Guarantees:
  - Only the minimal ``embeddings.create`` surface is used.
  - Provider-specific exceptions are mapped to :class:`EmbeddingError`
    and :class:`EmbeddingTransientError`.

Usage:
  Basic example::

    from electripy.ai.rag.adapters.openai_embeddings_adapter import OpenAiEmbeddingAdapter

    adapter = OpenAiEmbeddingAdapter(model="text-embedding-3-small")
    vectors = adapter.embed_texts(["hello", "world"])
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from ..errors import EmbeddingError, EmbeddingTransientError
from ..ports import EmbeddingPort


def _map_openai_exception(exc: Exception) -> None:
    """Map an OpenAI exception to a domain error and raise it.

    This helper inspects common attributes of OpenAI errors to derive a
    retryable/non-retryable categorisation without leaking the original
    exception type.
    """

    status_code: int | None = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", status_code)

    if status_code is not None and status_code >= 500:
        raise EmbeddingTransientError("Transient OpenAI error") from None

    raise EmbeddingError(str(exc)) from None


class OpenAiEmbeddingAdapter(EmbeddingPort):
    """OpenAI embeddings adapter implementing :class:`EmbeddingPort`.

    Notes:
        - Requires the ``openai`` package to be installed.
        - The client is created lazily at initialisation time. To avoid an
          import-time dependency, the module imports :mod:`openai` using
          :func:`importlib.import_module`.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            self._model = model
            return

        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:  # pragma: no cover - import-time dependency
            raise ImportError(
                "OpenAiEmbeddingAdapter requires the `openai` package. "
                "Install with `pip install openai`.",
            ) from exc

        self._client = openai_module.OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._model = model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI embeddings.

        Args:
            texts: Input texts.

        Returns:
            List of embedding vectors.
        """

        if not texts:
            return []

        try:
            response: Any = self._client.embeddings.create(model=self._model, input=list(texts))
        except Exception as exc:  # noqa: BLE001
            _map_openai_exception(exc)
            raise AssertionError("_map_openai_exception must raise") from exc

        data = getattr(response, "data", [])
        vectors: list[list[float]] = []
        for item in data:
            embedding = getattr(item, "embedding", None)
            if embedding is None:
                raise EmbeddingError("OpenAI response missing embedding field")
            vectors.append([float(v) for v in embedding])
        return vectors

