"""Configuration objects for the RAG indexing and retrieval kit.

Purpose:
  - Provide explicit, typed configuration for chunking and embedding
    behavior.

Guarantees:
  - All configuration is expressed as dataclasses with validation in
    ``__post_init__``.

Usage:
  Basic example::

    from electripy.ai.rag import ChunkingConfig

    config = ChunkingConfig(chunk_size_chars=1000, overlap_chars=200)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration for deterministic character-based chunking.

    Attributes:
        chunk_size_chars: Target number of characters per chunk.
        overlap_chars: Number of characters to overlap between adjacent
            chunks. Must be less than ``chunk_size_chars``.
    """

    chunk_size_chars: int = 1000
    overlap_chars: int = 200

    def __post_init__(self) -> None:
        if self.chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be positive")
        if not 0 <= self.overlap_chars < self.chunk_size_chars:
            raise ValueError("overlap_chars must be in [0, chunk_size_chars)")


@dataclass(slots=True)
class EmbeddingGatewaySettings:
    """Configuration for the embedding gateway.

    Attributes:
        max_batch_size: Maximum number of texts per embedding request.
        max_retries: Maximum number of attempts for transient errors.
        base_delay_s: Base delay before the first retry.
        max_delay_s: Maximum delay between retries.
        jitter_ratio: Jitter ratio in the range [0.0, 1.0].
    """

    max_batch_size: int = 64
    max_retries: int = 2
    base_delay_s: float = 0.1
    max_delay_s: float = 2.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay_s < 0:
            raise ValueError("base_delay_s must be >= 0")
        if self.max_delay_s <= 0:
            raise ValueError("max_delay_s must be > 0")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio must be between 0.0 and 1.0")

