from __future__ import annotations

"""ElectriPy AI namespace.

Purpose:
  - Group AI-related components for ElectriPy Studio.

Guarantees:
  - Public APIs for each AI submodule are re-exported from their own packages.

Usage:
  Basic example::

    from electripy.ai.llm_gateway import LlmGatewaySyncClient
"""

__all__ = [
  "llm_gateway",
  "rag",
]
