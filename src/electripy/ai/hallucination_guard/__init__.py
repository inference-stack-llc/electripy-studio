"""Grounding checks for hallucination-risk reduction."""

from __future__ import annotations

from .domain import GroundingCheckResult
from .services import evaluate_grounding, extract_citation_ids

__all__ = [
    "GroundingCheckResult",
    "extract_citation_ids",
    "evaluate_grounding",
]
