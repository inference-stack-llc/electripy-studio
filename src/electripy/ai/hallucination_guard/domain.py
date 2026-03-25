"""Domain models for grounding and hallucination checks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GroundingCheckResult:
    """Result of grounding validation for one generated response."""

    grounded: bool
    overlap_score: float
    citation_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
