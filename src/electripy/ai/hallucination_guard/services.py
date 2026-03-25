"""Grounding heuristics for reducing hallucination risk."""

from __future__ import annotations

import re

from .domain import GroundingCheckResult

_CITATION_PATTERN = re.compile(r"\[cite:([a-zA-Z0-9_\-:.]+)\]")
_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def extract_citation_ids(text: str) -> list[str]:
    """Extract stable citation ids from response text.

    Expected marker format: [cite:chunk-123]
    """

    return _CITATION_PATTERN.findall(text)


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in _WORD_PATTERN.findall(text)}


def evaluate_grounding(
    *,
    response_text: str,
    evidence_texts: list[str],
    min_overlap: float = 0.2,
) -> GroundingCheckResult:
    """Check if a response appears grounded in retrieved evidence.

    The score is a token-overlap ratio between response tokens and evidence tokens.
    """

    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be in [0, 1]")

    citations = extract_citation_ids(response_text)
    response_tokens = _token_set(response_text)
    evidence_tokens: set[str] = set()
    for evidence in evidence_texts:
        evidence_tokens.update(_token_set(evidence))

    if not response_tokens:
        return GroundingCheckResult(
            grounded=False,
            overlap_score=0.0,
            citation_ids=citations,
            reasons=["response_has_no_tokens"],
        )

    overlap = len(response_tokens & evidence_tokens) / float(len(response_tokens))
    reasons: list[str] = []
    if not citations:
        reasons.append("missing_citations")
    if overlap < min_overlap:
        reasons.append("low_evidence_overlap")

    return GroundingCheckResult(
        grounded=(not reasons),
        overlap_score=overlap,
        citation_ids=citations,
        reasons=reasons,
    )
