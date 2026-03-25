from __future__ import annotations

from electripy.ai.hallucination_guard import evaluate_grounding, extract_citation_ids


def test_extract_citation_ids() -> None:
    text = "Answer from [cite:chunk-1] and [cite:doc-2]."
    assert extract_citation_ids(text) == ["chunk-1", "doc-2"]


def test_grounding_flags_missing_citation() -> None:
    result = evaluate_grounding(
        response_text="Paris is in France",
        evidence_texts=["Paris is the capital of France"],
        min_overlap=0.1,
    )

    assert result.grounded is False
    assert "missing_citations" in result.reasons
