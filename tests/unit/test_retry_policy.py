from __future__ import annotations

from bgrag.serving.retry_policy import assess_question_risk, decide_hybrid_retry
from bgrag.types import ChunkRecord, EvidenceBundle, RetrievalAssessment, RetrievalCandidate, SourceFamily


def _chunk(chunk_id: str, *, doc_id: str | None = None, heading: str | None = None) -> ChunkRecord:
    resolved_doc_id = doc_id or chunk_id
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=resolved_doc_id,
        canonical_url=f"https://example.com/{resolved_doc_id}",
        title=resolved_doc_id,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text=f"Example evidence for {chunk_id}.",
        heading=heading,
    )


def test_assess_question_risk_flags_exactness_and_branch_sensitivity() -> None:
    result = assess_question_risk(
        "If the procurement is late, what happens, and compare the approval authority for the exception form number?"
    )

    assert result.risk_level == "high"
    assert result.exactness_sensitive is True
    assert result.branch_sensitive is True
    assert result.reasons


def test_decide_hybrid_retry_retries_on_bundle_weakness_plus_high_coverage_risk() -> None:
    raw_shortlist = [
        RetrievalCandidate(chunk=_chunk("c1", doc_id="doc_a", heading="Intro")),
        RetrievalCandidate(chunk=_chunk("c2", doc_id="doc_b", heading="Rule A")),
        RetrievalCandidate(chunk=_chunk("c3", doc_id="doc_c", heading="Rule B")),
        RetrievalCandidate(chunk=_chunk("c4", doc_id="doc_d", heading="Rule C")),
        RetrievalCandidate(chunk=_chunk("c5", doc_id="doc_e", heading="Rule D")),
        RetrievalCandidate(chunk=_chunk("c6", doc_id="doc_f", heading="Rule E")),
        RetrievalCandidate(chunk=_chunk("c7", doc_id="doc_g", heading="Rule F")),
        RetrievalCandidate(chunk=_chunk("c8", doc_id="doc_h", heading="Rule G")),
    ]
    evidence = EvidenceBundle(
        query="workflow question",
        raw_shortlist=raw_shortlist,
        packed_chunks=[
            _chunk("packed1", doc_id="doc_a", heading="Intro"),
            _chunk("packed2", doc_id="doc_a", heading="Intro"),
        ],
    )
    assessment = RetrievalAssessment(
        sufficient_for_answer=False,
        coverage_risk="high",
        exactness_risk="low",
        support_conflict=False,
        recommended_next_step="answer",
    )

    decision = decide_hybrid_retry(
        question="What happens if a workflow branch is missing?",
        evidence=evidence,
        retrieval_assessment=assessment,
        enable_official_site_escalation=False,
    )

    assert decision.recommended_next_step == "retry_retrieve"
    assert decision.bundle_risk.risk_level == "high"
    assert decision.bundle_risk.retry_signal is True
    assert decision.reasons


def test_decide_hybrid_retry_downgrades_browse_request_to_retry_when_browse_disabled() -> None:
    evidence = EvidenceBundle(
        query="contact question",
        raw_shortlist=[RetrievalCandidate(chunk=_chunk("c1"))],
        packed_chunks=[_chunk("c1")],
    )
    assessment = RetrievalAssessment(
        sufficient_for_answer=False,
        coverage_risk="medium",
        exactness_risk="high",
        support_conflict=False,
        recommended_next_step="browse_official",
    )

    decision = decide_hybrid_retry(
        question="What is the contact email?",
        evidence=evidence,
        retrieval_assessment=assessment,
        enable_official_site_escalation=False,
    )

    assert decision.recommended_next_step == "retry_retrieve"
    assert any("downgrade" in reason for reason in decision.reasons)
