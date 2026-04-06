from pathlib import Path

from bgrag.serving.bundle_risk import assess_bundle_risk
from bgrag.types import ChunkRecord, EvidenceBundle, RetrievalCandidate, SourceFamily

REPO_ROOT = Path(__file__).resolve().parents[2]


def _chunk(
    chunk_id: str,
    *,
    doc_id: str,
    heading_path: list[str],
    source_family: SourceFamily = SourceFamily.BUYERS_GUIDE,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        canonical_url=f"https://example.com/{doc_id}",
        title=doc_id,
        source_family=source_family,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text=f"text for {chunk_id}",
        heading_path=heading_path,
    )


def _candidate(chunk: ChunkRecord, score: float = 1.0) -> RetrievalCandidate:
    return RetrievalCandidate(chunk=chunk, blended_score=score, rerank_score=score)


def test_assess_bundle_risk_is_low_for_balanced_bundle() -> None:
    raw = [
        _candidate(_chunk("c1", doc_id="d1", heading_path=["A"])),
        _candidate(_chunk("c2", doc_id="d2", heading_path=["B"])),
        _candidate(_chunk("c3", doc_id="d3", heading_path=["C"], source_family=SourceFamily.BUY_CANADIAN_POLICY)),
        _candidate(_chunk("c4", doc_id="d4", heading_path=["D"], source_family=SourceFamily.TBS_DIRECTIVE)),
    ]
    evidence = EvidenceBundle(
        query="q",
        raw_shortlist=raw,
        selected_candidates=raw,
        packed_chunks=[candidate.chunk for candidate in raw],
    )

    risk = assess_bundle_risk(evidence)

    assert risk.risk_level == "low"
    assert risk.retry_signal is False
    assert risk.reasons == []


def test_assess_bundle_risk_flags_monopoly_and_breadth_loss() -> None:
    raw_chunks = [
        _chunk("c1", doc_id="d1", heading_path=["A"]),
        _chunk("c2", doc_id="d1", heading_path=["A", "sub"]),
        _chunk("c3", doc_id="d2", heading_path=["B"]),
        _chunk("c4", doc_id="d3", heading_path=["C"], source_family=SourceFamily.BUY_CANADIAN_POLICY),
        _chunk("c5", doc_id="d4", heading_path=["D"], source_family=SourceFamily.TBS_DIRECTIVE),
        _chunk("c6", doc_id="d5", heading_path=["E"]),
        _chunk("c7", doc_id="d6", heading_path=["F"]),
        _chunk("c8", doc_id="d7", heading_path=["G"]),
    ]
    packed = [
        raw_chunks[0],
        raw_chunks[1],
        _chunk("p3", doc_id="d1", heading_path=["A"]),
        _chunk("p4", doc_id="d1", heading_path=["A"]),
        _chunk("p5", doc_id="d1", heading_path=["A"]),
        _chunk("p6", doc_id="d2", heading_path=["B"]),
    ]
    evidence = EvidenceBundle(
        query="q",
        raw_shortlist=[_candidate(chunk) for chunk in raw_chunks],
        selected_candidates=[_candidate(chunk) for chunk in packed],
        packed_chunks=packed,
    )

    risk = assess_bundle_risk(evidence)

    assert risk.risk_level == "high"
    assert risk.retry_signal is True
    assert "packed bundle is dominated by one document" in risk.reasons
    assert risk.lost_document_breadth >= 3


def test_assess_bundle_risk_uses_selected_candidates_when_raw_shortlist_missing() -> None:
    packed = [
        _chunk("c1", doc_id="d1", heading_path=["A"]),
        _chunk("c2", doc_id="d1", heading_path=["A", "sub"]),
        _chunk("c3", doc_id="d2", heading_path=["B"]),
        _chunk("c4", doc_id="d3", heading_path=["C"]),
    ]
    evidence = EvidenceBundle(
        query="q",
        selected_candidates=[_candidate(chunk) for chunk in packed],
        packed_chunks=packed[:2],
    )

    risk = assess_bundle_risk(evidence)

    assert risk.raw_shortlist_count == 4
    assert risk.packed_chunk_count == 2
    assert risk.risk_level in {"medium", "high"}
