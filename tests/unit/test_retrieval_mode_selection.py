from bgrag.retrieval.mode_selection import (
    RetrievalModeDecision,
    _summarize_evidence,
    normalize_retrieval_mode_decision,
)
from bgrag.types import ChunkRecord, EvidenceBundle, SourceFamily


def _chunk(chunk_id: str, *, order: int, title: str, heading: str, url: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("__")[0],
        canonical_url=url,
        title=title,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text="Example procurement content.",
        heading=heading,
        heading_path=[title, heading],
        order=order,
    )


def test_normalize_retrieval_mode_decision_accepts_valid_mode() -> None:
    decision = normalize_retrieval_mode_decision(
        '{"mode":"page_family_expansion","rationale":"Need sibling page context."}'
    )
    assert decision == RetrievalModeDecision(
        mode="page_family_expansion",
        rationale="Need sibling page context.",
    )


def test_normalize_retrieval_mode_decision_falls_back_to_baseline() -> None:
    decision = normalize_retrieval_mode_decision('{"mode":"something_else","rationale":"x"}')
    assert decision.mode == "baseline"


def test_summarize_evidence_renders_chunk_metadata() -> None:
    evidence = EvidenceBundle(
        query="test question",
        packed_chunks=[
            _chunk(
                "doc1__section__7",
                order=7,
                title="Receive offers - Verify offers | CanadaBuys",
                heading="Exceptions list",
                url="https://example.com/verify-offers",
            )
        ],
    )
    rendered = _summarize_evidence(evidence, max_chunks=5)
    assert "chunk_id=doc1__section__7" in rendered
    assert "order=7" in rendered
    assert "Verify offers" in rendered
    assert "https://example.com/verify-offers" in rendered
