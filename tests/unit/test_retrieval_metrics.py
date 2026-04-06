from bgrag.eval.retrieval_metrics import compute_retrieval_metrics
from bgrag.types import ChunkRecord, EvalCase, EvalClaimEvidence, SourceFamily


def _chunk(chunk_id: str, canonical_url: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        canonical_url=canonical_url,
        title=chunk_id,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text="text",
    )


def test_compute_retrieval_metrics_scores_expected_urls_and_claim_evidence() -> None:
    case = EvalCase(
        id="T1",
        question="question",
        primary_urls=["https://example.com/primary"],
        supporting_urls=["https://example.com/support"],
        claim_evidence=[
            EvalClaimEvidence(claim="a", evidence_doc_urls=["https://example.com/primary"]),
            EvalClaimEvidence(claim="b", evidence_doc_urls=["https://example.com/missing"]),
        ],
    )

    metrics = compute_retrieval_metrics(
        case,
        [
            _chunk("c1", "https://example.com/primary"),
            _chunk("c2", "https://example.com/support"),
        ],
    )

    assert metrics.primary_url_hit is True
    assert metrics.supporting_url_hit is True
    assert metrics.expected_url_recall == 1.0
    assert metrics.claim_evidence_recall == 0.5

def test_compute_retrieval_metrics_requires_url_anchors() -> None:
    case = EvalCase(
        id="T2",
        question="question",
        expected_doc_prefixes=["buyers_guide__late_offers__abc"],
        supporting_doc_prefixes=["tbs__directive__xyz"],
        claim_evidence=[
            EvalClaimEvidence(claim="a", evidence_doc_prefixes=["buyers_guide__late_offers__abc"]),
            EvalClaimEvidence(claim="b", evidence_doc_prefixes=["missing__prefix"]),
        ],
    )

    metrics = compute_retrieval_metrics(
        case,
        [
            _chunk("buyers_guide__late_offers__abc__section__1", "https://example.com/primary"),
            _chunk("tbs__directive__xyz__section__2", "https://example.com/support"),
        ],
    )

    assert metrics.primary_url_hit is False
    assert metrics.supporting_url_hit is False
    assert metrics.expected_url_recall == 0.0
    assert metrics.claim_evidence_recall == 0.0
