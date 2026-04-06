from bgrag.retrieval_benchmark import (
    RetrievalBenchmarkCaseResult,
    _compute_overall_metrics,
    _first_expected_rank,
)
from bgrag.types import ChunkRecord, EvalCase, SourceFamily


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


def test_first_expected_rank_uses_urls() -> None:
    case = EvalCase(id="T1", question="q", primary_urls=["https://example.com/match"])
    chunks = [
        _chunk("c1", "https://example.com/a"),
        _chunk("c2", "https://example.com/match"),
    ]
    assert _first_expected_rank(case, chunks) == 2

def test_first_expected_rank_requires_url_match() -> None:
    case = EvalCase(id="T2", question="q", expected_doc_prefixes=["buyers_guide__foo"])
    chunks = [
        _chunk("buyers_guide__bar__section__1", "https://example.com/a"),
        _chunk("buyers_guide__foo__section__2", "https://example.com/b"),
    ]
    assert _first_expected_rank(case, chunks) is None


def test_compute_overall_metrics_reports_rank_means_and_mrr() -> None:
    case_results = [
        RetrievalBenchmarkCaseResult(
            case_id="A",
            question="q1",
            retrieval_queries=["q1"],
            query_planning_seconds=0.1,
            query_embedding_seconds=0.2,
            retrieval_seconds=0.3,
            lexical_search_seconds=0.1,
            vector_search_seconds=0.05,
            candidate_fusion_seconds=0.02,
            rerank_seconds=0.01,
            packing_seconds=0.01,
            total_case_seconds=0.6,
            raw_shortlist_primary_url_hit=True,
            selected_primary_url_hit=True,
            packed_primary_url_hit=True,
            raw_shortlist_expected_url_recall=1.0,
            selected_expected_url_recall=1.0,
            packed_expected_url_recall=1.0,
            raw_shortlist_claim_evidence_recall=1.0,
            selected_claim_evidence_recall=1.0,
            packed_claim_evidence_recall=1.0,
            raw_shortlist_chunk_support_recall=0.0,
            selected_chunk_support_recall=0.0,
            packed_chunk_support_recall=0.0,
            claim_evidence_annotated=True,
            claim_chunk_support_annotated=False,
            first_expected_raw_shortlist_rank=1,
            first_expected_selected_rank=2,
            first_expected_packed_rank=2,
            top_raw_shortlist=[],
            top_selected=[],
            top_packed=[],
        ),
        RetrievalBenchmarkCaseResult(
            case_id="B",
            question="q2",
            retrieval_queries=["q2"],
            query_planning_seconds=0.2,
            query_embedding_seconds=0.3,
            retrieval_seconds=0.4,
            lexical_search_seconds=0.1,
            vector_search_seconds=0.06,
            candidate_fusion_seconds=0.03,
            rerank_seconds=0.01,
            packing_seconds=0.02,
            total_case_seconds=0.9,
            raw_shortlist_primary_url_hit=False,
            selected_primary_url_hit=False,
            packed_primary_url_hit=False,
            raw_shortlist_expected_url_recall=0.0,
            selected_expected_url_recall=0.0,
            packed_expected_url_recall=0.0,
            raw_shortlist_claim_evidence_recall=0.0,
            selected_claim_evidence_recall=0.0,
            packed_claim_evidence_recall=0.0,
            raw_shortlist_chunk_support_recall=0.0,
            selected_chunk_support_recall=0.0,
            packed_chunk_support_recall=0.0,
            claim_evidence_annotated=False,
            claim_chunk_support_annotated=False,
            first_expected_raw_shortlist_rank=None,
            first_expected_selected_rank=None,
            first_expected_packed_rank=None,
            top_raw_shortlist=[],
            top_selected=[],
            top_packed=[],
        ),
    ]

    metrics = _compute_overall_metrics(case_results)

    assert metrics["case_count"] == 2
    assert metrics["raw_shortlist_first_expected_rank_mean_hit_only"] == 1
    assert metrics["selected_first_expected_rank_mean_hit_only"] == 2
    assert metrics["packed_first_expected_rank_mean_hit_only"] == 2
    assert metrics["raw_shortlist_mrr"] == 1.0
    assert metrics["selected_mrr"] == 0.5
    assert metrics["packed_mrr"] == 0.5
    assert metrics["raw_shortlist_miss_count"] == 1
    assert metrics["selected_miss_count"] == 1
    assert metrics["packed_miss_count"] == 1
    assert metrics["claim_evidence_annotated_case_count"] == 1
