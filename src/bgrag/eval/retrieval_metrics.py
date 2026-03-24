"""Deterministic retrieval-quality metrics for eval cases."""

from __future__ import annotations

from dataclasses import dataclass

from bgrag.types import ChunkRecord, EvalCase


@dataclass
class RetrievalMetricBundle:
    primary_url_hit: bool
    supporting_url_hit: bool
    expected_url_recall: float
    claim_evidence_recall: float
    claim_evidence_annotated: bool


def _normalized_urls(urls: list[str]) -> set[str]:
    return {url.strip().rstrip("/") for url in urls if url.strip()}


def _chunk_urls(chunks: list[ChunkRecord]) -> set[str]:
    return {chunk.canonical_url.strip().rstrip("/") for chunk in chunks if chunk.canonical_url}


def compute_retrieval_metrics(case: EvalCase, chunks: list[ChunkRecord]) -> RetrievalMetricBundle:
    chunk_urls = _chunk_urls(chunks)
    primary_urls = _normalized_urls(case.primary_urls)
    supporting_urls = _normalized_urls(case.supporting_urls)
    expected_urls = primary_urls | supporting_urls

    if expected_urls:
        expected_url_recall = len(chunk_urls & expected_urls) / len(expected_urls)
    else:
        expected_url_recall = 0.0

    claim_hits = 0
    claim_evidence_annotated = bool(case.claim_evidence)
    if case.claim_evidence:
        for item in case.claim_evidence:
            evidence_urls = _normalized_urls(item.evidence_doc_urls)
            if evidence_urls and (chunk_urls & evidence_urls):
                claim_hits += 1
        claim_evidence_recall = claim_hits / len(case.claim_evidence)
    else:
        claim_evidence_recall = 0.0

    return RetrievalMetricBundle(
        primary_url_hit=bool(chunk_urls & primary_urls),
        supporting_url_hit=bool(chunk_urls & supporting_urls) if supporting_urls else False,
        expected_url_recall=expected_url_recall,
        claim_evidence_recall=claim_evidence_recall,
        claim_evidence_annotated=claim_evidence_annotated,
    )
