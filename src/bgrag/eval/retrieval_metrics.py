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


def _chunk_doc_ids(chunks: list[ChunkRecord]) -> set[str]:
    return {chunk.doc_id.strip() for chunk in chunks if chunk.doc_id}


def _prefix_hit(prefixes: list[str], chunk_doc_ids: set[str]) -> bool:
    normalized_prefixes = [prefix.strip() for prefix in prefixes if prefix.strip()]
    return any(doc_id.startswith(prefix) for prefix in normalized_prefixes for doc_id in chunk_doc_ids)


def compute_retrieval_metrics(case: EvalCase, chunks: list[ChunkRecord]) -> RetrievalMetricBundle:
    chunk_urls = _chunk_urls(chunks)
    chunk_doc_ids = _chunk_doc_ids(chunks)
    primary_urls = _normalized_urls(case.primary_urls)
    supporting_urls = _normalized_urls(case.supporting_urls)
    primary_prefix_hit = _prefix_hit(case.expected_doc_prefixes, chunk_doc_ids) if not primary_urls else False
    supporting_prefix_hit = _prefix_hit(case.supporting_doc_prefixes, chunk_doc_ids) if not supporting_urls else False
    expected_anchor_total = len(primary_urls | supporting_urls)
    expected_anchor_hits = len(chunk_urls & (primary_urls | supporting_urls))
    if not primary_urls and case.expected_doc_prefixes:
        expected_anchor_total += 1
        if primary_prefix_hit:
            expected_anchor_hits += 1
    if not supporting_urls and case.supporting_doc_prefixes:
        expected_anchor_total += 1
        if supporting_prefix_hit:
            expected_anchor_hits += 1

    if expected_anchor_total:
        expected_url_recall = expected_anchor_hits / expected_anchor_total
    else:
        expected_url_recall = 0.0

    claim_hits = 0
    claim_evidence_annotated = bool(case.claim_evidence)
    if case.claim_evidence:
        for item in case.claim_evidence:
            evidence_urls = _normalized_urls(item.evidence_doc_urls)
            if evidence_urls and (chunk_urls & evidence_urls):
                claim_hits += 1
            elif not evidence_urls and _prefix_hit(item.evidence_doc_prefixes, chunk_doc_ids):
                claim_hits += 1
        claim_evidence_recall = claim_hits / len(case.claim_evidence)
    else:
        claim_evidence_recall = 0.0

    return RetrievalMetricBundle(
        primary_url_hit=bool(chunk_urls & primary_urls) or primary_prefix_hit,
        supporting_url_hit=(bool(chunk_urls & supporting_urls) or supporting_prefix_hit) if (supporting_urls or case.supporting_doc_prefixes) else False,
        expected_url_recall=expected_url_recall,
        claim_evidence_recall=claim_evidence_recall,
        claim_evidence_annotated=claim_evidence_annotated,
    )
