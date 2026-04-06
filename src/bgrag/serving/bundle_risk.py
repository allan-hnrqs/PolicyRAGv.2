"""Deterministic evidence-bundle weakness scoring."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from bgrag.types import EvidenceBundle


class BundleRiskAssessment(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    retry_signal: bool
    packed_chunk_count: int
    raw_shortlist_count: int
    packed_document_count: int
    raw_document_count: int
    packed_heading_count: int
    raw_heading_count: int
    packed_source_family_count: int
    raw_source_family_count: int
    top_document_share: float
    top_heading_share: float
    lost_document_breadth: int
    lost_heading_breadth: int
    lost_source_family_breadth: int
    reasons: list[str] = Field(default_factory=list)


def _heading_key_for_chunk(chunk) -> tuple[str, ...]:
    if getattr(chunk, "heading_path", None):
        return tuple(chunk.heading_path)
    if getattr(chunk, "heading", None):
        return (str(chunk.heading),)
    return (str(chunk.title),)


def assess_bundle_risk(evidence: EvidenceBundle) -> BundleRiskAssessment:
    raw_candidates = evidence.raw_shortlist or evidence.selected_candidates or evidence.candidates
    packed_chunks = evidence.packed_chunks

    raw_chunks = [candidate.chunk for candidate in raw_candidates]
    packed_chunk_count = len(packed_chunks)
    raw_shortlist_count = len(raw_candidates)

    packed_documents = Counter(chunk.doc_id or chunk.canonical_url for chunk in packed_chunks)
    raw_documents = Counter(chunk.doc_id or chunk.canonical_url for chunk in raw_chunks)
    packed_headings = Counter(_heading_key_for_chunk(chunk) for chunk in packed_chunks)
    raw_headings = Counter(_heading_key_for_chunk(chunk) for chunk in raw_chunks)
    packed_source_families = Counter(chunk.source_family.value for chunk in packed_chunks)
    raw_source_families = Counter(chunk.source_family.value for chunk in raw_chunks)

    packed_document_count = len(packed_documents)
    raw_document_count = len(raw_documents)
    packed_heading_count = len(packed_headings)
    raw_heading_count = len(raw_headings)
    packed_source_family_count = len(packed_source_families)
    raw_source_family_count = len(raw_source_families)

    top_document_share = (
        max(packed_documents.values()) / packed_chunk_count if packed_chunk_count and packed_documents else 0.0
    )
    top_heading_share = (
        max(packed_headings.values()) / packed_chunk_count if packed_chunk_count and packed_headings else 0.0
    )

    lost_document_breadth = max(0, raw_document_count - packed_document_count)
    lost_heading_breadth = max(0, raw_heading_count - packed_heading_count)
    lost_source_family_breadth = max(0, raw_source_family_count - packed_source_family_count)

    reasons: list[str] = []
    risk_points = 0

    if packed_chunk_count <= 2 and raw_shortlist_count >= 4:
        risk_points += 1
        reasons.append("packed bundle collapsed to very few chunks")
    if packed_chunk_count <= 4 and raw_shortlist_count >= 8:
        risk_points += 1
        reasons.append("packed bundle is short relative to the shortlist")
    if packed_chunk_count >= 6 and top_document_share >= 0.75:
        risk_points += 2
        reasons.append("packed bundle is dominated by one document")
    if packed_chunk_count >= 6 and top_heading_share >= 0.5:
        risk_points += 1
        reasons.append("packed bundle is dominated by one heading")
    if packed_document_count <= 2 and lost_document_breadth >= 3:
        risk_points += 1
        reasons.append("packing dropped substantial document breadth from the shortlist")
    if raw_source_family_count > packed_source_family_count and packed_source_family_count <= 1:
        risk_points += 1
        reasons.append("packing collapsed cross-source coverage")
    if packed_heading_count <= 2 and lost_heading_breadth >= 4:
        risk_points += 1
        reasons.append("packing collapsed heading breadth")

    if risk_points >= 3:
        risk_level: Literal["low", "medium", "high"] = "high"
    elif risk_points >= 1:
        risk_level = "medium"
    else:
        risk_level = "low"

    retry_signal = risk_level != "low" and raw_shortlist_count > packed_chunk_count

    return BundleRiskAssessment(
        risk_level=risk_level,
        retry_signal=retry_signal,
        packed_chunk_count=packed_chunk_count,
        raw_shortlist_count=raw_shortlist_count,
        packed_document_count=packed_document_count,
        raw_document_count=raw_document_count,
        packed_heading_count=packed_heading_count,
        raw_heading_count=raw_heading_count,
        packed_source_family_count=packed_source_family_count,
        raw_source_family_count=raw_source_family_count,
        top_document_share=top_document_share,
        top_heading_share=top_heading_share,
        lost_document_breadth=lost_document_breadth,
        lost_heading_breadth=lost_heading_breadth,
        lost_source_family_breadth=lost_source_family_breadth,
        reasons=reasons,
    )
