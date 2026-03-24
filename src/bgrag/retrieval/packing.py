"""Ranked-chunk packing helpers.

These helpers shape a ranked candidate list into a better evidence bundle
without relying on question-family-specific routing logic.
"""

from __future__ import annotations

from collections import Counter
from math import ceil

from bgrag.types import ChunkRecord


def _heading_group_key(chunk: ChunkRecord) -> tuple[str, ...]:
    if chunk.heading_path:
        return tuple(chunk.heading_path)
    if chunk.heading:
        return (chunk.heading,)
    return (chunk.title,)


def _document_group_key(chunk: ChunkRecord) -> str:
    return chunk.doc_id or chunk.canonical_url


def diversify_ranked_chunks(
    chunks: list[ChunkRecord],
    target_k: int,
    *,
    cover_fraction: float = 0.5,
    max_per_document: int = 8,
    max_per_heading: int = 4,
    seed_chunks_per_heading: int = 2,
) -> list[ChunkRecord]:
    """Reorder a ranked list to preserve some breadth before deepening.

    The first pass takes the best chunk from distinct heading groups until the
    configured coverage budget is reached. The second pass fills the remaining
    target slots while enforcing soft caps on per-document and per-heading
    dominance. Any unselected tail is appended in original order so callers can
    still inspect the complete ranked list if needed.
    """

    if target_k <= 0 or not chunks:
        return chunks

    coverage_budget = min(target_k, max(1, min(3, target_k), ceil(target_k * cover_fraction)))
    selected: list[ChunkRecord] = []
    selected_ids: set[str] = set()
    heading_counts: Counter[tuple[str, ...]] = Counter()
    document_counts: Counter[str] = Counter()
    covered_headings: set[tuple[str, ...]] = set()

    def can_take(chunk: ChunkRecord, *, require_new_heading: bool) -> bool:
        heading_key = _heading_group_key(chunk)
        document_key = _document_group_key(chunk)
        if require_new_heading and heading_key in covered_headings:
            return False
        if max_per_document > 0 and document_counts[document_key] >= max_per_document:
            return False
        if max_per_heading > 0 and heading_counts[heading_key] >= max_per_heading:
            return False
        return True

    def take(chunk: ChunkRecord) -> None:
        heading_key = _heading_group_key(chunk)
        document_key = _document_group_key(chunk)
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        heading_counts[heading_key] += 1
        document_counts[document_key] += 1
        covered_headings.add(heading_key)

    for chunk in chunks:
        if len(selected) >= coverage_budget:
            break
        if chunk.chunk_id in selected_ids:
            continue
        if not can_take(chunk, require_new_heading=True):
            continue
        heading_key = _heading_group_key(chunk)
        seeded = 0
        for sibling in chunks:
            if len(selected) >= coverage_budget:
                break
            if sibling.chunk_id in selected_ids:
                continue
            if _heading_group_key(sibling) != heading_key:
                continue
            if not can_take(sibling, require_new_heading=False):
                continue
            take(sibling)
            seeded += 1
            if seeded >= seed_chunks_per_heading:
                break

    for chunk in chunks:
        if len(selected) >= target_k:
            break
        if chunk.chunk_id in selected_ids:
            continue
        if can_take(chunk, require_new_heading=False):
            take(chunk)

    for chunk in chunks:
        if len(selected) >= target_k:
            break
        if chunk.chunk_id in selected_ids:
            continue
        take(chunk)

    remaining = [chunk for chunk in chunks if chunk.chunk_id not in selected_ids]
    return selected + remaining
