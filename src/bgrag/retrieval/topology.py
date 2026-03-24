"""Source topology policies for Buyer’s Guide-first retrieval."""

from __future__ import annotations

from bgrag.registry import source_topology_registry
from bgrag.types import ChunkRecord


EXPLICIT_SUPPORT_MARKERS = (
    "buy canadian",
    "tbs",
    "treasury board",
    "directive on the management of procurement",
    "canadian content",
    "trade agreement",
    "reciprocal procurement",
)

PRIMARY_EVIDENCE_SUPPORT_MARKERS = (
    "buy canadian policy",
    "directive on the management of procurement",
    "treasury board",
    "tbs",
)
DEFAULT_SUPPORT_SHARE = 0.25


def _needs_supporting_sources(question: str, primary_chunks: list[ChunkRecord]) -> bool:
    normalized = question.lower()
    if any(marker in normalized for marker in EXPLICIT_SUPPORT_MARKERS):
        return True
    for chunk in primary_chunks:
        text = chunk.text.lower()
        if any(marker in text for marker in PRIMARY_EVIDENCE_SUPPORT_MARKERS):
            return True
    return False


def _support_slot_budget(top_k: int, available_support: int) -> int:
    if top_k <= 0 or available_support <= 0:
        return 0
    return min(available_support, max(1, round(top_k * DEFAULT_SUPPORT_SHARE)))


def bg_primary_support_fallback(question: str, grouped: dict[str, list[ChunkRecord]], top_k: int) -> list[ChunkRecord]:
    primary_pool = grouped.get("buyers_guide", [])
    support_pool: list[ChunkRecord] = []
    support_pool.extend(grouped.get("buy_canadian_policy", []))
    support_pool.extend(grouped.get("tbs_directive", []))

    if not _needs_supporting_sources(question, primary_pool[:top_k]):
        return primary_pool[:top_k]

    support_slots = _support_slot_budget(top_k, len(support_pool))
    primary_slots = max(0, top_k - support_slots)

    primary = primary_pool[:primary_slots]
    support = support_pool[:support_slots]
    combined = primary + support

    if len(combined) < top_k:
        remaining = top_k - len(combined)
        combined.extend(primary_pool[len(primary) : len(primary) + remaining])
    if len(combined) < top_k:
        remaining = top_k - len(combined)
        combined.extend(support_pool[len(support) : len(support) + remaining])

    return combined[:top_k]


def unified_source_hybrid(question: str, grouped: dict[str, list[ChunkRecord]], top_k: int) -> list[ChunkRecord]:
    del question
    merged: list[ChunkRecord] = []
    for family in ("buyers_guide", "buy_canadian_policy", "tbs_directive"):
        merged.extend(grouped.get(family, []))
    return merged[:top_k]


source_topology_registry.register("bg_primary_support_fallback", bg_primary_support_fallback)
source_topology_registry.register("unified_source_hybrid", unified_source_hybrid)
