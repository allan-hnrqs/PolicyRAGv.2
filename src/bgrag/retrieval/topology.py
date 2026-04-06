"""Source topology policies for Buyer’s Guide-first retrieval."""

from __future__ import annotations

from bgrag.registry import source_topology_registry
from bgrag.types import ChunkRecord

_AUTHORITY_SUPPORT_KEYWORDS = (
    "treasury board",
    "directive",
    "policy framework",
    "buy canadian",
    "canadian content",
    "reciprocal",
    "foreign supplier",
    "contracting authorit",
    "governance",
    "ratification",
)
_AUTHORITY_SUPPORT_CLUSTER_LIMIT = 5


def _requires_authority_support(question: str) -> bool:
    normalized = question.lower()
    return any(keyword in normalized for keyword in _AUTHORITY_SUPPORT_KEYWORDS)


def _leading_doc_cluster(pool: list[ChunkRecord], max_chunks: int) -> list[ChunkRecord]:
    if not pool or max_chunks <= 0:
        return []
    lead_doc_id = pool[0].doc_id
    cluster: list[ChunkRecord] = []
    for chunk in pool:
        if chunk.doc_id != lead_doc_id:
            continue
        cluster.append(chunk)
        if len(cluster) >= max_chunks:
            break
    return cluster


def bg_primary_support_fallback(question: str, grouped: dict[str, list[ChunkRecord]], top_k: int) -> list[ChunkRecord]:
    del question
    primary_pool = grouped.get("buyers_guide", [])
    support_pool: list[ChunkRecord] = []
    support_pool.extend(grouped.get("buy_canadian_policy", []))
    support_pool.extend(grouped.get("tbs_directive", []))

    combined = primary_pool[:top_k]
    if len(combined) >= top_k:
        return combined

    remaining = top_k - len(combined)
    combined.extend(support_pool[:remaining])
    return combined[:top_k]


def bg_primary_authority_reserve(question: str, grouped: dict[str, list[ChunkRecord]], top_k: int) -> list[ChunkRecord]:
    del question
    primary_pool = grouped.get("buyers_guide", [])
    tbs_pool = grouped.get("tbs_directive", [])
    policy_pool = grouped.get("buy_canadian_policy", [])

    reserved_support: list[ChunkRecord] = []
    if tbs_pool:
        reserved_support.append(tbs_pool[0])
    elif policy_pool:
        reserved_support.append(policy_pool[0])

    primary_budget = max(0, top_k - len(reserved_support))
    combined = list(primary_pool[:primary_budget])
    selected_ids = {chunk.chunk_id for chunk in combined}

    for chunk in reserved_support:
        if chunk.chunk_id not in selected_ids and len(combined) < top_k:
            combined.append(chunk)
            selected_ids.add(chunk.chunk_id)

    support_pools: tuple[list[ChunkRecord], ...] = (tbs_pool, policy_pool)
    for pool in support_pools:
        for chunk in pool:
            if len(combined) >= top_k:
                return combined[:top_k]
            if chunk.chunk_id in selected_ids:
                continue
            combined.append(chunk)
            selected_ids.add(chunk.chunk_id)

    return combined[:top_k]


def bg_primary_selective_authority_reserve(question: str, grouped: dict[str, list[ChunkRecord]], top_k: int) -> list[ChunkRecord]:
    if not _requires_authority_support(question):
        return bg_primary_support_fallback(question, grouped, top_k)

    primary_pool = grouped.get("buyers_guide", [])
    tbs_pool = grouped.get("tbs_directive", [])
    policy_pool = grouped.get("buy_canadian_policy", [])

    reserved_support: list[ChunkRecord] = []
    if tbs_pool:
        reserved_support.append(tbs_pool[0])
    elif policy_pool:
        reserved_support.append(policy_pool[0])

    if not reserved_support:
        return bg_primary_support_fallback(question, grouped, top_k)

    leading_primary_budget = min(3, max(0, top_k - len(reserved_support)))
    combined = list(primary_pool[:leading_primary_budget])
    selected_ids = {chunk.chunk_id for chunk in combined}

    for chunk in reserved_support:
        if chunk.chunk_id not in selected_ids and len(combined) < top_k:
            combined.append(chunk)
            selected_ids.add(chunk.chunk_id)

    for chunk in primary_pool[leading_primary_budget:]:
        if len(combined) >= top_k:
            return combined[:top_k]
        if chunk.chunk_id in selected_ids:
            continue
        combined.append(chunk)
        selected_ids.add(chunk.chunk_id)

    for pool in (tbs_pool, policy_pool):
        for chunk in pool:
            if len(combined) >= top_k:
                return combined[:top_k]
            if chunk.chunk_id in selected_ids:
                continue
            combined.append(chunk)
            selected_ids.add(chunk.chunk_id)

    return combined[:top_k]


def bg_primary_selective_authority_cluster(question: str, grouped: dict[str, list[ChunkRecord]], top_k: int) -> list[ChunkRecord]:
    if not _requires_authority_support(question):
        return bg_primary_support_fallback(question, grouped, top_k)

    primary_pool = grouped.get("buyers_guide", [])
    tbs_pool = grouped.get("tbs_directive", [])
    policy_pool = grouped.get("buy_canadian_policy", [])

    support_cluster = _leading_doc_cluster(tbs_pool, _AUTHORITY_SUPPORT_CLUSTER_LIMIT)
    if not support_cluster:
        support_cluster = _leading_doc_cluster(policy_pool, _AUTHORITY_SUPPORT_CLUSTER_LIMIT)

    if not support_cluster:
        return bg_primary_support_fallback(question, grouped, top_k)

    leading_primary_budget = min(2, max(0, top_k - len(support_cluster)))
    combined = list(primary_pool[:leading_primary_budget])
    selected_ids = {chunk.chunk_id for chunk in combined}

    for chunk in support_cluster:
        if chunk.chunk_id not in selected_ids and len(combined) < top_k:
            combined.append(chunk)
            selected_ids.add(chunk.chunk_id)

    for chunk in primary_pool[leading_primary_budget:]:
        if len(combined) >= top_k:
            return combined[:top_k]
        if chunk.chunk_id in selected_ids:
            continue
        combined.append(chunk)
        selected_ids.add(chunk.chunk_id)

    for pool in (tbs_pool, policy_pool):
        for chunk in pool:
            if len(combined) >= top_k:
                return combined[:top_k]
            if chunk.chunk_id in selected_ids:
                continue
            combined.append(chunk)
            selected_ids.add(chunk.chunk_id)

    return combined[:top_k]


def unified_source_hybrid(question: str, grouped: dict[str, list[ChunkRecord]], top_k: int) -> list[ChunkRecord]:
    del question
    merged: list[ChunkRecord] = []
    for family in ("buyers_guide", "buy_canadian_policy", "tbs_directive"):
        merged.extend(grouped.get(family, []))
    return merged[:top_k]


def ranked_passthrough(question: str, grouped: dict[str, list[ChunkRecord]], top_k: int) -> list[ChunkRecord]:
    del question
    merged: list[ChunkRecord] = []
    for chunks in grouped.values():
        merged.extend(chunks)
    return merged[:top_k]


source_topology_registry.register("bg_primary_support_fallback", bg_primary_support_fallback)
source_topology_registry.register("bg_primary_authority_reserve", bg_primary_authority_reserve)
source_topology_registry.register("bg_primary_selective_authority_reserve", bg_primary_selective_authority_reserve)
source_topology_registry.register("bg_primary_selective_authority_cluster", bg_primary_selective_authority_cluster)
source_topology_registry.register("unified_source_hybrid", unified_source_hybrid)
source_topology_registry.register("ranked_passthrough", ranked_passthrough)
