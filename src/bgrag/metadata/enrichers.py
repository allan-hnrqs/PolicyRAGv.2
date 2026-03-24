"""Metadata enrichers for normalized documents and chunks."""

from __future__ import annotations

from bgrag.registry import metadata_enricher_registry
from bgrag.types import ChunkRecord, NormalizedDocument


def authority_metadata(document: NormalizedDocument, chunk: ChunkRecord) -> ChunkRecord:
    chunk.metadata["authority_rank"] = document.authority_rank
    chunk.metadata["source_family"] = document.source_family.value
    return chunk


def lineage_metadata(document: NormalizedDocument, chunk: ChunkRecord) -> ChunkRecord:
    chunk.metadata["lineage_doc_ids"] = ",".join(document.graph.lineage_doc_ids)
    chunk.metadata["lineage_urls"] = ",".join(document.graph.lineage_urls)
    return chunk


def scope_tag_metadata(document: NormalizedDocument, chunk: ChunkRecord) -> ChunkRecord:
    tags: list[str] = []
    normalized = chunk.text.lower()
    if "standing offer" in normalized:
        tags.append("standing_offer")
    if "supply arrangement" in normalized:
        tags.append("supply_arrangement")
    if "trade agreement" in normalized:
        tags.append("trade_agreement")
    if "acan" in normalized:
        tags.append("acan")
    if "vendor performance" in normalized or "vpcm" in normalized:
        tags.append("vpcm")
    chunk.metadata["scope_tags"] = ",".join(tags)
    return chunk


def source_topology_metadata(document: NormalizedDocument, chunk: ChunkRecord) -> ChunkRecord:
    chunk.metadata["bg_primary"] = document.source_family.value == "buyers_guide"
    chunk.metadata["has_external_links"] = bool(document.graph.outgoing_in_scope_links)
    return chunk


metadata_enricher_registry.register("authority_metadata", authority_metadata)
metadata_enricher_registry.register("lineage_metadata", lineage_metadata)
metadata_enricher_registry.register("scope_tag_metadata", scope_tag_metadata)
metadata_enricher_registry.register("source_topology_metadata", source_topology_metadata)
