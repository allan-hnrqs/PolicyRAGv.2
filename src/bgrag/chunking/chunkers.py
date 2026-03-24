"""Chunker implementations."""

from __future__ import annotations

import bgrag.metadata.enrichers  # Ensure enrichers are registered before use.

from bgrag.registry import chunker_registry, metadata_enricher_registry
from bgrag.types import ChunkRecord, NormalizedDocument


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _apply_enrichers(document: NormalizedDocument, chunk: ChunkRecord, enrichers: list[str]) -> ChunkRecord:
    enriched = chunk
    for name in enrichers:
        enriched = metadata_enricher_registry.get(name)(document, enriched)
    return enriched


def section_chunker(document: NormalizedDocument, enrichers: list[str] | None = None, **_: object) -> list[ChunkRecord]:
    enrichers = enrichers or []
    chunks: list[ChunkRecord] = []
    for index, block in enumerate(document.structure_blocks):
        if block.block_type == "heading":
            continue
        chunk = ChunkRecord(
            chunk_id=f"{document.doc_id}__section__{index}",
            doc_id=document.doc_id,
            canonical_url=document.canonical_url,
            title=document.title,
            source_family=document.source_family,
            authority_rank=document.authority_rank,
            chunker_name="section_chunker",
            chunk_type=block.block_type,
            text=block.text,
            heading=block.heading,
            heading_path=list(block.heading_path),
            section_id=block.block_id,
            order=block.order,
            token_estimate=_estimate_tokens(block.text),
        )
        chunks.append(_apply_enrichers(document, chunk, enrichers))
    return chunks


def block_chunker(document: NormalizedDocument, enrichers: list[str] | None = None, **_: object) -> list[ChunkRecord]:
    enrichers = enrichers or []
    chunks: list[ChunkRecord] = []
    buffer: list[str] = []
    current_heading: list[str] = []
    start_order = 0
    for block in document.structure_blocks:
        if block.block_type == "heading":
            if buffer:
                text = "\n".join(buffer).strip()
                chunk = ChunkRecord(
                    chunk_id=f"{document.doc_id}__block__{start_order}",
                    doc_id=document.doc_id,
                    canonical_url=document.canonical_url,
                    title=document.title,
                    source_family=document.source_family,
                    authority_rank=document.authority_rank,
                    chunker_name="block_chunker",
                    chunk_type="block_group",
                    text=text,
                    heading=current_heading[-1] if current_heading else None,
                    heading_path=list(current_heading),
                    section_id=f"block_{start_order:04d}",
                    order=start_order,
                    token_estimate=_estimate_tokens(text),
                )
                chunks.append(_apply_enrichers(document, chunk, enrichers))
                buffer = []
            current_heading = list(block.heading_path)
            start_order = block.order
            continue
        if not buffer:
            start_order = block.order
        buffer.append(block.text)
    if buffer:
        text = "\n".join(buffer).strip()
        chunk = ChunkRecord(
            chunk_id=f"{document.doc_id}__block__{start_order}",
            doc_id=document.doc_id,
            canonical_url=document.canonical_url,
            title=document.title,
            source_family=document.source_family,
            authority_rank=document.authority_rank,
            chunker_name="block_chunker",
            chunk_type="block_group",
            text=text,
            heading=current_heading[-1] if current_heading else None,
            heading_path=list(current_heading),
            section_id=f"block_{start_order:04d}",
            order=start_order,
            token_estimate=_estimate_tokens(text),
        )
        chunks.append(_apply_enrichers(document, chunk, enrichers))
    return chunks


def sliding_window_chunker(
    document: NormalizedDocument,
    enrichers: list[str] | None = None,
    window_chars: int = 1200,
    overlap_chars: int = 200,
    **_: object,
) -> list[ChunkRecord]:
    enrichers = enrichers or []
    text = document.raw_text.strip()
    chunks: list[ChunkRecord] = []
    if not text:
        return chunks
    start = 0
    order = 0
    while start < len(text):
        end = min(len(text), start + window_chars)
        window = text[start:end].strip()
        if not window:
            break
        chunk = ChunkRecord(
            chunk_id=f"{document.doc_id}__window__{order}",
            doc_id=document.doc_id,
            canonical_url=document.canonical_url,
            title=document.title,
            source_family=document.source_family,
            authority_rank=document.authority_rank,
            chunker_name="sliding_window_chunker",
            chunk_type="sliding_window",
            text=window,
            order=order,
            token_estimate=_estimate_tokens(window),
        )
        chunks.append(_apply_enrichers(document, chunk, enrichers))
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
        order += 1
    return chunks


chunker_registry.register("section_chunker", section_chunker)
chunker_registry.register("block_chunker", block_chunker)
chunker_registry.register("sliding_window_chunker", sliding_window_chunker)
