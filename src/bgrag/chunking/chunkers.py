"""Chunker implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass

import bgrag.metadata.enrichers  # Ensure enrichers are registered before use.

from bgrag.registry import chunker_registry, metadata_enricher_registry
from bgrag.types import ChunkRecord, NormalizedDocument, StructureBlock

_MAX_SECTION_CHARS = 700
_TARGET_GROUP_CHARS = 140
_MAX_GROUP_CHARS = 360
_SHORT_GROUPABLE_CHARS = 80
_LISTISH_BLOCK_TYPES = {"list_item", "table", "table_row", "definition_detail"}
_LABELISH_BLOCK_TYPES = {"paragraph", "definition_term", "list_item"}


@dataclass(frozen=True)
class _ChunkUnit:
    block: StructureBlock
    chunk_type: str
    text: str


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _apply_enrichers(document: NormalizedDocument, chunk: ChunkRecord, enrichers: list[str]) -> ChunkRecord:
    enriched = chunk
    for name in enrichers:
        enriched = metadata_enricher_registry.get(name)(document, enriched)
    return enriched


def _normalize_piece(text: str) -> str:
    return " ".join(text.split()).strip()


def _split_text_parts(text: str, *, max_chars: int) -> list[str]:
    normalized = _normalize_piece(text)
    if len(normalized) <= max_chars:
        return [normalized]

    for pattern in (r"(?<=[.!?])\s+", r";\s+", r"(?<=:)\s+(?=[A-Z0-9])"):
        parts = [_normalize_piece(part) for part in re.split(pattern, normalized) if _normalize_piece(part)]
        if len(parts) <= 1:
            continue
        grouped: list[str] = []
        current = parts[0]
        for part in parts[1:]:
            candidate = f"{current} {part}"
            if len(candidate) <= max_chars:
                current = candidate
                continue
            grouped.append(current)
            current = part
        grouped.append(current)
        if grouped and all(len(part) <= max_chars for part in grouped):
            return grouped

    words = normalized.split()
    if not words:
        return []
    fallback: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        fallback.append(current)
        current = word
    fallback.append(current)
    return fallback


def _split_table_row_unit(unit: _ChunkUnit, *, max_chars: int) -> list[_ChunkUnit]:
    text = _normalize_piece(unit.text)
    if len(text) <= max_chars:
        return [unit]

    segments = [_normalize_piece(segment) for segment in text.split(" | ") if _normalize_piece(segment)]
    if len(segments) <= 1:
        return [_ChunkUnit(block=unit.block, chunk_type=unit.chunk_type, text=piece) for piece in _split_text_parts(text, max_chars=max_chars)]

    context = segments[0]
    derived: list[_ChunkUnit] = []
    for segment in segments[1:]:
        combined = f"{context} | {segment}"
        if len(combined) <= max_chars:
            derived.append(_ChunkUnit(block=unit.block, chunk_type="table_row", text=combined))
            continue
        available = max(200, max_chars - len(context) - 3)
        for piece in _split_text_parts(segment, max_chars=available):
            derived.append(_ChunkUnit(block=unit.block, chunk_type="table_row", text=f"{context} | {piece}"))
    return derived or [unit]


def _split_block_into_units(block: StructureBlock) -> list[_ChunkUnit]:
    base = _ChunkUnit(block=block, chunk_type=block.block_type, text=_normalize_piece(block.text))
    if not base.text:
        return []
    if block.block_type in {"table", "table_row"} and "|" in base.text:
        return _split_table_row_unit(base, max_chars=_MAX_SECTION_CHARS)
    if len(base.text) <= _MAX_SECTION_CHARS:
        return [base]
    return [_ChunkUnit(block=block, chunk_type=block.block_type, text=piece) for piece in _split_text_parts(base.text, max_chars=_MAX_SECTION_CHARS)]


def _is_label_like(unit: _ChunkUnit) -> bool:
    if unit.chunk_type not in _LABELISH_BLOCK_TYPES:
        return False
    if unit.text.endswith(":"):
        return True
    words = unit.text.split()
    if (
        unit.chunk_type in {"paragraph", "definition_term"}
        and unit.text
        and unit.text[0].isupper()
        and len(unit.text) <= 40
        and len(words) <= 4
        and unit.text[-1].isalnum()
    ):
        return True
    return False


def _is_shortish(unit: _ChunkUnit) -> bool:
    if len(unit.text) < _SHORT_GROUPABLE_CHARS:
        return True
    if _is_label_like(unit):
        return True
    if unit.chunk_type in {"definition_term", "table_row"} and len(unit.text) < 160:
        return True
    return False


def _same_heading(a: _ChunkUnit, b: _ChunkUnit) -> bool:
    return list(a.block.heading_path) == list(b.block.heading_path)


def _can_group(group: list[_ChunkUnit], candidate: _ChunkUnit) -> bool:
    if not group or not _same_heading(group[0], candidate):
        return False
    group_types = {unit.chunk_type for unit in group}
    if all(unit.chunk_type in _LISTISH_BLOCK_TYPES for unit in group) and candidate.chunk_type in _LISTISH_BLOCK_TYPES:
        return True
    if _is_label_like(group[-1]) and candidate.chunk_type in (_LISTISH_BLOCK_TYPES | {"paragraph", "definition_detail"}):
        return True
    if group[0].chunk_type == "definition_term" and candidate.chunk_type in {"definition_detail", "paragraph", "list_item"}:
        return True
    if group_types <= {"paragraph", "definition_term", "definition_detail"} and candidate.chunk_type in {
        "paragraph",
        "definition_term",
        "definition_detail",
    }:
        return True
    return False


def _group_text(group: list[_ChunkUnit]) -> str:
    if not group:
        return ""
    lines: list[str] = []
    starts_with_label = _is_label_like(group[0]) or group[0].chunk_type == "definition_term"
    for index, unit in enumerate(group):
        if index == 0:
            lines.append(unit.text)
            continue
        if starts_with_label or unit.chunk_type in _LISTISH_BLOCK_TYPES:
            lines.append(f"- {unit.text}")
        else:
            lines.append(unit.text)
    return "\n".join(lines)


def _merge_chunk_units(units: list[_ChunkUnit]) -> list[_ChunkUnit]:
    merged: list[_ChunkUnit] = []
    index = 0
    while index < len(units):
        current = units[index]
        if not _is_shortish(current):
            merged.append(current)
            index += 1
            continue

        group = [current]
        total = len(current.text)
        lookahead = index + 1
        while lookahead < len(units):
            candidate = units[lookahead]
            if not _can_group(group, candidate):
                break
            projected = total + 1 + len(candidate.text)
            if projected > _MAX_GROUP_CHARS and total >= _TARGET_GROUP_CHARS:
                break
            group.append(candidate)
            total = projected
            lookahead += 1
            if total >= _TARGET_GROUP_CHARS and not _is_shortish(candidate):
                break

        if len(group) == 1:
            merged.append(current)
            index += 1
            continue

        merged.append(
            _ChunkUnit(
                block=group[0].block,
                chunk_type=group[0].chunk_type if len({unit.chunk_type for unit in group}) == 1 else "grouped_section",
                text=_group_text(group),
            )
        )
        index = lookahead
    return merged


def _shape_section_units(document: NormalizedDocument) -> list[_ChunkUnit]:
    units: list[_ChunkUnit] = []
    for block in document.structure_blocks:
        if block.block_type == "heading":
            continue
        units.extend(_split_block_into_units(block))
    return _merge_chunk_units(units)


def section_chunker(document: NormalizedDocument, enrichers: list[str] | None = None, **_: object) -> list[ChunkRecord]:
    enrichers = enrichers or []
    chunks: list[ChunkRecord] = []
    for index, unit in enumerate(_shape_section_units(document)):
        block = unit.block
        chunk = ChunkRecord(
            chunk_id=f"{document.doc_id}__section__{index}",
            doc_id=document.doc_id,
            canonical_url=document.canonical_url,
            title=document.title,
            source_family=document.source_family,
            authority_rank=document.authority_rank,
            chunker_name="section_chunker",
            chunk_type=unit.chunk_type,
            text=unit.text,
            heading=block.heading,
            heading_path=list(block.heading_path),
            section_id=block.block_id,
            order=block.order,
            token_estimate=_estimate_tokens(unit.text),
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
