"""Span-aware evidence shaping for answer-time packing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import cohere

from bgrag.types import ChunkRecord, RetrievalCandidate


@dataclass(frozen=True)
class SpanCandidate:
    chunk: ChunkRecord
    parent_chunk_id: str
    parent_rank: int
    local_index: int
    rerank_score: float = 0.0


def _normalize_piece(text: str) -> str:
    return " ".join(text.split()).strip()


def _split_text_parts(text: str, *, max_chars: int) -> list[str]:
    normalized = _normalize_piece(text)
    if not normalized:
        return []
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
        if all(len(part) <= max_chars for part in grouped):
            return grouped

    words = normalized.split()
    if not words:
        return []
    pieces: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current)
        current = word
    pieces.append(current)
    return pieces


def _is_label_like(text: str) -> bool:
    stripped = _normalize_piece(text)
    if not stripped:
        return False
    if stripped.endswith(":"):
        return True
    words = stripped.split()
    return (
        stripped[0].isupper()
        and len(stripped) <= 40
        and len(words) <= 4
        and stripped[-1].isalnum()
    )


def _table_row_spans(text: str, *, max_chars: int) -> list[str]:
    normalized = _normalize_piece(text)
    if len(normalized) <= max_chars:
        return [normalized]
    segments = [_normalize_piece(segment) for segment in normalized.split(" | ") if _normalize_piece(segment)]
    if len(segments) <= 1:
        return _split_text_parts(normalized, max_chars=max_chars)
    context = segments[0]
    spans: list[str] = []
    for segment in segments[1:]:
        combined = f"{context} | {segment}"
        if len(combined) <= max_chars:
            spans.append(combined)
            continue
        available = max(120, max_chars - len(context) - 3)
        for piece in _split_text_parts(segment, max_chars=available):
            spans.append(f"{context} | {piece}")
    return spans


def _line_spans(text: str, *, max_chars: int) -> list[str]:
    raw_lines = [line.strip() for line in text.splitlines() if _normalize_piece(line)]
    if len(raw_lines) <= 1:
        return _split_text_parts(text, max_chars=max_chars)

    first = _normalize_piece(raw_lines[0])
    bullet_lines = [line for line in raw_lines[1:] if line.lstrip().startswith("-")]
    spans: list[str] = []
    if _is_label_like(first) and bullet_lines:
        for line in raw_lines[1:]:
            normalized = _normalize_piece(line.lstrip("-").strip())
            if not normalized:
                continue
            spans.extend(_split_text_parts(f"{first} {normalized}", max_chars=max_chars))
        return spans

    for line in raw_lines:
        normalized = _normalize_piece(line.lstrip("-").strip())
        if not normalized:
            continue
        spans.extend(_split_text_parts(normalized, max_chars=max_chars))
    return spans


def split_chunk_into_spans(chunk: ChunkRecord, *, max_chars: int) -> list[ChunkRecord]:
    text = _normalize_piece(chunk.text)
    if not text:
        return []
    if chunk.chunk_type == "table_row" or " | " in text:
        texts = _table_row_spans(text, max_chars=max_chars)
    else:
        texts = _line_spans(chunk.text, max_chars=max_chars)

    spans: list[ChunkRecord] = []
    for index, span_text in enumerate(texts):
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "parent_chunk_id": chunk.chunk_id,
                "evidence_unit": "span",
                "span_index": index,
            }
        )
        spans.append(
            chunk.model_copy(
                update={
                    "chunk_id": f"{chunk.chunk_id}__span__{index}",
                    "chunk_type": "table_row_span" if chunk.chunk_type == "table_row" else "span",
                    "text": span_text,
                    "metadata": metadata,
                    "token_estimate": max(1, len(span_text) // 4),
                }
            )
        )
    return spans or [chunk]


def _select_with_parent_caps(
    ordered: list[SpanCandidate],
    *,
    max_units: int,
    max_per_chunk: int,
    existing_parent_counts: Counter[str] | None = None,
) -> list[ChunkRecord]:
    selected: list[ChunkRecord] = []
    per_parent: Counter[str] = Counter(existing_parent_counts or {})
    for span in ordered:
        if len(selected) >= max_units:
            break
        if max_per_chunk > 0 and per_parent[span.parent_chunk_id] >= max_per_chunk:
            continue
        selected.append(span.chunk)
        per_parent[span.parent_chunk_id] += 1
    return selected


def build_span_packed_chunks(
    *,
    question: str,
    candidates: list[RetrievalCandidate],
    max_units: int,
    max_chars: int,
    candidate_chunk_limit: int,
    max_per_chunk: int,
    rerank_client: cohere.ClientV2 | None,
    rerank_model: str,
    rerank_top_n: int,
) -> list[ChunkRecord]:
    if not candidates or max_units <= 0:
        return []

    parent_candidates = candidates[: max(1, min(candidate_chunk_limit, len(candidates)))]
    span_candidates: list[SpanCandidate] = []
    for parent_rank, candidate in enumerate(parent_candidates, start=1):
        for local_index, span_chunk in enumerate(split_chunk_into_spans(candidate.chunk, max_chars=max_chars)):
            span_candidates.append(
                SpanCandidate(
                    chunk=span_chunk,
                    parent_chunk_id=candidate.chunk.chunk_id,
                    parent_rank=parent_rank,
                    local_index=local_index,
                )
            )
    if not span_candidates:
        return [candidate.chunk for candidate in candidates[:max_units]]

    fallback_order = sorted(
        span_candidates,
        key=lambda item: (item.parent_rank, item.local_index, item.chunk.order),
    )
    if rerank_client is None or rerank_top_n <= 0:
        return _select_with_parent_caps(
            fallback_order,
            max_units=max_units,
            max_per_chunk=max_per_chunk,
        )

    response = rerank_client.rerank(
        model=rerank_model,
        query=question,
        documents=[span.chunk.text for span in span_candidates],
        top_n=min(rerank_top_n, len(span_candidates)),
    )
    reranked: list[SpanCandidate] = [
        SpanCandidate(
            chunk=span_candidates[result.index].chunk,
            parent_chunk_id=span_candidates[result.index].parent_chunk_id,
            parent_rank=span_candidates[result.index].parent_rank,
            local_index=span_candidates[result.index].local_index,
            rerank_score=float(result.relevance_score),
        )
        for result in response.results
    ]
    selected = _select_with_parent_caps(
        reranked,
        max_units=max_units,
        max_per_chunk=max_per_chunk,
    )
    if len(selected) >= max_units:
        return selected

    selected_ids = {chunk.chunk_id for chunk in selected}
    fallback_fill = [
        span for span in fallback_order if span.chunk.chunk_id not in selected_ids
    ]
    existing_parent_counts = Counter(str(chunk.metadata.get("parent_chunk_id", "")) for chunk in selected)
    selected.extend(
        _select_with_parent_caps(
            fallback_fill,
            max_units=max_units - len(selected),
            max_per_chunk=max_per_chunk,
            existing_parent_counts=existing_parent_counts,
        )
    )
    return selected[:max_units]
