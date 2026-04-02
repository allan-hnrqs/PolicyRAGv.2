"""Corpus and chunk-shape audit helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from bgrag.types import ChunkRecord, NormalizedDocument


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * p))
    return ordered[index]


def _text_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "count": len(values),
        "mean": round(mean(values), 2),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def build_corpus_audit(
    documents: list[NormalizedDocument],
    chunks: list[ChunkRecord],
) -> dict[str, object]:
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    duplicated_first_block_docs: list[dict[str, object]] = []
    table_spill_docs: list[dict[str, object]] = []
    short_fragment_docs: list[dict[str, object]] = []
    long_tail_examples = sorted(chunks, key=lambda item: len(item.text), reverse=True)[:20]
    tiny_examples = [chunk for chunk in sorted(chunks, key=lambda item: len(item.text)) if len(chunk.text) <= 25][:30]

    docs_by_id = {document.doc_id: document for document in documents}
    source_family_counts = Counter(document.source_family.value for document in documents)
    chunk_type_counts = Counter(chunk.chunk_type for chunk in chunks)

    for doc_id, doc_chunks in by_doc.items():
        ranked = sorted(doc_chunks, key=lambda item: item.order)
        if not ranked:
            continue
        first = ranked[0]
        later_nontrivial = [chunk for chunk in ranked[1:] if len(chunk.text) >= 20]
        if later_nontrivial and all(chunk.text in first.text for chunk in later_nontrivial):
            document = docs_by_id.get(doc_id)
            duplicated_first_block_docs.append(
                {
                    "doc_id": doc_id,
                    "title": document.title if document else first.title,
                    "source_family": document.source_family.value if document else first.source_family.value,
                    "first_chunk_id": first.chunk_id,
                    "first_chunk_chars": len(first.text),
                    "later_chunk_count": len(later_nontrivial),
                }
            )

        has_table = any(chunk.chunk_type == "table" for chunk in ranked)
        tiny_count = sum(1 for chunk in ranked if len(chunk.text) <= 25)
        if has_table and tiny_count:
            document = docs_by_id.get(doc_id)
            table_spill_docs.append(
                {
                    "doc_id": doc_id,
                    "title": document.title if document else first.title,
                    "tiny_chunk_count": tiny_count,
                }
            )

        if len(ranked) >= 10:
            short_count = sum(1 for chunk in ranked if len(chunk.text) < 40)
            share = short_count / len(ranked)
            if share >= 0.25:
                document = docs_by_id.get(doc_id)
                short_fragment_docs.append(
                    {
                        "doc_id": doc_id,
                        "title": document.title if document else first.title,
                        "short_chunk_count": short_count,
                        "total_chunks": len(ranked),
                        "short_share": round(share, 2),
                    }
                )

    duplicated_first_block_docs.sort(key=lambda item: int(item["first_chunk_chars"]), reverse=True)
    table_spill_docs.sort(key=lambda item: int(item["tiny_chunk_count"]), reverse=True)
    short_fragment_docs.sort(key=lambda item: (float(item["short_share"]), int(item["short_chunk_count"])), reverse=True)

    duplicated_chars = sum(int(item["first_chunk_chars"]) for item in duplicated_first_block_docs)
    total_chunk_chars = sum(chunk_lengths)

    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "source_family_counts": dict(source_family_counts),
        "chunk_type_counts": dict(chunk_type_counts),
        "chunk_char_stats": _text_stats(chunk_lengths),
        "tiny_chunk_counts": {
            "le_25_chars": sum(1 for length in chunk_lengths if length <= 25),
            "le_50_chars": sum(1 for length in chunk_lengths if length <= 50),
        },
        "oversized_chunk_counts": {
            "ge_1200_chars": sum(1 for length in chunk_lengths if length >= 1200),
            "ge_2000_chars": sum(1 for length in chunk_lengths if length >= 2000),
            "ge_8000_chars": sum(1 for length in chunk_lengths if length >= 8000),
        },
        "duplicated_first_block_summary": {
            "doc_count": len(duplicated_first_block_docs),
            "duplicated_first_block_chars": duplicated_chars,
            "share_of_total_chunk_chars": round(duplicated_chars / max(1, total_chunk_chars), 4),
            "examples": duplicated_first_block_docs[:20],
        },
        "table_spill_summary": {
            "doc_count": len(table_spill_docs),
            "examples": table_spill_docs[:20],
        },
        "short_fragment_doc_summary": {
            "doc_count": len(short_fragment_docs),
            "examples": short_fragment_docs[:20],
        },
        "long_chunk_examples": [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "chunk_type": chunk.chunk_type,
                "chars": len(chunk.text),
            }
            for chunk in long_tail_examples
        ],
        "tiny_chunk_examples": [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "chunk_type": chunk.chunk_type,
                "chars": len(chunk.text),
                "text": chunk.text,
            }
            for chunk in tiny_examples
        ],
    }


def write_corpus_audit(path: Path, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
