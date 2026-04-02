from datetime import datetime, timezone

from bgrag.corpus_audit import build_corpus_audit
from bgrag.types import ChunkRecord, NormalizedDocument, SourceFamily, SourceGraph, StructureBlock


def _document(doc_id: str, title: str, blocks: list[StructureBlock]) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id=doc_id,
        title=title,
        source_url=f"https://example.test/{doc_id}",
        canonical_url=f"https://example.test/{doc_id}",
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        fetched_at=datetime.now(timezone.utc),
        content_hash=doc_id,
        word_count=0,
        extraction_method="test",
        breadcrumbs=[],
        graph=SourceGraph(),
        structure_blocks=blocks,
        raw_text="",
        markdown_text="",
    )


def _block(order: int, block_type: str, text: str) -> StructureBlock:
    return StructureBlock(
        block_id=f"block_{order:04d}",
        block_type=block_type,
        heading="Heading" if block_type != "heading" else text,
        heading_path=["Heading"] if block_type != "heading" else [text],
        text=text,
        order=order,
    )


def _chunk(doc_id: str, order: int, chunk_type: str, text: str, title: str = "Title") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{doc_id}__section__{order}",
        doc_id=doc_id,
        canonical_url=f"https://example.test/{doc_id}",
        title=title,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type=chunk_type,
        text=text,
        heading="Heading",
        heading_path=["Heading"],
        section_id=f"block_{order:04d}",
        order=order,
        token_estimate=max(1, len(text) // 4),
    )


def test_build_corpus_audit_detects_duplicated_first_block_and_table_spill() -> None:
    docs = [
        _document(
            "doc_a",
            "Doc A",
            [
                _block(0, "paragraph", "Alpha intro Beta rule requires documented approval Gamma rule requires publication notice"),
                _block(1, "paragraph", "Beta rule requires documented approval"),
                _block(2, "paragraph", "Gamma rule requires publication notice"),
            ],
        ),
        _document(
            "doc_b",
            "Doc B",
            [
                _block(0, "table", "Instrument | Electronic?"),
                _block(1, "paragraph", "yes"),
                _block(2, "paragraph", "no"),
            ],
        ),
    ]
    chunks = [
        _chunk("doc_a", 0, "paragraph", "Alpha intro Beta rule requires documented approval Gamma rule requires publication notice", title="Doc A"),
        _chunk("doc_a", 1, "paragraph", "Beta rule requires documented approval", title="Doc A"),
        _chunk("doc_a", 2, "paragraph", "Gamma rule requires publication notice", title="Doc A"),
        _chunk("doc_b", 0, "table", "Instrument | Electronic?", title="Doc B"),
        _chunk("doc_b", 1, "paragraph", "yes", title="Doc B"),
        _chunk("doc_b", 2, "paragraph", "no", title="Doc B"),
    ]

    audit = build_corpus_audit(docs, chunks)

    assert audit["chunk_count"] == 6
    assert audit["duplicated_first_block_summary"]["doc_count"] == 1
    assert audit["table_spill_summary"]["doc_count"] == 1
    assert audit["tiny_chunk_counts"]["le_25_chars"] == 3
