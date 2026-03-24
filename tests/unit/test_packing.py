from bgrag.retrieval.packing import diversify_ranked_chunks
from bgrag.types import ChunkRecord, SourceFamily


def _chunk(
    chunk_id: str,
    *,
    doc_id: str,
    heading_path: list[str],
    text: str = "text",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        canonical_url=f"https://example.com/{doc_id}",
        title=doc_id,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text=text,
        heading_path=heading_path,
    )


def test_diversify_ranked_chunks_surfaces_distinct_heading_groups_first() -> None:
    chunks = [
        _chunk("a1", doc_id="doc-a", heading_path=["Root", "Standing offer"]),
        _chunk("a2", doc_id="doc-a", heading_path=["Root", "Standing offer"]),
        _chunk("a3", doc_id="doc-a", heading_path=["Root", "Standing offer"]),
        _chunk("b1", doc_id="doc-b", heading_path=["Root", "Supply arrangement"]),
        _chunk("c1", doc_id="doc-c", heading_path=["Root", "Comparison"]),
    ]

    reordered = diversify_ranked_chunks(
        chunks,
        target_k=4,
        cover_fraction=0.5,
        max_per_document=8,
        max_per_heading=4,
        seed_chunks_per_heading=1,
    )

    first_four = [chunk.chunk_id for chunk in reordered[:4]]
    assert "b1" in first_four
    assert "c1" in first_four


def test_diversify_ranked_chunks_caps_heading_dominance_in_target_prefix() -> None:
    chunks = [
        _chunk("a1", doc_id="doc-a", heading_path=["Root", "Section A"]),
        _chunk("a2", doc_id="doc-a", heading_path=["Root", "Section A"]),
        _chunk("a3", doc_id="doc-a", heading_path=["Root", "Section A"]),
        _chunk("a4", doc_id="doc-a", heading_path=["Root", "Section A"]),
        _chunk("b1", doc_id="doc-b", heading_path=["Root", "Section B"]),
        _chunk("c1", doc_id="doc-c", heading_path=["Root", "Section C"]),
    ]

    reordered = diversify_ranked_chunks(
        chunks,
        target_k=4,
        cover_fraction=0.5,
        max_per_document=8,
        max_per_heading=2,
        seed_chunks_per_heading=1,
    )

    first_four = reordered[:4]
    section_a_count = sum(1 for chunk in first_four if chunk.heading_path[-1] == "Section A")
    assert section_a_count <= 2


def test_diversify_ranked_chunks_keeps_top_siblings_together_for_selected_heading() -> None:
    chunks = [
        _chunk("a1", doc_id="doc-a", heading_path=["Root", "Section A"]),
        _chunk("a2", doc_id="doc-a", heading_path=["Root", "Section A"]),
        _chunk("b1", doc_id="doc-b", heading_path=["Root", "Section B"]),
        _chunk("c1", doc_id="doc-c", heading_path=["Root", "Section C"]),
    ]

    reordered = diversify_ranked_chunks(
        chunks,
        target_k=4,
        cover_fraction=0.5,
        max_per_document=8,
        max_per_heading=4,
        seed_chunks_per_heading=2,
    )

    assert [chunk.chunk_id for chunk in reordered[:2]] == ["a1", "a2"]
