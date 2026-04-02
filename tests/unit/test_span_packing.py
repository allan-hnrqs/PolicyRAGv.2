from types import SimpleNamespace

from bgrag.retrieval.span_packing import build_span_packed_chunks, split_chunk_into_spans
from bgrag.types import ChunkRecord, RetrievalCandidate, SourceFamily


def _chunk(chunk_id: str, text: str, *, chunk_type: str = "paragraph") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        canonical_url=f"https://example.com/{chunk_id}",
        title=chunk_id,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type=chunk_type,
        text=text,
        heading_path=["Root", chunk_id],
    )


def _candidate(chunk_id: str, text: str, *, score: float, chunk_type: str = "paragraph") -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=_chunk(chunk_id, text, chunk_type=chunk_type),
        blended_score=score,
    )


def test_split_chunk_into_spans_preserves_table_context() -> None:
    spans = split_chunk_into_spans(
        _chunk(
            "c1",
            "Standing offer | Canada is not obligated until a call-up is issued. | Supply arrangement | Not a contract.",
            chunk_type="table_row",
        ),
        max_chars=80,
    )

    assert len(spans) == 3
    assert spans[0].chunk_id == "c1__span__0"
    assert spans[0].text.startswith("Standing offer |")
    assert spans[1].text.startswith("Standing offer |")
    assert spans[2].text.startswith("Standing offer |")


def test_build_span_packed_chunks_respects_parent_caps_across_rerank_and_fill() -> None:
    candidates = [
        _candidate("c1", "Label:\n- First supported point.\n- Second supported point.", score=1.0),
        _candidate("c2", "Independent paragraph with other evidence.", score=0.9),
    ]

    class FakeClient:
        def rerank(self, *, model: str, query: str, documents: list[str], top_n: int):
            del model, query, top_n
            # Prefer the first span from c1 and leave the rest to fallback fill.
            return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=0.99)])

    packed = build_span_packed_chunks(
        question="What is the rule?",
        candidates=candidates,
        max_units=2,
        max_chars=80,
        candidate_chunk_limit=2,
        max_per_chunk=1,
        rerank_client=FakeClient(),
        rerank_model="rerank-v4.0-fast",
        rerank_top_n=1,
    )

    assert len(packed) == 2
    parent_ids = [chunk.metadata["parent_chunk_id"] for chunk in packed]
    assert parent_ids.count("c1") == 1
    assert parent_ids.count("c2") == 1
