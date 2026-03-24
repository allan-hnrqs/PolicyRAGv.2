from bgrag.retrieval.topology import bg_primary_support_fallback, unified_source_hybrid
from bgrag.types import ChunkRecord, SourceFamily


def _chunk(chunk_id: str, family: SourceFamily, text: str = "text") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        canonical_url=f"https://example.com/{chunk_id}",
        title=chunk_id,
        source_family=family,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text=text,
    )


def test_bg_primary_support_fallback_prefers_buyers_guide() -> None:
    grouped = {
        "buyers_guide": [_chunk("bg1", SourceFamily.BUYERS_GUIDE), _chunk("bg2", SourceFamily.BUYERS_GUIDE)],
        "buy_canadian_policy": [_chunk("bc1", SourceFamily.BUY_CANADIAN_POLICY)],
    }
    result = bg_primary_support_fallback("simple buyer question", grouped, top_k=2)
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bg2"]


def test_unified_source_hybrid_uses_all_sources() -> None:
    grouped = {
        "buyers_guide": [_chunk("bg1", SourceFamily.BUYERS_GUIDE)],
        "buy_canadian_policy": [_chunk("bc1", SourceFamily.BUY_CANADIAN_POLICY)],
        "tbs_directive": [_chunk("tbs1", SourceFamily.TBS_DIRECTIVE)],
    }
    result = unified_source_hybrid("policy question", grouped, top_k=3)
    assert len(result) == 3


def test_bg_primary_support_fallback_reserves_slots_for_support_when_needed() -> None:
    grouped = {
        "buyers_guide": [
            _chunk("bg1", SourceFamily.BUYERS_GUIDE, "buyers guide procurement text"),
            _chunk("bg2", SourceFamily.BUYERS_GUIDE, "buyers guide procurement text"),
            _chunk("bg3", SourceFamily.BUYERS_GUIDE, "buyers guide procurement text"),
            _chunk("bg4", SourceFamily.BUYERS_GUIDE, "buyers guide procurement text"),
        ],
        "buy_canadian_policy": [
            _chunk("bc1", SourceFamily.BUY_CANADIAN_POLICY, "buy canadian trade agreement policy text"),
            _chunk("bc2", SourceFamily.BUY_CANADIAN_POLICY, "buy canadian trade agreement policy text"),
        ],
    }
    result = bg_primary_support_fallback(
        "For a supply arrangement, when do trade agreements kick in?",
        grouped,
        top_k=4,
    )
    assert any(chunk.source_family == SourceFamily.BUY_CANADIAN_POLICY for chunk in result)
    assert len(result) == 4
