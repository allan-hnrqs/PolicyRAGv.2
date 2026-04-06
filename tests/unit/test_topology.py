from bgrag.retrieval.topology import (
    bg_primary_authority_reserve,
    bg_primary_selective_authority_cluster,
    bg_primary_selective_authority_reserve,
    bg_primary_support_fallback,
    unified_source_hybrid,
)
from bgrag.types import ChunkRecord, SourceFamily


def _chunk(chunk_id: str, family: SourceFamily, text: str = "text", *, doc_id: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id or chunk_id,
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


def test_bg_primary_support_fallback_uses_support_only_after_primary_is_exhausted() -> None:
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
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bg2", "bg3", "bg4"]


def test_bg_primary_support_fallback_backfills_with_support_when_primary_is_short() -> None:
    grouped = {
        "buyers_guide": [_chunk("bg1", SourceFamily.BUYERS_GUIDE), _chunk("bg2", SourceFamily.BUYERS_GUIDE)],
        "buy_canadian_policy": [_chunk("bc1", SourceFamily.BUY_CANADIAN_POLICY)],
        "tbs_directive": [_chunk("tbs1", SourceFamily.TBS_DIRECTIVE)],
    }
    result = bg_primary_support_fallback("policy question", grouped, top_k=4)
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bg2", "bc1", "tbs1"]


def test_bg_primary_authority_reserve_keeps_one_tbs_chunk_even_when_primary_fills_budget() -> None:
    grouped = {
        "buyers_guide": [
            _chunk("bg1", SourceFamily.BUYERS_GUIDE),
            _chunk("bg2", SourceFamily.BUYERS_GUIDE),
            _chunk("bg3", SourceFamily.BUYERS_GUIDE),
        ],
        "tbs_directive": [_chunk("tbs1", SourceFamily.TBS_DIRECTIVE)],
        "buy_canadian_policy": [_chunk("bc1", SourceFamily.BUY_CANADIAN_POLICY)],
    }
    result = bg_primary_authority_reserve("authority question", grouped, top_k=3)
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bg2", "tbs1"]


def test_bg_primary_authority_reserve_falls_back_to_policy_when_tbs_missing() -> None:
    grouped = {
        "buyers_guide": [
            _chunk("bg1", SourceFamily.BUYERS_GUIDE),
            _chunk("bg2", SourceFamily.BUYERS_GUIDE),
        ],
        "buy_canadian_policy": [_chunk("bc1", SourceFamily.BUY_CANADIAN_POLICY)],
    }
    result = bg_primary_authority_reserve("policy question", grouped, top_k=2)
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bc1"]


def test_bg_primary_selective_authority_reserve_behaves_like_default_for_non_authority_questions() -> None:
    grouped = {
        "buyers_guide": [
            _chunk("bg1", SourceFamily.BUYERS_GUIDE),
            _chunk("bg2", SourceFamily.BUYERS_GUIDE),
            _chunk("bg3", SourceFamily.BUYERS_GUIDE),
        ],
        "tbs_directive": [_chunk("tbs1", SourceFamily.TBS_DIRECTIVE)],
    }
    result = bg_primary_selective_authority_reserve("How do I handle a late paper offer?", grouped, top_k=3)
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bg2", "bg3"]


def test_bg_primary_selective_authority_reserve_inserts_tbs_early_for_authority_questions() -> None:
    grouped = {
        "buyers_guide": [
            _chunk("bg1", SourceFamily.BUYERS_GUIDE),
            _chunk("bg2", SourceFamily.BUYERS_GUIDE),
            _chunk("bg3", SourceFamily.BUYERS_GUIDE),
            _chunk("bg4", SourceFamily.BUYERS_GUIDE),
        ],
        "tbs_directive": [_chunk("tbs1", SourceFamily.TBS_DIRECTIVE)],
    }
    result = bg_primary_selective_authority_reserve(
        "According to the Treasury Board directive, who owns the policy framework?",
        grouped,
        top_k=4,
    )
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bg2", "bg3", "tbs1"]


def test_bg_primary_selective_authority_cluster_keeps_multiple_chunks_from_same_support_doc() -> None:
    grouped = {
        "buyers_guide": [
            _chunk("bg1", SourceFamily.BUYERS_GUIDE),
            _chunk("bg2", SourceFamily.BUYERS_GUIDE),
            _chunk("bg3", SourceFamily.BUYERS_GUIDE),
            _chunk("bg4", SourceFamily.BUYERS_GUIDE),
        ],
        "tbs_directive": [
            _chunk("tbs462", SourceFamily.TBS_DIRECTIVE, doc_id="tbs_doc"),
            _chunk("tbs459", SourceFamily.TBS_DIRECTIVE, doc_id="tbs_doc"),
            _chunk("tbs460", SourceFamily.TBS_DIRECTIVE, doc_id="tbs_doc"),
            _chunk("tbs463", SourceFamily.TBS_DIRECTIVE, doc_id="tbs_doc"),
            _chunk("other_doc", SourceFamily.TBS_DIRECTIVE, doc_id="other_doc"),
        ],
    }
    result = bg_primary_selective_authority_cluster(
        "According to the Treasury Board directive, who owns the policy framework and what must authorities document?",
        grouped,
        top_k=6,
    )
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bg2", "tbs462", "tbs459", "tbs460", "tbs463"]


def test_bg_primary_selective_authority_cluster_uses_default_for_non_authority_questions() -> None:
    grouped = {
        "buyers_guide": [
            _chunk("bg1", SourceFamily.BUYERS_GUIDE),
            _chunk("bg2", SourceFamily.BUYERS_GUIDE),
            _chunk("bg3", SourceFamily.BUYERS_GUIDE),
        ],
        "tbs_directive": [
            _chunk("tbs1", SourceFamily.TBS_DIRECTIVE, doc_id="tbs_doc"),
            _chunk("tbs2", SourceFamily.TBS_DIRECTIVE, doc_id="tbs_doc"),
        ],
    }
    result = bg_primary_selective_authority_cluster(
        "How do I handle a late paper offer?",
        grouped,
        top_k=3,
    )
    assert [chunk.chunk_id for chunk in result] == ["bg1", "bg2", "bg3"]
