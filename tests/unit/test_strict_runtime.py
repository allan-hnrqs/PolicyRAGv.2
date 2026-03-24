from pathlib import Path
from types import SimpleNamespace

import pytest

import bgrag.pipeline as pipeline
from bgrag.config import Settings
from bgrag.pipeline import build_answer_callback, run_build_index
from bgrag.retrieval.retriever import HybridRetriever
from bgrag.types import AnswerResult, ChunkRecord, EvidenceBundle, NormalizedDocument, SourceFamily, SourceGraph

REPO_ROOT = Path(__file__).resolve().parents[2]


def _chunk(chunk_id: str = "bg1") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        canonical_url=f"https://example.com/{chunk_id}",
        title=chunk_id,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text="Example procurement content.",
    )


def _document(doc_id: str, canonical_url: str, *, parent_url: str | None = None, child_doc_ids: list[str] | None = None) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id=doc_id,
        title=doc_id,
        source_url=canonical_url,
        canonical_url=canonical_url,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        fetched_at="2026-03-23T00:00:00Z",
        content_hash=doc_id,
        word_count=10,
        extraction_method="test",
        graph=SourceGraph(
            parent_url=parent_url,
            parent_doc_id="parent" if parent_url else None,
            child_doc_ids=child_doc_ids or [],
        ),
        raw_text="test",
        markdown_text="test",
    )


def test_build_index_requires_cohere_key() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="")
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        run_build_index(settings, "baseline")


def test_lexical_search_requires_elasticsearch() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    retriever = HybridRetriever(settings, elastic=None)
    with pytest.raises(RuntimeError, match="Elasticsearch-backed lexical search"):
        retriever.lexical_search("buyer question", chunks=[_chunk()], top_k=5)


def test_retrieve_requires_embedding_store() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    retriever = HybridRetriever(settings, elastic=None)
    with pytest.raises(RuntimeError, match="populated chunk embedding store"):
        retriever.retrieve(
            question="buyer question",
            chunks=[_chunk()],
            query_embedding=[0.1, 0.2],
            chunk_embeddings={},
            source_topology="bg_primary_support_fallback",
            top_k=5,
            candidate_k=5,
            retrieval_alpha=0.7,
        )


def test_build_answer_callback_requires_complete_embedding_store(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    monkeypatch.setattr(pipeline, "load_index_manifest", lambda settings, namespace=None: {"namespace": "test"})
    monkeypatch.setattr(pipeline, "read_embedding_store", lambda path: {})
    with pytest.raises(RuntimeError, match="populated embedding store"):
        build_answer_callback(settings, "baseline", chunks=[_chunk()])


def test_build_answer_callback_rejects_partial_embedding_store(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    monkeypatch.setattr(pipeline, "load_index_manifest", lambda settings, namespace=None: {"namespace": "test"})
    monkeypatch.setattr(pipeline, "read_embedding_store", lambda path: {"other": [0.1, 0.2]})
    with pytest.raises(RuntimeError, match="embeddings for every loaded chunk"):
        build_answer_callback(settings, "baseline", chunks=[_chunk()])


def test_build_answer_callback_can_select_page_family_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    chunk = _chunk("doc1__section__7").model_copy(
        update={
            "doc_id": "doc1",
            "canonical_url": "https://example.com/workflow/a",
            "title": "Workflow A",
            "heading": "Deep rule",
            "heading_path": ["Workflow A", "Deep rule"],
            "order": 7,
        }
    )
    retrieval_calls: list[bool] = []

    profile = SimpleNamespace(
        answering=SimpleNamespace(strategy="inline_evidence_chat"),
        retrieval=SimpleNamespace(
            enable_query_decomposition=False,
            max_expanded_queries=2,
            source_topology="bg_primary_support_fallback",
            top_k=3,
            candidate_k=3,
            retrieval_alpha=0.7,
            rerank_top_n=0,
            enable_mmr_diversity=False,
            mmr_lambda=0.75,
            enable_ranked_chunk_diversity=False,
            diversity_cover_fraction=0.5,
            max_chunks_per_document=8,
            max_chunks_per_heading=4,
            seed_chunks_per_heading=2,
            query_fusion_rrf_k=60,
            per_query_candidate_k=24,
            enable_retrieval_mode_selection=True,
            retrieval_mode_selector_max_chunks=5,
            enable_page_intro_expansion=False,
            page_intro_candidate_k=8,
            page_intro_max_order=10,
            enable_document_context_expansion=False,
            document_context_seed_docs=2,
            document_context_candidate_k=12,
            document_context_neighbor_docs=2,
            enable_structural_context_augmentation=False,
            structural_context_seed_docs=2,
            structural_context_intro_max_order=10,
            structural_context_same_heading_k=2,
            structural_context_nearby_k=3,
            structural_context_nearby_window=12,
            structural_context_neighbor_docs=2,
            enable_document_seed_retrieval=True,
            document_seed_ranking_mode="rerank_docs",
            document_seed_scope="local_graph",
            document_seed_scope_docs=4,
            document_seed_docs=3,
            document_seed_intro_max_order=10,
            document_seed_intro_chunks=3,
            document_seed_candidate_k=12,
            document_seed_max_chars=1400,
        ),
    )

    monkeypatch.setattr(pipeline, "load_profile", lambda profile_name, settings: profile)
    monkeypatch.setattr(pipeline, "load_index_manifest", lambda settings, namespace=None: {"namespace": "test"})
    monkeypatch.setattr(pipeline, "read_embedding_store", lambda path: {chunk.chunk_id: [1.0, 0.0]})
    monkeypatch.setattr(pipeline, "read_normalized_documents", lambda path: [])
    monkeypatch.setattr(pipeline, "build_es_client", lambda settings: object())
    monkeypatch.setattr(pipeline, "require_es_available", lambda elastic, url: None)

    class _FakeEmbedder:
        def __init__(self, settings: Settings) -> None:
            del settings

        def embed_texts(self, texts: list[str], input_type: str):
            del texts, input_type
            return [[1.0, 0.0]]

    class _FakeRetriever:
        def __init__(self, settings: Settings, elastic=None, index_namespace=None, documents=None) -> None:
            del settings, elastic, index_namespace, documents

        def retrieve(self, **kwargs):
            retrieval_calls.append(bool(kwargs["enable_document_seed_retrieval"]))
            evidence = EvidenceBundle(
                query=kwargs["question"],
                packed_chunks=[chunk],
                retrieval_queries=list(kwargs["retrieval_queries"]),
                notes=[],
            )
            if kwargs["enable_document_seed_retrieval"]:
                evidence.notes.append("document_seed_retrieval_applied")
            return evidence

    class _FakeSelector:
        def __init__(self, settings: Settings) -> None:
            del settings

        def select(self, question: str, evidence: EvidenceBundle, *, max_chunks: int):
            del question, evidence, max_chunks
            return SimpleNamespace(mode="page_family_expansion", rationale="Need page family coverage.")

    monkeypatch.setattr(pipeline, "CohereEmbedder", _FakeEmbedder)
    monkeypatch.setattr(pipeline, "HybridRetriever", _FakeRetriever)
    monkeypatch.setattr(pipeline, "CohereRetrievalModeSelector", _FakeSelector)
    monkeypatch.setattr(
        pipeline.answer_strategy_registry,
        "get",
        lambda name: (
            lambda settings, question, evidence: AnswerResult(
                question=question,
                answer_text="test answer",
                strategy_name=name,
                model_name="command-a-03-2025",
                evidence_bundle=evidence,
            )
        ),
    )

    answer_callback = build_answer_callback(settings, "selective_localized_document_rerank_seed_retrieval", chunks=[chunk])
    result = answer_callback(SimpleNamespace(question="workflow question"))

    assert retrieval_calls == [False, True]
    assert "retrieval_mode_selected:page_family_expansion" in result.evidence_bundle.notes


def test_retrieve_uses_candidate_k_for_candidate_pool() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")

    class _SpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None)
            self.lexical_top_k: int | None = None
            self.rerank_input_count: int | None = None

        def lexical_search(self, question: str, chunks: list[ChunkRecord], top_k: int) -> dict[str, float]:
            self.lexical_top_k = top_k
            return {chunk.chunk_id: float(index + 1) for index, chunk in enumerate(chunks)}

        def rerank(self, question: str, candidates, top_n: int):
            self.rerank_input_count = len(candidates)
            return candidates

    retriever = _SpyRetriever(settings)
    chunks = [_chunk(f"bg{i}") for i in range(6)]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    evidence = retriever.retrieve(
        question="buyer question",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=2,
        candidate_k=4,
        retrieval_alpha=0.7,
        rerank_top_n=0,
    )

    assert retriever.lexical_top_k == 4
    assert retriever.rerank_input_count == 4
    assert len(evidence.packed_chunks) == 2


def test_mmr_reorder_prefers_diverse_candidate_after_first_pick() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    retriever = HybridRetriever(settings, elastic=None)
    candidates = [
        type("RC", (), {"chunk": _chunk("bg1"), "blended_score": 0.95})(),
        type("RC", (), {"chunk": _chunk("bg2"), "blended_score": 0.94})(),
        type("RC", (), {"chunk": _chunk("bg3"), "blended_score": 0.80})(),
    ]
    embeddings = {
        "bg1": [1.0, 0.0],
        "bg2": [0.99, 0.01],
        "bg3": [0.0, 1.0],
    }

    reordered = retriever.mmr_reorder(candidates, embeddings, mmr_lambda=0.75)

    assert [item.chunk.chunk_id for item in reordered[:3]] == ["bg1", "bg3", "bg2"]


def test_retrieve_can_expand_page_intro_and_document_context() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    parent_url = "https://example.com/workflow"
    parent = _document("parent", parent_url, child_doc_ids=["doc_a", "doc_b"])
    doc_a = _document("doc_a", "https://example.com/workflow/a", parent_url=parent_url)
    doc_b = _document("doc_b", "https://example.com/workflow/b", parent_url=parent_url)

    class _ExpansionSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None, documents=[parent, doc_a, doc_b])

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, top_k
            active_ids = allowed_chunk_ids or {chunk.chunk_id for chunk in chunks}
            scores = {chunk_id: 0.0 for chunk_id in active_ids}
            if "doc_a__section__20" in active_ids:
                scores["doc_a__section__20"] = 10.0
            if "doc_a__section__3" in active_ids:
                scores["doc_a__section__3"] = 9.0
            if "doc_b__section__2" in active_ids:
                scores["doc_b__section__2"] = 8.0
            return scores

        def rerank(self, question: str, candidates, top_n: int):
            del question, top_n
            return candidates

    retriever = _ExpansionSpyRetriever(settings)
    chunks = [
        _chunk("doc_a__section__20").model_copy(update={"doc_id": "doc_a", "order": 20, "text": "deep workflow rule"}),
        _chunk("doc_a__section__3").model_copy(update={"doc_id": "doc_a", "order": 3, "text": "intro workflow rule"}),
        _chunk("doc_b__section__2").model_copy(update={"doc_id": "doc_b", "order": 2, "text": "sibling workflow rule"}),
    ]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    evidence = retriever.retrieve(
        question="workflow question",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=3,
        candidate_k=1,
        retrieval_alpha=1.0,
        rerank_top_n=0,
        enable_page_intro_expansion=True,
        page_intro_candidate_k=2,
        page_intro_max_order=10,
        enable_document_context_expansion=True,
        document_context_seed_docs=1,
        document_context_candidate_k=2,
        document_context_neighbor_docs=1,
    )

    packed_ids = {chunk.chunk_id for chunk in evidence.packed_chunks}
    assert "doc_a__section__20" in packed_ids
    assert "doc_a__section__3" in packed_ids
    assert "doc_b__section__2" in packed_ids
    assert "page_intro_expansion_applied" in evidence.notes
    assert "document_context_expansion_applied" in evidence.notes


def test_retrieve_can_add_structural_context_companions() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    parent_url = "https://example.com/workflow"
    parent = _document("parent", parent_url, child_doc_ids=["doc_a", "doc_b"])
    doc_a = _document("doc_a", "https://example.com/workflow/a", parent_url=parent_url)
    doc_b = _document("doc_b", "https://example.com/workflow/b", parent_url=parent_url)

    class _StructuralSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None, documents=[parent, doc_a, doc_b])

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, chunks, top_k, allowed_chunk_ids
            return {"doc_a__section__20": 10.0}

        def rerank(self, question: str, candidates, top_n: int):
            del question, top_n
            return candidates

    retriever = _StructuralSpyRetriever(settings)
    chunks = [
        _chunk("doc_a__section__20").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 20,
                "heading": "Specific rule",
                "heading_path": ["Workflow", "Branch A", "Specific rule"],
                "text": "specific workflow rule about reissuing",
            }
        ),
        _chunk("doc_a__section__7").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 7,
                "heading": "Overview",
                "heading_path": ["Workflow", "Overview"],
                "text": "overview workflow start page",
            }
        ),
        _chunk("doc_a__section__18").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 18,
                "heading": "Branch A",
                "heading_path": ["Workflow", "Branch A", "Conditions"],
                "text": "nearby workflow condition about reissuing",
            }
        ),
        _chunk("doc_b__section__2").model_copy(
            update={
                "doc_id": "doc_b",
                "canonical_url": "https://example.com/workflow/b",
                "order": 2,
                "heading": "Sibling overview",
                "heading_path": ["Workflow", "Sibling overview"],
                "text": "sibling page overview for reissued documents",
            }
        ),
    ]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    evidence = retriever.retrieve(
        question="workflow question about reissuing documents",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=4,
        candidate_k=1,
        retrieval_alpha=1.0,
        rerank_top_n=0,
        enable_structural_context_augmentation=True,
        structural_context_seed_docs=1,
        structural_context_intro_max_order=10,
        structural_context_same_heading_k=1,
        structural_context_nearby_k=1,
        structural_context_nearby_window=5,
        structural_context_neighbor_docs=1,
    )

    packed_ids = {chunk.chunk_id for chunk in evidence.packed_chunks}
    assert "doc_a__section__20" in packed_ids
    assert "doc_a__section__7" in packed_ids
    assert "doc_a__section__18" in packed_ids
    assert "doc_b__section__2" in packed_ids
    assert "structural_context_augmentation_applied" in evidence.notes


def test_retrieve_can_seed_pages_from_intro_context() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    doc_a = _document("doc_a", "https://example.com/workflow/a")
    doc_b = _document("doc_b", "https://example.com/workflow/b")

    class _PageSeedSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None, documents=[doc_a, doc_b])

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, chunks, top_k
            active_ids = allowed_chunk_ids or set()
            scores = {chunk_id: 0.0 for chunk_id in active_ids}
            if "doc_a__section__0" in active_ids:
                scores["doc_a__section__0"] = 9.0
            if "doc_a__section__5" in active_ids:
                scores["doc_a__section__5"] = 7.0
            if "doc_b__section__0" in active_ids:
                scores["doc_b__section__0"] = 3.0
            return scores

        def rerank(self, question: str, candidates, top_n: int):
            del question, top_n
            return candidates

    retriever = _PageSeedSpyRetriever(settings)
    chunks = [
        _chunk("doc_a__section__0").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 0,
                "heading": "Overview",
                "heading_path": ["Workflow", "Overview"],
                "text": "overview of reissuing solicitations",
            }
        ),
        _chunk("doc_a__section__5").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 5,
                "heading": "Documents",
                "heading_path": ["Workflow", "Documents"],
                "text": "reissued solicitation documents and notice",
            }
        ),
        _chunk("doc_a__section__20").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 20,
                "heading": "Deep detail",
                "heading_path": ["Workflow", "Deep detail"],
                "text": "deep detail about reissued files",
            }
        ),
        _chunk("doc_b__section__0").model_copy(
            update={
                "doc_id": "doc_b",
                "canonical_url": "https://example.com/workflow/b",
                "order": 0,
                "heading": "Other overview",
                "heading_path": ["Other", "Overview"],
                "text": "other unrelated page",
            }
        ),
    ]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    evidence = retriever.retrieve(
        question="when should we reissue the solicitation and what goes in the documents",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=3,
        candidate_k=1,
        retrieval_alpha=1.0,
        rerank_top_n=0,
        enable_document_seed_retrieval=True,
        document_seed_docs=1,
        document_seed_intro_max_order=10,
        document_seed_intro_chunks=2,
        document_seed_candidate_k=3,
    )

    packed_ids = {chunk.chunk_id for chunk in evidence.packed_chunks}
    assert "doc_a__section__0" in packed_ids
    assert "doc_a__section__5" in packed_ids
    assert "doc_a__section__20" in packed_ids
    assert "document_seed_retrieval_applied" in evidence.notes


def test_retrieve_can_seed_pages_with_document_rerank() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    doc_a = _document("doc_a", "https://example.com/workflow/a")
    doc_b = _document("doc_b", "https://example.com/workflow/b")

    class _DocumentRerankSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None, documents=[doc_a, doc_b])
            self.rerank_client = SimpleNamespace(
                rerank=lambda **kwargs: SimpleNamespace(
                    results=[
                        SimpleNamespace(index=0, relevance_score=0.95),
                        SimpleNamespace(index=1, relevance_score=0.25),
                    ]
                )
            )

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, chunks, top_k
            active_ids = allowed_chunk_ids or set()
            scores = {chunk_id: 0.0 for chunk_id in active_ids}
            if "doc_a__section__0" in active_ids:
                scores["doc_a__section__0"] = 8.0
            if "doc_a__section__5" in active_ids:
                scores["doc_a__section__5"] = 7.0
            return scores

        def rerank(self, question: str, candidates, top_n: int):
            del question, top_n
            return candidates

    retriever = _DocumentRerankSpyRetriever(settings)
    chunks = [
        _chunk("doc_a__section__0").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 0,
                "heading": "Overview",
                "heading_path": ["Workflow", "Overview"],
                "text": "overview of reissuing solicitations",
            }
        ),
        _chunk("doc_a__section__5").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 5,
                "heading": "Documents",
                "heading_path": ["Workflow", "Documents"],
                "text": "reissued solicitation documents and notice",
            }
        ),
        _chunk("doc_a__section__20").model_copy(
            update={
                "doc_id": "doc_a",
                "canonical_url": "https://example.com/workflow/a",
                "order": 20,
                "heading": "Deep detail",
                "heading_path": ["Workflow", "Deep detail"],
                "text": "deep detail about reissued files",
            }
        ),
        _chunk("doc_b__section__0").model_copy(
            update={
                "doc_id": "doc_b",
                "canonical_url": "https://example.com/workflow/b",
                "order": 0,
                "heading": "Other overview",
                "heading_path": ["Other", "Overview"],
                "text": "other unrelated page",
            }
        ),
    ]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    evidence = retriever.retrieve(
        question="when should we reissue the solicitation and what goes in the documents",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=3,
        candidate_k=1,
        retrieval_alpha=1.0,
        rerank_top_n=0,
        enable_document_seed_retrieval=True,
        document_seed_ranking_mode="rerank_docs",
        document_seed_docs=1,
        document_seed_intro_max_order=10,
        document_seed_intro_chunks=2,
        document_seed_candidate_k=3,
        document_seed_max_chars=400,
    )

    packed_ids = {chunk.chunk_id for chunk in evidence.packed_chunks}
    assert "doc_a__section__0" in packed_ids
    assert "doc_a__section__5" in packed_ids
    assert "doc_a__section__20" in packed_ids
    assert "document_seed_retrieval_applied" in evidence.notes


def test_document_rerank_seed_can_scope_to_local_graph() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    parent_url = "https://example.com/workflow"
    parent = _document("parent", parent_url, child_doc_ids=["doc_a", "doc_b"])
    doc_a = _document("doc_a", "https://example.com/workflow/a", parent_url=parent_url)
    doc_b = _document("doc_b", "https://example.com/workflow/b", parent_url=parent_url)
    doc_c = _document("doc_c", "https://example.com/unrelated/c")
    captured_documents: list[str] = []

    class _ScopedDocumentRerankSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None, documents=[parent, doc_a, doc_b, doc_c])
            self.rerank_client = SimpleNamespace(
                rerank=lambda **kwargs: self._fake_rerank(**kwargs)
            )

        def _fake_rerank(self, **kwargs):
            captured_documents[:] = list(kwargs["documents"])
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=1, relevance_score=0.95),
                    SimpleNamespace(index=0, relevance_score=0.80),
                ]
            )

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, chunks, top_k
            active_ids = allowed_chunk_ids or set()
            scores = {chunk_id: 0.0 for chunk_id in active_ids}
            if "doc_a__section__20" in active_ids:
                scores["doc_a__section__20"] = 9.0
            if "doc_a__section__0" in active_ids:
                scores["doc_a__section__0"] = 7.0
            if "doc_b__section__0" in active_ids:
                scores["doc_b__section__0"] = 6.0
            if "doc_c__section__0" in active_ids:
                scores["doc_c__section__0"] = 5.0
            return scores

        def rerank(self, question: str, candidates, top_n: int):
            del question, top_n
            return candidates

    retriever = _ScopedDocumentRerankSpyRetriever(settings)
    chunks = [
        _chunk("doc_a__section__20").model_copy(update={"doc_id": "doc_a", "canonical_url": "https://example.com/workflow/a", "order": 20, "text": "deep workflow detail"}),
        _chunk("doc_a__section__0").model_copy(update={"doc_id": "doc_a", "canonical_url": "https://example.com/workflow/a", "order": 0, "text": "doc a overview"}),
        _chunk("doc_b__section__0").model_copy(update={"doc_id": "doc_b", "canonical_url": "https://example.com/workflow/b", "order": 0, "text": "doc b sibling overview"}),
        _chunk("doc_c__section__0").model_copy(update={"doc_id": "doc_c", "canonical_url": "https://example.com/unrelated/c", "order": 0, "text": "unrelated overview"}),
    ]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    evidence = retriever.retrieve(
        question="workflow question",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=3,
        candidate_k=1,
        retrieval_alpha=1.0,
        rerank_top_n=0,
        enable_document_seed_retrieval=True,
        document_seed_ranking_mode="rerank_docs",
        document_seed_scope="local_graph",
        document_seed_scope_docs=1,
        document_seed_docs=2,
        document_seed_intro_max_order=10,
        document_seed_intro_chunks=1,
        document_seed_candidate_k=3,
        document_seed_max_chars=400,
    )

    assert len(captured_documents) == 2
    packed_ids = {chunk.chunk_id for chunk in evidence.packed_chunks}
    assert "doc_b__section__0" in packed_ids
    assert "doc_c__section__0" not in packed_ids


def test_document_rerank_seed_can_scope_to_local_lineage_only() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    parent_url = "https://example.com/workflow"
    parent = _document("parent", parent_url, child_doc_ids=["doc_a", "doc_b"])
    doc_a = _document("doc_a", "https://example.com/workflow/a", parent_url=parent_url)
    doc_a.graph.outgoing_in_scope_links = ["https://example.com/unrelated/c"]
    doc_b = _document("doc_b", "https://example.com/workflow/b", parent_url=parent_url)
    doc_c = _document("doc_c", "https://example.com/unrelated/c")
    captured_documents: list[str] = []

    class _LineageDocumentRerankSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None, documents=[parent, doc_a, doc_b, doc_c])
            self.rerank_client = SimpleNamespace(
                rerank=lambda **kwargs: self._fake_rerank(**kwargs)
            )

        def _fake_rerank(self, **kwargs):
            captured_documents[:] = list(kwargs["documents"])
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=1, relevance_score=0.95),
                    SimpleNamespace(index=0, relevance_score=0.80),
                ]
            )

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, chunks, top_k
            active_ids = allowed_chunk_ids or set()
            scores = {chunk_id: 0.0 for chunk_id in active_ids}
            if "doc_a__section__20" in active_ids:
                scores["doc_a__section__20"] = 9.0
            if "doc_a__section__0" in active_ids:
                scores["doc_a__section__0"] = 7.0
            if "doc_b__section__0" in active_ids:
                scores["doc_b__section__0"] = 6.0
            if "doc_c__section__0" in active_ids:
                scores["doc_c__section__0"] = 5.0
            return scores

        def rerank(self, question: str, candidates, top_n: int):
            del question, top_n
            return candidates

    retriever = _LineageDocumentRerankSpyRetriever(settings)
    chunks = [
        _chunk("doc_a__section__20").model_copy(update={"doc_id": "doc_a", "canonical_url": "https://example.com/workflow/a", "order": 20, "text": "deep workflow detail"}),
        _chunk("doc_a__section__0").model_copy(update={"doc_id": "doc_a", "canonical_url": "https://example.com/workflow/a", "order": 0, "text": "doc a overview"}),
        _chunk("doc_b__section__0").model_copy(update={"doc_id": "doc_b", "canonical_url": "https://example.com/workflow/b", "order": 0, "text": "doc b sibling overview"}),
        _chunk("doc_c__section__0").model_copy(update={"doc_id": "doc_c", "canonical_url": "https://example.com/unrelated/c", "order": 0, "text": "unrelated overview"}),
    ]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    evidence = retriever.retrieve(
        question="workflow question",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=3,
        candidate_k=1,
        retrieval_alpha=1.0,
        rerank_top_n=0,
        enable_document_seed_retrieval=True,
        document_seed_ranking_mode="rerank_docs",
        document_seed_scope="local_lineage",
        document_seed_scope_docs=1,
        document_seed_docs=2,
        document_seed_intro_max_order=10,
        document_seed_intro_chunks=1,
        document_seed_candidate_k=3,
        document_seed_max_chars=400,
    )

    assert len(captured_documents) == 2
    packed_ids = {chunk.chunk_id for chunk in evidence.packed_chunks}
    assert "doc_b__section__0" in packed_ids
    assert "doc_c__section__0" not in packed_ids
