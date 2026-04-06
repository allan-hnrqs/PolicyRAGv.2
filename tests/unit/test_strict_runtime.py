from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import bgrag.pipeline as pipeline
from bgrag.config import Settings
from bgrag.pipeline import build_answer_callback, run_build_index
from bgrag.retrieval.retriever import HybridRetriever, _format_rerank_document
from bgrag.types import (
    AnswerResult,
    ChunkRecord,
    EvidenceBundle,
    NormalizedDocument,
    RetrievalAssessment,
    RetrievalCandidate,
    SourceFamily,
    SourceGraph,
)

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


def test_build_index_indexes_vectors_into_elasticsearch(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    captured: dict[str, object] = {}
    chunk = _chunk("bg1")

    monkeypatch.setattr(
        pipeline,
        "load_profile",
        lambda profile_name, settings: SimpleNamespace(retrieval=SimpleNamespace(source_topology="bg_primary_support_fallback")),
    )
    monkeypatch.setattr(pipeline, "read_chunks", lambda path: [chunk])
    monkeypatch.setattr(pipeline, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(pipeline, "require_search_available", lambda client, settings, backend: None)
    monkeypatch.setattr(
        pipeline,
        "derive_index_namespace",
        lambda settings, profile_name: "phase1_test_namespace",
    )
    monkeypatch.setattr(pipeline, "write_embedding_store", lambda path, vectors: captured.setdefault("vectors", vectors))
    monkeypatch.setattr(pipeline, "write_index_manifest", lambda settings, namespace, manifest: None)
    monkeypatch.setattr(pipeline, "set_active_index_namespace", lambda settings, namespace: None)

    class _FakeEmbedder:
        def __init__(self, settings: Settings) -> None:
            del settings

        def embed_texts(self, texts: list[str], input_type: str):
            assert texts == [chunk.text]
            assert input_type == "search_document"
            return [[0.1, 0.2]]

    monkeypatch.setattr(pipeline, "CohereEmbedder", _FakeEmbedder)

    def _capture_index_chunks(elastic, chunks, namespace, *, embeddings=None, backend=None):
        captured["index_chunks"] = {
            "elastic": elastic,
            "chunks": chunks,
            "namespace": namespace,
            "embeddings": embeddings,
            "backend": backend,
        }

    monkeypatch.setattr(pipeline, "index_chunks_for_backend", _capture_index_chunks)

    stats = run_build_index(settings, "baseline")

    assert stats["embedding_count"] == 1
    assert captured["vectors"] == {chunk.chunk_id: [0.1, 0.2]}
    assert captured["index_chunks"]["namespace"] == "phase1_test_namespace"
    assert captured["index_chunks"]["embeddings"] == {chunk.chunk_id: [0.1, 0.2]}
    assert captured["index_chunks"]["backend"] == "elasticsearch"


def test_lexical_search_requires_elasticsearch() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    retriever = HybridRetriever(settings, elastic=None)
    with pytest.raises(RuntimeError, match="Elasticsearch-backed lexical search"):
        retriever.lexical_search("buyer question", chunks=[_chunk()], top_k=5)


def test_retrieve_requires_embedding_store() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    retriever = HybridRetriever(settings, elastic=None)
    with pytest.raises(RuntimeError, match="requires a populated chunk embedding store"):
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


def test_retrieve_es_knn_path_does_not_require_embedding_store() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")

    class _EsOnlyRetriever(HybridRetriever):
        def lexical_search(self, question: str, chunks: list[ChunkRecord], top_k: int, **kwargs) -> dict[str, float]:
            del question, top_k, kwargs
            return {chunks[0].chunk_id: 0.4}

        def vector_search(
            self,
            query_embedding,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            num_candidates: int,
            allowed_chunk_ids=None,
        ) -> dict[str, float]:
            del query_embedding, top_k, num_candidates, allowed_chunk_ids
            return {chunks[0].chunk_id: 0.9}

    retriever = _EsOnlyRetriever(settings, elastic=object())
    evidence = retriever.retrieve(
        question="buyer question",
        chunks=[_chunk()],
        query_embedding=[0.1, 0.2],
        chunk_embeddings=None,
        source_topology="unified_source_hybrid",
        top_k=1,
        candidate_k=1,
        retrieval_alpha=0.7,
        dense_retrieval_backend="elasticsearch_knn",
        rerank_top_n=0,
    )

    assert [chunk.chunk_id for chunk in evidence.packed_chunks] == ["bg1"]
    assert [candidate.chunk.chunk_id for candidate in evidence.raw_shortlist] == ["bg1"]


def test_ranked_passthrough_preserves_cross_family_rank_order() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    bg = _chunk("bg1")
    support = _chunk("bc1").model_copy(update={"source_family": SourceFamily.BUY_CANADIAN_POLICY})

    class _RankedRetriever(HybridRetriever):
        def lexical_search(self, question: str, chunks: list[ChunkRecord], top_k: int, **kwargs) -> dict[str, float]:
            del question, top_k, kwargs
            return {bg.chunk_id: 0.1, support.chunk_id: 0.9}

        def vector_search(
            self,
            query_embedding,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            num_candidates: int,
            allowed_chunk_ids=None,
        ) -> dict[str, float]:
            del query_embedding, top_k, num_candidates, allowed_chunk_ids
            return {bg.chunk_id: 0.1, support.chunk_id: 0.9}

    retriever = _RankedRetriever(settings, elastic=object())
    evidence = retriever.retrieve(
        question="buyer question",
        chunks=[bg, support],
        query_embedding=[0.1, 0.2],
        chunk_embeddings=None,
        source_topology="ranked_passthrough",
        top_k=2,
        candidate_k=2,
        retrieval_alpha=0.7,
        dense_retrieval_backend="elasticsearch_knn",
        rerank_top_n=0,
    )

    assert [candidate.chunk.chunk_id for candidate in evidence.raw_shortlist] == ["bc1", "bg1"]
    assert [chunk.chunk_id for chunk in evidence.packed_chunks] == ["bc1", "bg1"]


def test_build_answer_callback_requires_complete_embedding_store(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    monkeypatch.setattr(pipeline, "load_index_manifest", lambda settings, namespace=None: {"namespace": "test"})
    monkeypatch.setattr(pipeline, "read_embedding_store", lambda path: {})
    with pytest.raises(RuntimeError, match="requires a populated embedding store"):
        build_answer_callback(settings, "baseline", chunks=[_chunk()])


def test_build_answer_callback_rejects_partial_embedding_store(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    monkeypatch.setattr(pipeline, "load_index_manifest", lambda settings, namespace=None: {"namespace": "test"})
    monkeypatch.setattr(pipeline, "read_embedding_store", lambda path: {"other": [0.1, 0.2]})
    with pytest.raises(RuntimeError, match="requires embeddings for every loaded chunk"):
        build_answer_callback(settings, "baseline", chunks=[_chunk()])


def test_build_answer_callback_allows_es_knn_without_embedding_store(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    chunk = _chunk()
    profile = SimpleNamespace(
        answering=SimpleNamespace(strategy="inline_evidence_chat", max_packed_docs=4),
        retrieval=SimpleNamespace(
            enable_query_decomposition=False,
            max_expanded_queries=0,
            source_topology="unified_source_hybrid",
            top_k=1,
            candidate_k=1,
            retrieval_mode="hybrid_es_rerank",
            retrieval_alpha=0.7,
            rerank_top_n=0,
            dense_retrieval_backend="elasticsearch_knn",
            es_knn_num_candidates=8,
            enable_mmr_diversity=False,
            mmr_lambda=0.75,
            enable_ranked_chunk_diversity=False,
            diversity_cover_fraction=0.5,
            max_chunks_per_document=8,
            max_chunks_per_heading=4,
            seed_chunks_per_heading=2,
            query_fusion_rrf_k=60,
            per_query_candidate_k=1,
            enable_parallel_query_branches=False,
            enable_page_intro_expansion=False,
            page_intro_candidate_k=0,
            page_intro_max_order=0,
            enable_document_context_expansion=False,
            document_context_seed_docs=0,
            document_context_candidate_k=0,
            document_context_neighbor_docs=0,
            enable_structural_context_augmentation=False,
            structural_context_seed_docs=0,
            structural_context_intro_max_order=0,
            structural_context_same_heading_k=0,
            structural_context_nearby_k=0,
            structural_context_nearby_window=0,
            structural_context_neighbor_docs=0,
            enable_document_seed_retrieval=False,
            document_seed_ranking_mode="intro_pool",
            document_seed_scope="corpus",
            document_seed_scope_docs=0,
            document_seed_docs=0,
            document_seed_intro_max_order=0,
            document_seed_intro_chunks=0,
            document_seed_candidate_k=0,
            document_seed_max_chars=0,
            enable_retrieval_mode_selection=False,
        ),
    )

    monkeypatch.setattr(pipeline, "load_profile", lambda profile_name, settings: profile)
    monkeypatch.setattr(pipeline, "load_index_manifest", lambda settings, namespace=None: {"namespace": "test"})
    monkeypatch.setattr(pipeline, "read_embedding_store", lambda path: {})
    monkeypatch.setattr(pipeline, "read_normalized_documents", lambda path: [])
    monkeypatch.setattr(pipeline, "build_runtime_settings", lambda settings, profile: settings)
    monkeypatch.setattr(pipeline, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(pipeline, "require_search_available", lambda client, settings, backend: None)
    monkeypatch.setattr(
        pipeline,
        "answer_strategy_registry",
        SimpleNamespace(
            get=lambda name: (
                lambda settings, question, evidence, *, persona=None: AnswerResult(
                    question=question,
                    answer_text="answer",
                    strategy_name=name,
                    model_name="command-a-03-2025",
                    evidence_bundle=evidence,
                )
            )
        ),
    )

    class _StubEmbedder:
        def __init__(self, settings: Settings) -> None:
            del settings

        def embed_texts(self, texts, input_type="search_query"):
            del texts, input_type
            return [[0.1, 0.2]]

    class _StubRetriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, **kwargs):
            assert kwargs["chunk_embeddings"] == {}
            return EvidenceBundle(
                query=kwargs["question"],
                raw_shortlist=[RetrievalCandidate(chunk=chunk)],
                selected_candidates=[RetrievalCandidate(chunk=chunk)],
                candidates=[RetrievalCandidate(chunk=chunk)],
                packed_chunks=[chunk],
            )

    monkeypatch.setattr(pipeline, "CohereEmbedder", _StubEmbedder)
    monkeypatch.setattr(pipeline, "HybridRetriever", _StubRetriever)

    answer_callback = build_answer_callback(settings, "baseline_vector", chunks=[chunk])
    result = answer_callback(SimpleNamespace(question="buyer question"))

    assert result.evidence_bundle is not None
    assert [packed.chunk_id for packed in result.evidence_bundle.packed_chunks] == [chunk.chunk_id]


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
            dense_retrieval_backend="local_embedding_store",
            es_knn_num_candidates=12,
            enable_mmr_diversity=False,
            mmr_lambda=0.75,
            enable_ranked_chunk_diversity=False,
            diversity_cover_fraction=0.5,
            max_chunks_per_document=8,
            max_chunks_per_heading=4,
            seed_chunks_per_heading=2,
            query_fusion_rrf_k=60,
            per_query_candidate_k=24,
            enable_parallel_query_branches=False,
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
    monkeypatch.setattr(pipeline, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(pipeline, "require_search_available", lambda client, settings, backend: None)

    class _FakeEmbedder:
        def __init__(self, settings: Settings) -> None:
            del settings

        def embed_texts(self, texts: list[str], input_type: str):
            del texts, input_type
            return [[1.0, 0.0]]

    class _FakeRetriever:
        def __init__(self, settings: Settings, elastic=None, index_namespace=None, documents=None, search_backend=None) -> None:
            del settings, elastic, index_namespace, documents, search_backend

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


def test_retrieve_can_use_elasticsearch_knn_backend_and_records_stage_timings() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")

    class _VectorSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=object())
            self.vector_calls = 0

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, chunks, top_k, allowed_chunk_ids
            return {"bg1": 4.0}

        def vector_search(
            self,
            query_embedding,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            num_candidates: int,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del query_embedding, chunks, top_k, num_candidates, allowed_chunk_ids
            self.vector_calls += 1
            return {"bg2": 0.92}

        def rerank(self, question: str, candidates, top_n: int):
            del question, top_n
            return candidates

    retriever = _VectorSpyRetriever(settings)
    chunks = [_chunk("bg1"), _chunk("bg2"), _chunk("bg3")]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}

    evidence = retriever.retrieve(
        question="buyer question",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=2,
        candidate_k=2,
        retrieval_alpha=0.7,
        rerank_top_n=0,
        dense_retrieval_backend="elasticsearch_knn",
        es_knn_num_candidates=16,
    )

    packed_ids = {chunk.chunk_id for chunk in evidence.packed_chunks}
    assert retriever.vector_calls == 1
    assert packed_ids == {"bg1", "bg2"}
    assert set(evidence.timings) >= {
        "lexical_search_seconds",
        "vector_search_seconds",
        "candidate_fusion_seconds",
        "rerank_seconds",
        "packing_seconds",
    }


def test_opensearch_lexical_search_uses_body_query_shape() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    chunk = _chunk("bg1")
    captured: dict[str, object] = {}

    class _Indices:
        def exists(self, index: str) -> bool:
            captured["index"] = index
            return True

    class _Client:
        def __init__(self) -> None:
            self.indices = _Indices()

        def search(self, *, index, body):
            captured["search_index"] = index
            captured["body"] = body
            return {"hits": {"hits": [{"_id": chunk.chunk_id, "_score": 3.5}]}}

    retriever = HybridRetriever(settings, elastic=_Client(), search_backend="opensearch")
    scores = retriever.lexical_search("buyer question", chunks=[chunk], top_k=5)

    assert scores == {chunk.chunk_id: 3.5}
    assert captured["search_index"] == "bgrag_chunks_default_buyers_guide"
    assert captured["body"]["size"] == 5
    assert captured["body"]["query"]["multi_match"]["query"] == "buyer question"


def test_opensearch_vector_search_uses_knn_query_body_shape() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    chunk = _chunk("bg1")
    captured: dict[str, object] = {}

    class _Indices:
        def exists(self, index: str) -> bool:
            captured["index"] = index
            return True

    class _Client:
        def __init__(self) -> None:
            self.indices = _Indices()

        def search(self, *, index, body):
            captured["search_index"] = index
            captured["body"] = body
            return {"hits": {"hits": [{"_id": chunk.chunk_id, "_score": 0.95}]}}

    retriever = HybridRetriever(settings, elastic=_Client(), search_backend="opensearch")
    scores = retriever.vector_search(
        [0.1, 0.2],
        chunks=[chunk],
        top_k=4,
        num_candidates=12,
    )

    assert chunk.chunk_id in scores
    assert captured["search_index"] == "bgrag_chunks_default_buyers_guide"
    assert captured["body"]["size"] == 4
    assert captured["body"]["query"]["knn"]["embedding"]["k"] == 4
    assert captured["body"]["query"]["knn"]["embedding"]["vector"] == [0.1, 0.2]


def test_rerank_top_n_zero_does_not_call_rerank() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")

    class _NoRerankRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None)

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, top_k, allowed_chunk_ids
            return {chunk.chunk_id: float(index + 1) for index, chunk in enumerate(chunks)}

        def rerank(self, question: str, candidates, top_n: int):
            raise AssertionError("rerank should not be called when rerank_top_n is 0")

    retriever = _NoRerankRetriever(settings)
    chunks = [_chunk("bg1"), _chunk("bg2")]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}

    evidence = retriever.retrieve(
        question="buyer question",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=2,
        candidate_k=2,
        retrieval_alpha=1.0,
        rerank_top_n=0,
    )

    assert evidence.timings["rerank_seconds"] == 0.0
    assert "cohere_rerank_applied" not in evidence.notes


def test_shortlist_rerank_uses_structured_rerank_documents() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")

    class _FakeRerankClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def rerank(self, *, model: str, query: str, documents: list[str], top_n: int):
            self.calls.append(
                {
                    "model": model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": top_n,
                }
            )

            class _Result:
                def __init__(self, index: int, relevance_score: float) -> None:
                    self.index = index
                    self.relevance_score = relevance_score

            class _Response:
                def __init__(self) -> None:
                    self.results = [_Result(1, 0.9), _Result(0, 0.4)]

            return _Response()

    retriever = HybridRetriever(settings, elastic=None)
    fake_client = _FakeRerankClient()
    retriever.rerank_client = fake_client
    first = RetrievalCandidate(chunk=_chunk("bg1"), blended_score=0.7)
    second = RetrievalCandidate(
        chunk=_chunk("bg2").model_copy(
            update={
                "heading": "Deep rule",
                "heading_path": ["Workflow A", "Deep rule"],
                "canonical_url": "https://example.com/workflow/a",
                "text": "Line one.\nLine two.",
            }
        ),
        blended_score=0.6,
    )

    reranked = retriever.rerank("buyer question", [first, second], top_n=2)

    assert fake_client.calls
    documents = fake_client.calls[0]["documents"]
    assert isinstance(documents, list)
    assert "title: bg1" in documents[0]
    assert "text: |" in documents[0]
    assert "heading_path: Workflow A > Deep rule" in documents[1]
    assert reranked[0].chunk.chunk_id == "bg2"
    assert reranked[0].rerank_score == 0.9


def test_parallel_query_branches_can_run_off_main_thread() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    main_thread_id = threading.get_ident()
    thread_ids: set[int] = set()

    class _ParallelSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=None)

        def _score_query_candidates(self, **kwargs):
            del kwargs
            thread_ids.add(threading.get_ident())
            return [], {
                "lexical_search_seconds": 0.0,
                "vector_search_seconds": 0.0,
                "candidate_fusion_seconds": 0.0,
            }

    retriever = _ParallelSpyRetriever(settings)
    retriever._retrieve_multi_query_candidates(
        retrieval_queries=["q1", "q2", "q3"],
        query_embeddings=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        chunks=[_chunk("bg1")],
        chunk_embeddings={"bg1": [1.0, 0.0]},
        retrieval_alpha=0.7,
        candidate_k=4,
        per_query_candidate_k=4,
        query_fusion_rrf_k=60,
        dense_retrieval_backend="local_embedding_store",
        es_knn_num_candidates=12,
        enable_parallel_query_branches=True,
        stage_timings={},
    )

    assert thread_ids
    assert any(thread_id != main_thread_id for thread_id in thread_ids)


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
    assert retriever.rerank_input_count is None
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


def test_format_rerank_document_includes_chunk_metadata() -> None:
    chunk = _chunk("bg1").model_copy(
        update={
            "heading": "Deep rule",
            "heading_path": ["Workflow A", "Deep rule"],
            "canonical_url": "https://example.com/workflow/a",
            "text": "Line one.\nLine two.",
        }
    )

    rendered = _format_rerank_document(chunk)

    assert "title: bg1" in rendered
    assert "heading: Deep rule" in rendered
    assert "heading_path: Workflow A > Deep rule" in rendered
    assert "source_family: buyers_guide" in rendered
    assert "canonical_url: https://example.com/workflow/a" in rendered
    assert "text: |" in rendered
    assert "  Line one." in rendered
    assert "  Line two." in rendered


def test_retrieve_can_use_rerank_all_corpus_mode_without_lexical_or_dense_search() -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")

    class _FullCorpusRerankSpyRetriever(HybridRetriever):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings, elastic=object())
            self.calls: list[tuple[str, int, list[str]]] = []

        def lexical_search(
            self,
            question: str,
            chunks: list[ChunkRecord],
            top_k: int,
            *,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del question, chunks, top_k, allowed_chunk_ids
            raise AssertionError("lexical_search should not run in rerank_all_corpus mode")

        def dense_search(
            self,
            *,
            query_embedding,
            chunks: list[ChunkRecord],
            chunk_embeddings,
            top_k: int,
            dense_retrieval_backend: str,
            es_knn_num_candidates: int,
            allowed_chunk_ids: set[str] | None = None,
        ) -> dict[str, float]:
            del query_embedding, chunks, chunk_embeddings, top_k, dense_retrieval_backend, es_knn_num_candidates, allowed_chunk_ids
            raise AssertionError("dense_search should not run in rerank_all_corpus mode")

        def rerank_all_corpus(self, question: str, chunks: list[ChunkRecord], top_n: int) -> list[RetrievalCandidate]:
            self.calls.append((question, top_n, [chunk.chunk_id for chunk in chunks]))
            return [
                RetrievalCandidate(chunk=chunks[1], rerank_score=0.91, blended_score=0.91),
                RetrievalCandidate(chunk=chunks[0], rerank_score=0.74, blended_score=0.74),
            ]

    retriever = _FullCorpusRerankSpyRetriever(settings)
    chunks = [_chunk("bg1"), _chunk("bg2"), _chunk("bg3")]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}

    evidence = retriever.retrieve(
        question="buyer question",
        chunks=chunks,
        query_embedding=[1.0, 0.0],
        chunk_embeddings=embeddings,
        source_topology="bg_primary_support_fallback",
        top_k=2,
        candidate_k=2,
        retrieval_mode="rerank_all_corpus",
        retrieval_alpha=0.7,
        rerank_top_n=0,
    )

    assert retriever.calls == [("buyer question", 2, ["bg1", "bg2", "bg3"])]
    assert [candidate.chunk.chunk_id for candidate in evidence.candidates] == ["bg2", "bg1"]
    assert [chunk.chunk_id for chunk in evidence.packed_chunks] == ["bg2", "bg1"]
    assert "full_corpus_rerank_retrieval_applied" in evidence.notes
    assert evidence.timings["lexical_search_seconds"] == 0.0
    assert evidence.timings["vector_search_seconds"] == 0.0
    assert evidence.timings["candidate_fusion_seconds"] == 0.0
    assert evidence.timings["rerank_seconds"] >= 0.0


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


def test_build_answer_callback_hybrid_retry_trigger_can_force_one_indexed_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(project_root=REPO_ROOT, cohere_api_key="test-key")
    chunk = _chunk("doc_a__section__0").model_copy(
        update={
            "doc_id": "doc_a",
            "canonical_url": "https://example.com/doc_a",
            "heading": "Intro",
            "heading_path": ["Workflow", "Intro"],
        }
    )
    sibling = _chunk("doc_b__section__0").model_copy(
        update={
            "doc_id": "doc_b",
            "canonical_url": "https://example.com/doc_b",
            "heading": "Rule",
            "heading_path": ["Workflow", "Rule"],
        }
    )
    raw_shortlist = [
        RetrievalCandidate(chunk=chunk),
        RetrievalCandidate(chunk=sibling),
        RetrievalCandidate(chunk=_chunk("doc_c__section__0").model_copy(update={"doc_id": "doc_c", "heading": "A"})),
        RetrievalCandidate(chunk=_chunk("doc_d__section__0").model_copy(update={"doc_id": "doc_d", "heading": "B"})),
        RetrievalCandidate(chunk=_chunk("doc_e__section__0").model_copy(update={"doc_id": "doc_e", "heading": "C"})),
        RetrievalCandidate(chunk=_chunk("doc_f__section__0").model_copy(update={"doc_id": "doc_f", "heading": "D"})),
        RetrievalCandidate(chunk=_chunk("doc_g__section__0").model_copy(update={"doc_id": "doc_g", "heading": "E"})),
        RetrievalCandidate(chunk=_chunk("doc_h__section__0").model_copy(update={"doc_id": "doc_h", "heading": "F"})),
    ]
    first_bundle = EvidenceBundle(
        query="workflow question",
        raw_shortlist=raw_shortlist,
        selected_candidates=raw_shortlist,
        candidates=raw_shortlist,
        packed_chunks=[chunk, chunk.model_copy(update={"chunk_id": "doc_a__section__1"})],
    )
    second_bundle = EvidenceBundle(
        query="workflow question",
        raw_shortlist=[RetrievalCandidate(chunk=chunk), RetrievalCandidate(chunk=sibling)],
        selected_candidates=[RetrievalCandidate(chunk=chunk), RetrievalCandidate(chunk=sibling)],
        candidates=[RetrievalCandidate(chunk=chunk), RetrievalCandidate(chunk=sibling)],
        packed_chunks=[chunk, sibling],
    )
    retrieval_calls: list[tuple[int, int, int]] = []
    assessments = [
        RetrievalAssessment(
            sufficient_for_answer=False,
            coverage_risk="high",
            exactness_risk="low",
            support_conflict=False,
            recommended_next_step="answer",
        ),
        RetrievalAssessment(
            sufficient_for_answer=True,
            coverage_risk="low",
            exactness_risk="low",
            support_conflict=False,
            recommended_next_step="answer",
        ),
    ]

    monkeypatch.setattr(pipeline, "load_index_manifest", lambda settings, namespace=None: {"namespace": "test"})
    monkeypatch.setattr(pipeline, "read_embedding_store", lambda path: {})
    monkeypatch.setattr(pipeline, "read_normalized_documents", lambda path: [])
    monkeypatch.setattr(pipeline, "build_runtime_settings", lambda settings, profile: settings)
    monkeypatch.setattr(pipeline, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(pipeline, "require_search_available", lambda client, settings, backend: None)
    monkeypatch.setattr(
        pipeline,
        "answer_strategy_registry",
        SimpleNamespace(
            get=lambda name: (
                lambda settings, question, evidence, *, persona=None: AnswerResult(
                    question=question,
                    answer_text="final answer",
                    strategy_name=name,
                    model_name="command-a-03-2025",
                    evidence_bundle=evidence,
                )
            )
        ),
    )

    class _StubEmbedder:
        def __init__(self, settings: Settings) -> None:
            del settings

        def embed_texts(self, texts, input_type="search_query"):
            del texts, input_type
            return [[0.1, 0.2]]

    class _StubRetriever:
        def __init__(self, *args, **kwargs) -> None:
            self.call_index = 0

        def retrieve(self, **kwargs):
            retrieval_calls.append(
                (
                    kwargs["candidate_k"],
                    kwargs["rerank_top_n"],
                    kwargs["per_query_candidate_k"],
                )
            )
            bundle = first_bundle if self.call_index == 0 else second_bundle
            self.call_index += 1
            return bundle.model_copy(deep=True)

    def _fake_assess_retrieval(*args, **kwargs):
        del args, kwargs
        return assessments.pop(0)

    monkeypatch.setattr(pipeline, "CohereEmbedder", _StubEmbedder)
    monkeypatch.setattr(pipeline, "HybridRetriever", _StubRetriever)
    monkeypatch.setattr(
        pipeline,
        "CohereQueryExpander",
        lambda settings: SimpleNamespace(expand=lambda question, max_queries: []),
    )
    monkeypatch.setattr(pipeline, "assess_retrieval", _fake_assess_retrieval)

    answer_callback = build_answer_callback(
        settings,
        "baseline_vector_rerank_shortlist_hybrid_retry",
        chunks=[chunk, sibling],
    )
    result = answer_callback(SimpleNamespace(question="What happens if the workflow branch changes?"))

    assert retrieval_calls == [(48, 48, 24), (64, 64, 32)]
    assert result.serve_trace is not None
    assert result.serve_trace.serve_mode == "indexed_retry"
    assert result.serve_trace.escalation_decision == "answer"
    assert result.serve_trace.retry_retrieval_assessment is not None
    assert result.serve_trace.retry_policy is not None
    assert result.serve_trace.retry_policy["recommended_next_step"] == "answer"
    assert result.serve_trace.retrieval_stats["initial_raw_shortlist_count"] == 8
    assert result.serve_trace.retrieval_stats["initial_packed_count"] == 2
    assert result.evidence_bundle is not None
    assert [packed.chunk_id for packed in result.evidence_bundle.packed_chunks] == [chunk.chunk_id, sibling.chunk_id]
