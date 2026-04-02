"""End-to-end orchestration helpers."""

from __future__ import annotations

import json
from time import perf_counter
from pathlib import Path

import bgrag.answering.strategies  # Ensure answer strategies are registered.
from bgrag.answering.strategies import inline_evidence_chat
from bgrag.chunking.chunkers import section_chunker
from bgrag.collect.collector import DEFAULT_SEED_URLS, crawl_scope, raw_snapshot_stem, write_raw_snapshot
from bgrag.config import Settings
from bgrag.corpus_audit import build_corpus_audit, write_corpus_audit
from bgrag.indexing.corpus_store import read_chunks, read_normalized_documents, write_chunks, write_normalized_documents
from bgrag.indexing.elastic import build_es_client, index_chunks, require_es_available
from bgrag.indexing.embedder import CohereEmbedder, read_embedding_store, write_embedding_store
from bgrag.manifests import (
    build_index_manifest,
    derive_index_namespace,
    index_embeddings_path,
    load_index_manifest,
    set_active_index_namespace,
    write_index_manifest,
)
from bgrag.normalize.normalizer import assign_graph_relationships, normalize_document
from bgrag.profiles.loader import load_profile
from bgrag.profiles.runtime import build_runtime_settings
from bgrag.registry import answer_strategy_registry, chunker_registry
from bgrag.retrieval.mode_selection import CohereRetrievalModeSelector
from bgrag.retrieval.query_expansion import CohereQueryExpander
from bgrag.retrieval.retriever import HybridRetriever
from bgrag.types import AnswerResult, ChunkRecord, NormalizedDocument, SourceDocument


def run_collect(
    settings: Settings,
    seed_urls: list[str] | None = None,
    max_pages: int = 300,
) -> list[NormalizedDocument]:
    settings.ensure_directories()
    active_seeds = seed_urls or DEFAULT_SEED_URLS
    results = crawl_scope(active_seeds, max_pages=max_pages)
    raw_dir = settings.resolve(settings.raw_dir)
    write_raw_snapshot(raw_dir, results)
    normalized = assign_graph_relationships([normalize_document(result.document) for result in results])
    write_normalized_documents(settings.resolve(Path("datasets/corpus/documents")), normalized)
    manifest_path = settings.resolve(Path("datasets/corpus/collection_manifest.json"))
    family_counts: dict[str, int] = {}
    for document in normalized:
        family_counts[document.source_family.value] = family_counts.get(document.source_family.value, 0) + 1
    manifest_path.write_text(
        json.dumps(
            {
                "seed_urls": active_seeds,
                "max_pages": max_pages,
                "document_count": len(normalized),
                "family_counts": family_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return normalized


def run_build_corpus(settings: Settings, profile_name: str) -> list[ChunkRecord]:
    profile = load_profile(profile_name, settings)
    documents = read_normalized_documents(settings.resolve(Path("datasets/corpus/documents")))
    chunker = chunker_registry.get(profile.chunking.chunker)
    chunks: list[ChunkRecord] = []
    for document in documents:
        chunks.extend(
            chunker(
                document,
                enrichers=profile.chunking.metadata_enrichers,
                window_chars=profile.chunking.sliding_window_chars,
                overlap_chars=profile.chunking.sliding_window_overlap,
            )
        )
    write_chunks(settings.resolve(Path("datasets/corpus/chunks.jsonl")), chunks)
    write_corpus_audit(
        settings.resolve(Path("datasets/corpus/chunk_audit.json")),
        build_corpus_audit(documents, chunks),
    )
    return chunks


def run_refresh_normalized_from_raw(settings: Settings) -> list[NormalizedDocument]:
    raw_dir = settings.resolve(Path("datasets/raw"))
    existing = read_normalized_documents(settings.resolve(Path("datasets/corpus/documents")))
    refreshed: list[NormalizedDocument] = []
    for document in existing:
        raw_path = raw_dir / f"{raw_snapshot_stem(document.canonical_url)}.html"
        if not raw_path.exists():
            raise RuntimeError(f"Missing raw snapshot for {document.canonical_url}: {raw_path}")
        source_document = SourceDocument(
            source_url=document.source_url,
            fetched_at=document.fetched_at,
            final_url=document.canonical_url,
            status_code=200,
            html=raw_path.read_text(encoding="utf-8"),
            discovered_links=list(document.graph.outgoing_in_scope_links),
        )
        refreshed.append(normalize_document(source_document))
    refreshed = assign_graph_relationships(refreshed)
    write_normalized_documents(settings.resolve(Path("datasets/corpus/documents")), refreshed)
    return refreshed


def run_build_index(
    settings: Settings,
    profile_name: str,
    limit_chunks: int = 0,
    index_namespace: str | None = None,
) -> dict[str, int | bool | str]:
    profile = load_profile(profile_name, settings)
    settings.require_cohere_key("Index building with Cohere embeddings")
    chunks = read_chunks(settings.resolve(Path("datasets/corpus/chunks.jsonl")))
    if limit_chunks > 0:
        chunks = chunks[:limit_chunks]
    namespace = index_namespace or derive_index_namespace(settings, profile_name)
    elastic = build_es_client(settings)
    require_es_available(elastic, settings.elastic_url)
    embedder = CohereEmbedder(settings)
    vectors = embedder.embed_texts([chunk.text for chunk in chunks], input_type="search_document")
    vector_map = {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors)}
    index_chunks(elastic, chunks, namespace=namespace, embeddings=vector_map)
    write_embedding_store(index_embeddings_path(settings, namespace), vector_map)
    manifest = build_index_manifest(settings, profile_name, namespace=namespace, chunk_count=len(chunks))
    write_index_manifest(settings, namespace, manifest)
    set_active_index_namespace(settings, namespace)
    return {
        "chunk_count": len(chunks),
        "embedding_count": len(vector_map),
        "topology": profile.retrieval.source_topology,
        "embedding_model": settings.cohere_embed_model,
        "index_namespace": namespace,
    }


def build_answer_callback(
    settings: Settings,
    profile_name: str,
    chunks: list[ChunkRecord] | None = None,
    index_namespace: str | None = None,
):
    settings.require_cohere_key("Answer generation")
    profile = load_profile(profile_name, settings)
    answering_profile = profile.answering
    runtime_settings = build_runtime_settings(settings, profile)
    index_manifest = load_index_manifest(settings, index_namespace)
    namespace = str(index_manifest["namespace"])
    chunks = chunks or read_chunks(settings.resolve(Path("datasets/corpus/chunks.jsonl")))
    documents = read_normalized_documents(settings.resolve(Path("datasets/corpus/documents")))
    embedding_store = read_embedding_store(index_embeddings_path(settings, namespace))
    if not embedding_store:
        raise RuntimeError(
            "Querying requires a populated embedding store. "
            "Run `bgrag build-index` before querying or evaluation."
        )
    missing_embeddings = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in embedding_store]
    if missing_embeddings:
        raise RuntimeError(
            "Querying requires embeddings for every loaded chunk. "
            "Rebuild the index without partial limits before querying or evaluation."
        )
    elastic = build_es_client(settings)
    require_es_available(elastic, settings.elastic_url)
    embedder = CohereEmbedder(settings)
    retriever = HybridRetriever(runtime_settings, elastic=elastic, index_namespace=namespace, documents=documents)
    answer_strategy = answer_strategy_registry.get(profile.answering.strategy)
    query_expander = CohereQueryExpander(runtime_settings) if profile.retrieval.enable_query_decomposition else None
    retrieval_mode_selector = (
        CohereRetrievalModeSelector(runtime_settings) if profile.retrieval.enable_retrieval_mode_selection else None
    )

    def answer_case(case) -> AnswerResult:
        start = perf_counter()
        retrieval_queries = [case.question]
        after_plan = start
        if query_expander is not None:
            expanded = query_expander.expand(case.question, profile.retrieval.max_expanded_queries)
            retrieval_queries.extend(expanded)
            after_plan = perf_counter()
        query_embeddings = embedder.embed_texts(retrieval_queries, input_type="search_query")
        after_embed = perf_counter()
        def retrieve_evidence(*, enable_document_seed_retrieval: bool):
            return retriever.retrieve(
                question=case.question,
                chunks=chunks,
                query_embedding=query_embeddings[0] if query_embeddings else None,
                chunk_embeddings=embedding_store,
                source_topology=profile.retrieval.source_topology,
                top_k=min(profile.retrieval.top_k, getattr(answering_profile, "max_packed_docs", settings.max_packed_docs)),
                candidate_k=profile.retrieval.candidate_k,
                retrieval_alpha=profile.retrieval.retrieval_alpha,
                rerank_top_n=profile.retrieval.rerank_top_n,
                dense_retrieval_backend=profile.retrieval.dense_retrieval_backend,
                es_knn_num_candidates=profile.retrieval.es_knn_num_candidates,
                enable_mmr_diversity=profile.retrieval.enable_mmr_diversity,
                mmr_lambda=profile.retrieval.mmr_lambda,
                enable_ranked_chunk_diversity=profile.retrieval.enable_ranked_chunk_diversity,
                diversity_cover_fraction=profile.retrieval.diversity_cover_fraction,
                max_chunks_per_document=profile.retrieval.max_chunks_per_document,
                max_chunks_per_heading=profile.retrieval.max_chunks_per_heading,
                seed_chunks_per_heading=profile.retrieval.seed_chunks_per_heading,
                retrieval_queries=retrieval_queries,
                query_embeddings=query_embeddings,
                query_fusion_rrf_k=profile.retrieval.query_fusion_rrf_k,
                per_query_candidate_k=profile.retrieval.per_query_candidate_k,
                enable_parallel_query_branches=profile.retrieval.enable_parallel_query_branches,
                enable_page_intro_expansion=profile.retrieval.enable_page_intro_expansion,
                page_intro_candidate_k=profile.retrieval.page_intro_candidate_k,
                page_intro_max_order=profile.retrieval.page_intro_max_order,
                enable_document_context_expansion=profile.retrieval.enable_document_context_expansion,
                document_context_seed_docs=profile.retrieval.document_context_seed_docs,
                document_context_candidate_k=profile.retrieval.document_context_candidate_k,
                document_context_neighbor_docs=profile.retrieval.document_context_neighbor_docs,
                enable_structural_context_augmentation=profile.retrieval.enable_structural_context_augmentation,
                structural_context_seed_docs=profile.retrieval.structural_context_seed_docs,
                structural_context_intro_max_order=profile.retrieval.structural_context_intro_max_order,
                structural_context_same_heading_k=profile.retrieval.structural_context_same_heading_k,
                structural_context_nearby_k=profile.retrieval.structural_context_nearby_k,
                structural_context_nearby_window=profile.retrieval.structural_context_nearby_window,
                structural_context_neighbor_docs=profile.retrieval.structural_context_neighbor_docs,
                enable_document_seed_retrieval=enable_document_seed_retrieval,
                document_seed_ranking_mode=profile.retrieval.document_seed_ranking_mode,
                document_seed_scope=profile.retrieval.document_seed_scope,
                document_seed_scope_docs=profile.retrieval.document_seed_scope_docs,
                document_seed_docs=profile.retrieval.document_seed_docs,
                document_seed_intro_max_order=profile.retrieval.document_seed_intro_max_order,
                document_seed_intro_chunks=profile.retrieval.document_seed_intro_chunks,
                document_seed_candidate_k=profile.retrieval.document_seed_candidate_k,
                document_seed_max_chars=profile.retrieval.document_seed_max_chars,
                evidence_unit=getattr(answering_profile, "evidence_unit", "chunk"),
                span_max_chars=getattr(answering_profile, "span_max_chars", 320),
                span_candidate_chunks=getattr(answering_profile, "span_candidate_chunks", 8),
                span_max_per_chunk=getattr(answering_profile, "span_max_per_chunk", 2),
                span_rerank_top_n=getattr(answering_profile, "span_rerank_top_n", 0),
            )

        retrieval_mode_selection_seconds = 0.0
        if retrieval_mode_selector is None:
            evidence = retrieve_evidence(
                enable_document_seed_retrieval=profile.retrieval.enable_document_seed_retrieval
            )
            after_retrieve = perf_counter()
        else:
            base_evidence = retrieve_evidence(enable_document_seed_retrieval=False)
            selection_start = perf_counter()
            decision = retrieval_mode_selector.select(
                case.question,
                base_evidence,
                max_chunks=profile.retrieval.retrieval_mode_selector_max_chunks,
            )
            selection_end = perf_counter()
            retrieval_mode_selection_seconds = selection_end - selection_start
            if decision.mode == "page_family_expansion":
                evidence = retrieve_evidence(enable_document_seed_retrieval=True)
                evidence.notes.append("retrieval_mode_selected:page_family_expansion")
            else:
                evidence = base_evidence
                evidence.notes.append("retrieval_mode_selected:baseline")
            if decision.rationale:
                evidence.notes.append(f"retrieval_mode_rationale:{decision.rationale}")
            after_retrieve = perf_counter()
        result = answer_strategy(runtime_settings, case.question, evidence)
        after_answer = perf_counter()
        result.timings.update(
            {
                "query_planning_seconds": after_plan - start if query_expander is not None else 0.0,
                "query_embedding_seconds": after_embed - after_plan,
                "retrieval_seconds": after_retrieve - after_embed,
                "retrieval_mode_selection_seconds": retrieval_mode_selection_seconds,
                "answer_generation_seconds": after_answer - after_retrieve,
                "total_answer_path_seconds": after_answer - start,
            }
        )
        if result.evidence_bundle is not None:
            result.timings.update(result.evidence_bundle.timings)
        return result

    return answer_case
