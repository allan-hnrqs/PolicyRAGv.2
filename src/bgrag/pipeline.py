"""End-to-end orchestration helpers."""

from __future__ import annotations

from collections import defaultdict
import json
from time import perf_counter
from pathlib import Path

from bgrag.chunking.chunkers import section_chunker
from bgrag.collect.collector import DEFAULT_SEED_URLS, crawl_scope, raw_snapshot_stem, write_raw_snapshot
from bgrag.config import Settings
from bgrag.corpus_audit import build_corpus_audit, write_corpus_audit
from bgrag.indexing.corpus_store import read_chunks, read_normalized_documents, write_chunks, write_normalized_documents
from bgrag.indexing.embedder import CohereEmbedder, read_embedding_store, write_embedding_store
from bgrag.indexing.search_backend import build_search_client, index_chunks_for_backend, require_search_available
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
from bgrag.profiles.models import RuntimeProfile
from bgrag.retrieval.mode_selection import CohereRetrievalModeSelector
from bgrag.retrieval.query_expansion import CohereQueryExpander
from bgrag.retrieval.retriever import HybridRetriever, requires_chunk_embedding_store
from bgrag.serving.agentic import build_agentic_escalation_runner, run_agentic_official_browse
from bgrag.serving.assessment import assess_retrieval
from bgrag.serving.retry_policy import decide_hybrid_retry
from bgrag.types import AnswerResult, ChunkRecord, ConversationState, NormalizedDocument, ServeTrace, SourceDocument


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
    search_backend = getattr(profile.retrieval, "search_backend", "elasticsearch")
    search_client = build_search_client(settings, search_backend)
    require_search_available(search_client, settings, search_backend)
    embedder = CohereEmbedder(settings)
    vectors = embedder.embed_texts([chunk.text for chunk in chunks], input_type="search_document")
    vector_map = {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors)}
    index_chunks_for_backend(
        search_client,
        chunks,
        namespace=namespace,
        embeddings=vector_map,
        backend=search_backend,
    )
    write_embedding_store(index_embeddings_path(settings, namespace), vector_map)
    manifest = build_index_manifest(
        settings,
        profile_name,
        namespace=namespace,
        chunk_count=len(chunks),
        search_backend=search_backend,
    )
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
    runtime_profile: RuntimeProfile | None = None,
):
    settings.require_cohere_key("Answer generation")
    profile = runtime_profile or load_profile(profile_name, settings)
    answering_profile = profile.answering
    runtime_settings = build_runtime_settings(settings, profile)
    index_manifest = load_index_manifest(settings, index_namespace)
    namespace = str(index_manifest["namespace"])
    chunks = chunks or read_chunks(settings.resolve(Path("datasets/corpus/chunks.jsonl")))
    documents = read_normalized_documents(settings.resolve(Path("datasets/corpus/documents")))
    retrieval_mode = getattr(profile.retrieval, "retrieval_mode", "hybrid_es_rerank")
    search_backend = getattr(profile.retrieval, "search_backend", "elasticsearch")
    embedding_store = read_embedding_store(index_embeddings_path(settings, namespace))
    if requires_chunk_embedding_store(
        dense_retrieval_backend=profile.retrieval.dense_retrieval_backend,
        enable_mmr_diversity=profile.retrieval.enable_mmr_diversity,
    ):
        if not embedding_store:
            raise RuntimeError(
                "This retrieval path requires a populated embedding store. "
                "Run `bgrag build-index` before querying or evaluation."
            )
        missing_embeddings = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in embedding_store]
        if missing_embeddings:
            raise RuntimeError(
                "This retrieval path requires embeddings for every loaded chunk. "
                "Rebuild the index without partial limits before querying or evaluation."
            )
    elastic = None
    if retrieval_mode != "rerank_all_corpus":
        elastic = build_search_client(settings, search_backend)
        require_search_available(elastic, settings, search_backend)
    embedder = CohereEmbedder(settings)
    retriever = HybridRetriever(
        runtime_settings,
        elastic=elastic,
        index_namespace=namespace,
        documents=documents,
        search_backend=search_backend,
    )
    if (
        hasattr(answer_strategy_registry, "_items")
        and profile.answering.strategy not in answer_strategy_registry._items
        and getattr(answer_strategy_registry.get, "__module__", "") == "bgrag.registry"
    ):
        import bgrag.answering.strategies  # noqa: F401  Ensure answer strategies are registered when needed.
    answer_strategy = answer_strategy_registry.get(profile.answering.strategy)
    query_expander = CohereQueryExpander(runtime_settings) if profile.retrieval.enable_query_decomposition else None
    retrieval_mode_selector = (
        CohereRetrievalModeSelector(runtime_settings) if profile.retrieval.enable_retrieval_mode_selection else None
    )
    hybrid_fusion_mode = getattr(profile.retrieval, "hybrid_fusion_mode", "weighted_alpha")
    hybrid_rrf_k = getattr(profile.retrieval, "hybrid_rrf_k", 60)
    enable_retrieval_assessment = getattr(profile.retrieval, "enable_retrieval_assessment", False)
    enable_hybrid_retry_trigger = getattr(profile.retrieval, "enable_hybrid_retry_trigger", False)
    assessment_max_chunks = getattr(profile.retrieval, "assessment_max_chunks", 8)
    answer_top_k = min(
        profile.retrieval.top_k,
        getattr(answering_profile, "max_packed_docs", settings.max_packed_docs),
    )
    retry_candidate_k = getattr(profile.retrieval, "retry_candidate_k", profile.retrieval.candidate_k)
    retry_rerank_top_n = getattr(profile.retrieval, "retry_rerank_top_n", profile.retrieval.rerank_top_n)
    retry_per_query_candidate_k = getattr(
        profile.retrieval,
        "retry_per_query_candidate_k",
        profile.retrieval.per_query_candidate_k,
    )
    enable_official_site_escalation = getattr(profile.retrieval, "enable_official_site_escalation", False)
    escalation_max_steps = getattr(profile.retrieval, "escalation_max_steps", 5)
    escalation_max_live_pages = getattr(profile.retrieval, "escalation_max_live_pages", 6)
    escalation_max_live_chunks = getattr(profile.retrieval, "escalation_max_live_chunks", 24)
    escalation_max_rerank_chunks = getattr(profile.retrieval, "escalation_max_rerank_chunks", 12)
    escalation_max_indexed_chunks = getattr(profile.retrieval, "escalation_max_indexed_chunks", answer_top_k)
    escalation_max_rerank_calls = getattr(profile.retrieval, "escalation_max_rerank_calls", 2)
    if enable_official_site_escalation and not enable_retrieval_assessment:
        raise RuntimeError("Official-site escalation requires retrieval assessment to be enabled.")
    assessment_model_name = (
        getattr(answering_profile, "assessment_model_name", None)
        or getattr(answering_profile, "planner_model_name", None)
        or runtime_settings.cohere_query_planner_model
    )
    def _accumulate_retrieval_timings(target: dict[str, float], evidence) -> None:
        for key in (
            "lexical_search_seconds",
            "vector_search_seconds",
            "candidate_fusion_seconds",
            "rerank_seconds",
            "packing_seconds",
        ):
            target[key] += float(evidence.timings.get(key, 0.0))

    def answer_case(case) -> AnswerResult:
        start = perf_counter()
        conversation_state = getattr(case, "conversation_state", None)
        retrieval_queries = [case.question]
        after_plan = start
        if query_expander is not None:
            expanded = query_expander.expand(case.question, profile.retrieval.max_expanded_queries)
            retrieval_queries.extend(expanded)
            after_plan = perf_counter()
        query_embeddings = embedder.embed_texts(retrieval_queries, input_type="search_query")
        after_embed = perf_counter()
        
        def retrieve_evidence(
            *,
            candidate_k: int,
            rerank_top_n: int,
            per_query_candidate_k: int,
        ):
            def _retrieve_core(*, enable_document_seed_retrieval: bool):
                return retriever.retrieve(
                    question=case.question,
                    chunks=chunks,
                    query_embedding=query_embeddings[0] if query_embeddings else None,
                    chunk_embeddings=embedding_store,
                    source_topology=profile.retrieval.source_topology,
                    top_k=answer_top_k,
                    candidate_k=candidate_k,
                    retrieval_mode=getattr(profile.retrieval, "retrieval_mode", "hybrid_es_rerank"),
                    hybrid_fusion_mode=hybrid_fusion_mode,
                    hybrid_rrf_k=hybrid_rrf_k,
                    retrieval_alpha=profile.retrieval.retrieval_alpha,
                    rerank_top_n=rerank_top_n,
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
                    per_query_candidate_k=per_query_candidate_k,
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

            local_selection_seconds = 0.0
            if retrieval_mode_selector is None:
                return (
                    _retrieve_core(
                        enable_document_seed_retrieval=profile.retrieval.enable_document_seed_retrieval
                    ),
                    local_selection_seconds,
                )

            base_evidence = _retrieve_core(enable_document_seed_retrieval=False)
            selection_start = perf_counter()
            decision = retrieval_mode_selector.select(
                case.question,
                base_evidence,
                max_chunks=profile.retrieval.retrieval_mode_selector_max_chunks,
            )
            selection_end = perf_counter()
            local_selection_seconds = selection_end - selection_start
            if decision.mode == "page_family_expansion":
                evidence = _retrieve_core(enable_document_seed_retrieval=True)
                evidence.notes.append("retrieval_mode_selected:page_family_expansion")
            else:
                evidence = base_evidence
                evidence.notes.append("retrieval_mode_selected:baseline")
            if decision.rationale:
                evidence.notes.append(f"retrieval_mode_rationale:{decision.rationale}")
            return evidence, local_selection_seconds

        retrieval_mode_selection_seconds = 0.0
        retrieval_stage_totals: dict[str, float] = defaultdict(float)
        serve_mode = "indexed_fast"
        escalation_decision = "answer"
        assessment_seconds = 0.0
        retry_retrieval_seconds = 0.0
        official_browse_seconds = 0.0
        initial_assessment = None
        retry_assessment = None
        retry_policy_decision = None

        evidence, selection_seconds = retrieve_evidence(
            candidate_k=profile.retrieval.candidate_k,
            rerank_top_n=profile.retrieval.rerank_top_n,
            per_query_candidate_k=profile.retrieval.per_query_candidate_k,
        )
        after_retrieve = perf_counter()
        initial_retrieval_seconds = after_retrieve - after_embed
        retrieval_mode_selection_seconds += selection_seconds
        _accumulate_retrieval_timings(retrieval_stage_totals, evidence)
        initial_raw_shortlist_count = len(evidence.raw_shortlist)
        initial_selected_count = len(evidence.selected_candidates or evidence.candidates)
        initial_packed_count = len(evidence.packed_chunks)

        if enable_retrieval_assessment:
            assessment_start = perf_counter()
            initial_assessment = assess_retrieval(
                runtime_settings,
                model_name=assessment_model_name,
                question=case.question,
                evidence=evidence,
                conversation_state=conversation_state,
                max_chunks=assessment_max_chunks,
            )
            assessment_seconds += perf_counter() - assessment_start
            evidence.retrieval_assessment = initial_assessment
            escalation_decision = initial_assessment.recommended_next_step
            if enable_hybrid_retry_trigger:
                retry_policy_decision = decide_hybrid_retry(
                    question=case.question,
                    evidence=evidence,
                    retrieval_assessment=initial_assessment,
                    conversation_state=conversation_state,
                    enable_official_site_escalation=enable_official_site_escalation,
                )
                escalation_decision = retry_policy_decision.recommended_next_step

            if initial_assessment.recommended_next_step == "retry_retrieve":
                if enable_hybrid_retry_trigger:
                    should_retry = escalation_decision == "retry_retrieve"
                else:
                    should_retry = True
            else:
                should_retry = escalation_decision == "retry_retrieve"

            if should_retry:
                serve_mode = "indexed_retry"
                retry_start = perf_counter()
                evidence, retry_selection_seconds = retrieve_evidence(
                    candidate_k=retry_candidate_k,
                    rerank_top_n=retry_rerank_top_n,
                    per_query_candidate_k=retry_per_query_candidate_k,
                )
                retry_retrieval_seconds = perf_counter() - retry_start
                after_retrieve = perf_counter()
                retrieval_mode_selection_seconds += retry_selection_seconds
                _accumulate_retrieval_timings(retrieval_stage_totals, evidence)
                assessment_start = perf_counter()
                retry_assessment = assess_retrieval(
                    runtime_settings,
                    model_name=assessment_model_name,
                    question=case.question,
                    evidence=evidence,
                    conversation_state=conversation_state,
                    max_chunks=assessment_max_chunks,
                )
                assessment_seconds += perf_counter() - assessment_start
                evidence.retrieval_assessment = retry_assessment
                escalation_decision = retry_assessment.recommended_next_step
                if enable_hybrid_retry_trigger:
                    retry_policy_decision = decide_hybrid_retry(
                        question=case.question,
                        evidence=evidence,
                        retrieval_assessment=retry_assessment,
                        conversation_state=conversation_state,
                        enable_official_site_escalation=enable_official_site_escalation,
                    )
                    escalation_decision = retry_policy_decision.recommended_next_step
                if escalation_decision == "retry_retrieve":
                    evidence.notes.append("retry_budget_exhausted")
                    escalation_decision = "answer"

        if escalation_decision == "browse_official":
            if not enable_official_site_escalation:
                raise RuntimeError(
                    "Retrieval assessment requested official-site browsing, but official-site escalation is disabled."
                )
            serve_mode = "official_browse"
            browse_start = perf_counter()
            escalation_deps = build_agentic_escalation_runner(
                settings=settings,
                runtime_settings=runtime_settings,
                question=case.question,
                answer_strategy=answer_strategy,
                planner_model_name=assessment_model_name,
                answer_top_k=answer_top_k,
                max_steps=escalation_max_steps,
                max_live_pages=escalation_max_live_pages,
                max_live_chunks=escalation_max_live_chunks,
                max_rerank_chunks=escalation_max_rerank_chunks,
                max_indexed_chunks=escalation_max_indexed_chunks,
                max_rerank_calls=escalation_max_rerank_calls,
                retriever=retriever,
                chunks=chunks,
                chunk_embeddings=embedding_store,
                documents=documents,
                dense_retrieval_backend=profile.retrieval.dense_retrieval_backend,
                retrieval_mode=profile.retrieval.retrieval_mode,
                hybrid_fusion_mode=hybrid_fusion_mode,
                hybrid_rrf_k=hybrid_rrf_k,
                retrieval_alpha=profile.retrieval.retrieval_alpha,
                candidate_k=profile.retrieval.candidate_k,
                rerank_top_n=profile.retrieval.rerank_top_n,
                es_knn_num_candidates=profile.retrieval.es_knn_num_candidates,
                conversation_state=conversation_state,
            )
            escalation_deps.current_assessment = retry_assessment or initial_assessment
            result, tool_trace = run_agentic_official_browse(escalation_deps)
            official_browse_seconds = perf_counter() - browse_start
            if result.evidence_bundle is not None:
                result.evidence_bundle.tool_trace = tool_trace
                result.evidence_bundle.retrieval_assessment = escalation_deps.current_assessment
            after_answer = perf_counter()
            browse_answer_generation_seconds = next(
                (
                    step.elapsed_seconds
                    for step in reversed(tool_trace)
                    if step.tool == "answer_from_evidence" and step.elapsed_seconds is not None
                ),
                0.0,
            )
        else:
            result = answer_strategy(runtime_settings, case.question, evidence)
            after_answer = perf_counter()
            browse_answer_generation_seconds = 0.0

        final_evidence = result.evidence_bundle
        serve_trace = ServeTrace(
            serve_mode=serve_mode,
            escalation_decision=escalation_decision,
            retrieval_assessment=initial_assessment,
            retry_retrieval_assessment=retry_assessment,
            retry_policy=(
                retry_policy_decision.model_dump(mode="json")
                if retry_policy_decision is not None
                else None
            ),
            conversation_state=conversation_state if isinstance(conversation_state, ConversationState) else None,
            timings={
                "assessment_seconds": assessment_seconds,
                "retry_retrieval_seconds": retry_retrieval_seconds,
                "official_browse_seconds": official_browse_seconds,
            },
            retrieval_stats={
                "retrieval_query_count": len(retrieval_queries),
                "initial_raw_shortlist_count": initial_raw_shortlist_count,
                "initial_selected_count": initial_selected_count,
                "initial_packed_count": initial_packed_count,
                "final_packed_count": len(final_evidence.packed_chunks) if final_evidence is not None else 0,
            },
            notes=list((final_evidence.notes if final_evidence is not None else [])),
        )

        result.serve_trace = serve_trace
        result.timings.update(
            {
                "query_planning_seconds": after_plan - start if query_expander is not None else 0.0,
                "query_embedding_seconds": after_embed - after_plan,
                "retrieval_seconds": initial_retrieval_seconds + retry_retrieval_seconds,
                "retrieval_mode_selection_seconds": retrieval_mode_selection_seconds,
                "assessment_seconds": assessment_seconds,
                "retry_retrieval_seconds": retry_retrieval_seconds,
                "official_browse_seconds": official_browse_seconds,
                "answer_generation_seconds": result.timings.get(
                    "answer_generation_seconds",
                    browse_answer_generation_seconds or (after_answer - after_retrieve),
                ),
                "total_answer_path_seconds": after_answer - start,
            }
        )
        result.timings.update(retrieval_stage_totals)
        if final_evidence is not None:
            result.timings.update(final_evidence.timings)
        return result

    return answer_case
