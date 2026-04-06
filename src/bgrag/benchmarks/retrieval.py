"""Retrieval-only benchmark runner for deterministic evidence inspection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from bgrag.config import Settings
from bgrag.eval.loader import load_eval_cases
from bgrag.eval.retrieval_metrics import compute_retrieval_metrics
from bgrag.indexing.corpus_store import read_chunks, read_normalized_documents
from bgrag.indexing.embedder import CohereEmbedder, read_embedding_store
from bgrag.indexing.search_backend import build_search_client, require_search_available
from bgrag.manifests import build_run_name, index_embeddings_path, load_index_manifest
from bgrag.profiles.loader import load_profile
from bgrag.profiles.models import RuntimeProfile
from bgrag.profiles.runtime import build_runtime_settings
from bgrag.retrieval.query_expansion import CohereQueryExpander
from bgrag.retrieval.retriever import HybridRetriever, requires_chunk_embedding_store
from bgrag.types import ChunkRecord, EvalCase, RetrievalCandidate


@dataclass
class RankedChunkRef:
    rank: int
    chunk_id: str
    canonical_url: str
    heading: str | None
    text_preview: str


@dataclass
class RetrievalBenchmarkCaseResult:
    case_id: str
    question: str
    retrieval_queries: list[str]
    query_planning_seconds: float
    query_embedding_seconds: float
    retrieval_seconds: float
    lexical_search_seconds: float
    vector_search_seconds: float
    candidate_fusion_seconds: float
    rerank_seconds: float
    packing_seconds: float
    total_case_seconds: float
    raw_shortlist_primary_url_hit: bool
    selected_primary_url_hit: bool
    packed_primary_url_hit: bool
    raw_shortlist_expected_url_recall: float
    selected_expected_url_recall: float
    packed_expected_url_recall: float
    raw_shortlist_claim_evidence_recall: float
    selected_claim_evidence_recall: float
    packed_claim_evidence_recall: float
    raw_shortlist_chunk_support_recall: float
    selected_chunk_support_recall: float
    packed_chunk_support_recall: float
    claim_evidence_annotated: bool
    claim_chunk_support_annotated: bool
    first_expected_raw_shortlist_rank: int | None
    first_expected_selected_rank: int | None
    first_expected_packed_rank: int | None
    top_raw_shortlist: list[RankedChunkRef]
    top_selected: list[RankedChunkRef]
    top_packed: list[RankedChunkRef]


@dataclass
class RetrievalBenchmarkRun:
    run_name: str
    created_at: str
    profile_name: str
    eval_path: str
    query_mode: str
    index_namespace: str
    case_results: list[RetrievalBenchmarkCaseResult]
    overall_metrics: dict[str, float | int | str | None]


def _normalized_urls(urls: list[str]) -> set[str]:
    return {url.strip().rstrip("/") for url in urls if url.strip()}


def _chunk_matches_case(chunk: ChunkRecord, case: EvalCase) -> bool:
    normalized_url = chunk.canonical_url.strip().rstrip("/")
    expected_urls = _normalized_urls(case.primary_urls + case.supporting_urls)
    return normalized_url in expected_urls


def _first_expected_rank(case: EvalCase, chunks: list[ChunkRecord]) -> int | None:
    for index, chunk in enumerate(chunks, start=1):
        if _chunk_matches_case(chunk, case):
            return index
    return None


def _summarize_chunks(chunks: list[ChunkRecord], *, limit: int) -> list[RankedChunkRef]:
    refs: list[RankedChunkRef] = []
    for index, chunk in enumerate(chunks[:limit], start=1):
        refs.append(
            RankedChunkRef(
                rank=index,
                chunk_id=chunk.chunk_id,
                canonical_url=chunk.canonical_url,
                heading=chunk.heading,
                text_preview=chunk.text[:220].replace("\n", " "),
            )
        )
    return refs


def _mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _rank_mean(values: list[int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _mrr(values: list[int | None]) -> float:
    present = [value for value in values if value is not None]
    return mean(1.0 / value for value in present) if present else 0.0


def _miss_count(values: list[int | None]) -> int:
    return sum(1 for value in values if value is None)


def _compute_overall_metrics(case_results: list[RetrievalBenchmarkCaseResult]) -> dict[str, float | int | str | None]:
    raw_ranks = [result.first_expected_raw_shortlist_rank for result in case_results]
    selected_ranks = [result.first_expected_selected_rank for result in case_results]
    packed_ranks = [result.first_expected_packed_rank for result in case_results]
    claim_annotated = [result for result in case_results if result.claim_evidence_annotated]
    chunk_annotated = [result for result in case_results if result.claim_chunk_support_annotated]
    return {
        "case_count": len(case_results),
        "mean_query_planning_seconds": _mean_or_zero([result.query_planning_seconds for result in case_results]),
        "mean_query_embedding_seconds": _mean_or_zero([result.query_embedding_seconds for result in case_results]),
        "mean_retrieval_seconds": _mean_or_zero([result.retrieval_seconds for result in case_results]),
        "mean_lexical_search_seconds": _mean_or_zero([result.lexical_search_seconds for result in case_results]),
        "mean_vector_search_seconds": _mean_or_zero([result.vector_search_seconds for result in case_results]),
        "mean_candidate_fusion_seconds": _mean_or_zero([result.candidate_fusion_seconds for result in case_results]),
        "mean_rerank_seconds": _mean_or_zero([result.rerank_seconds for result in case_results]),
        "mean_packing_seconds": _mean_or_zero([result.packing_seconds for result in case_results]),
        "mean_total_case_seconds": _mean_or_zero([result.total_case_seconds for result in case_results]),
        "raw_shortlist_primary_url_hit_rate": _mean_or_zero(
            [1.0 if result.raw_shortlist_primary_url_hit else 0.0 for result in case_results]
        ),
        "selected_primary_url_hit_rate": _mean_or_zero(
            [1.0 if result.selected_primary_url_hit else 0.0 for result in case_results]
        ),
        "packed_primary_url_hit_rate": _mean_or_zero(
            [1.0 if result.packed_primary_url_hit else 0.0 for result in case_results]
        ),
        "raw_shortlist_expected_url_recall_mean": _mean_or_zero(
            [result.raw_shortlist_expected_url_recall for result in case_results]
        ),
        "selected_expected_url_recall_mean": _mean_or_zero(
            [result.selected_expected_url_recall for result in case_results]
        ),
        "packed_expected_url_recall_mean": _mean_or_zero(
            [result.packed_expected_url_recall for result in case_results]
        ),
        "raw_shortlist_claim_evidence_recall_mean_annotated": _mean_or_zero(
            [result.raw_shortlist_claim_evidence_recall for result in claim_annotated]
        ),
        "selected_claim_evidence_recall_mean_annotated": _mean_or_zero(
            [result.selected_claim_evidence_recall for result in claim_annotated]
        ),
        "packed_claim_evidence_recall_mean_annotated": _mean_or_zero(
            [result.packed_claim_evidence_recall for result in claim_annotated]
        ),
        "raw_shortlist_chunk_support_recall_mean_annotated": _mean_or_zero(
            [result.raw_shortlist_chunk_support_recall for result in chunk_annotated]
        ),
        "selected_chunk_support_recall_mean_annotated": _mean_or_zero(
            [result.selected_chunk_support_recall for result in chunk_annotated]
        ),
        "packed_chunk_support_recall_mean_annotated": _mean_or_zero(
            [result.packed_chunk_support_recall for result in chunk_annotated]
        ),
        "raw_shortlist_first_expected_rank_mean_hit_only": _rank_mean(raw_ranks),
        "selected_first_expected_rank_mean_hit_only": _rank_mean(selected_ranks),
        "packed_first_expected_rank_mean_hit_only": _rank_mean(packed_ranks),
        "raw_shortlist_mrr": _mrr(raw_ranks),
        "selected_mrr": _mrr(selected_ranks),
        "packed_mrr": _mrr(packed_ranks),
        "raw_shortlist_miss_count": _miss_count(raw_ranks),
        "selected_miss_count": _miss_count(selected_ranks),
        "packed_miss_count": _miss_count(packed_ranks),
        "claim_evidence_annotated_case_count": len(claim_annotated),
        "claim_chunk_support_annotated_case_count": len(chunk_annotated),
    }


def _candidate_chunks(candidates: list[RetrievalCandidate]) -> list[ChunkRecord]:
    return [candidate.chunk for candidate in candidates]


def run_retrieval_benchmark(
    settings: Settings,
    *,
    eval_path: Path,
    profile_name: str,
    query_mode: str = "single",
    top_chunk_limit: int = 8,
    index_namespace: str | None = None,
    runtime_profile: RuntimeProfile | None = None,
) -> RetrievalBenchmarkRun:
    if query_mode not in {"single", "profile"}:
        raise ValueError(f"Unsupported query_mode: {query_mode}")

    profile = runtime_profile or load_profile(profile_name, settings)
    runtime_settings = build_runtime_settings(settings, profile)
    index_manifest = load_index_manifest(settings, index_namespace)
    resolved_index_namespace = str(index_manifest["namespace"])
    chunks = read_chunks(settings.resolve(Path("datasets/corpus/chunks.jsonl")))
    documents = read_normalized_documents(settings.resolve(Path("datasets/corpus/documents")))
    embedding_store = read_embedding_store(index_embeddings_path(settings, resolved_index_namespace))
    if requires_chunk_embedding_store(
        dense_retrieval_backend=profile.retrieval.dense_retrieval_backend,
        enable_mmr_diversity=profile.retrieval.enable_mmr_diversity,
    ) and not embedding_store:
        raise RuntimeError("Retrieval benchmark requires a populated embedding store for this retrieval path.")

    retrieval_mode = getattr(profile.retrieval, "retrieval_mode", "hybrid_es_rerank")
    search_backend = getattr(profile.retrieval, "search_backend", "elasticsearch")
    elastic = None
    if retrieval_mode != "rerank_all_corpus":
        elastic = build_search_client(settings, search_backend)
        require_search_available(elastic, settings, search_backend)

    settings.require_cohere_key("Retrieval benchmark")
    embedder = CohereEmbedder(settings)
    retriever = HybridRetriever(
        runtime_settings,
        elastic=elastic,
        index_namespace=resolved_index_namespace,
        documents=documents,
        search_backend=search_backend,
    )
    query_expander = CohereQueryExpander(runtime_settings) if profile.retrieval.enable_query_decomposition else None
    hybrid_fusion_mode = getattr(profile.retrieval, "hybrid_fusion_mode", "weighted_alpha")
    hybrid_rrf_k = getattr(profile.retrieval, "hybrid_rrf_k", 60)

    case_results: list[RetrievalBenchmarkCaseResult] = []
    for case in load_eval_cases(eval_path):
        case_start = perf_counter()
        retrieval_queries = [case.question]
        after_plan = case_start
        if query_mode == "profile" and query_expander is not None:
            retrieval_queries.extend(query_expander.expand(case.question, profile.retrieval.max_expanded_queries))
            after_plan = perf_counter()

        query_embeddings = embedder.embed_texts(retrieval_queries, input_type="search_query")
        after_embed = perf_counter()
        evidence = retriever.retrieve(
            question=case.question,
            chunks=chunks,
            query_embedding=query_embeddings[0] if query_embeddings else None,
            chunk_embeddings=embedding_store,
            source_topology=profile.retrieval.source_topology,
            top_k=min(profile.retrieval.top_k, profile.answering.max_packed_docs),
            candidate_k=profile.retrieval.candidate_k,
            retrieval_mode=retrieval_mode,
            hybrid_fusion_mode=hybrid_fusion_mode,
            hybrid_rrf_k=hybrid_rrf_k,
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
            enable_document_seed_retrieval=profile.retrieval.enable_document_seed_retrieval,
            document_seed_ranking_mode=profile.retrieval.document_seed_ranking_mode,
            document_seed_scope=profile.retrieval.document_seed_scope,
            document_seed_scope_docs=profile.retrieval.document_seed_scope_docs,
            document_seed_docs=profile.retrieval.document_seed_docs,
            document_seed_intro_max_order=profile.retrieval.document_seed_intro_max_order,
            document_seed_intro_chunks=profile.retrieval.document_seed_intro_chunks,
            document_seed_candidate_k=profile.retrieval.document_seed_candidate_k,
            document_seed_max_chars=profile.retrieval.document_seed_max_chars,
            evidence_unit=profile.answering.evidence_unit,
            span_max_chars=profile.answering.span_max_chars,
            span_candidate_chunks=profile.answering.span_candidate_chunks,
            span_max_per_chunk=profile.answering.span_max_per_chunk,
            span_rerank_top_n=profile.answering.span_rerank_top_n,
        )
        after_retrieve = perf_counter()

        raw_shortlist = evidence.raw_shortlist or evidence.selected_candidates or evidence.candidates
        selected_candidates = evidence.selected_candidates or evidence.candidates
        raw_shortlist_chunks = _candidate_chunks(raw_shortlist)
        selected_chunks = _candidate_chunks(selected_candidates)
        packed_chunks = evidence.packed_chunks

        raw_shortlist_metrics = compute_retrieval_metrics(case, raw_shortlist_chunks)
        selected_metrics = compute_retrieval_metrics(case, selected_chunks)
        packed_metrics = compute_retrieval_metrics(case, packed_chunks)

        case_results.append(
            RetrievalBenchmarkCaseResult(
                case_id=case.id,
                question=case.question,
                retrieval_queries=list(retrieval_queries),
                query_planning_seconds=after_plan - case_start,
                query_embedding_seconds=after_embed - after_plan,
                retrieval_seconds=after_retrieve - after_embed,
                lexical_search_seconds=evidence.timings.get("lexical_search_seconds", 0.0),
                vector_search_seconds=evidence.timings.get("vector_search_seconds", 0.0),
                candidate_fusion_seconds=evidence.timings.get("candidate_fusion_seconds", 0.0),
                rerank_seconds=evidence.timings.get("rerank_seconds", 0.0),
                packing_seconds=evidence.timings.get("packing_seconds", 0.0),
                total_case_seconds=after_retrieve - case_start,
                raw_shortlist_primary_url_hit=raw_shortlist_metrics.primary_url_hit,
                selected_primary_url_hit=selected_metrics.primary_url_hit,
                packed_primary_url_hit=packed_metrics.primary_url_hit,
                raw_shortlist_expected_url_recall=raw_shortlist_metrics.expected_url_recall,
                selected_expected_url_recall=selected_metrics.expected_url_recall,
                packed_expected_url_recall=packed_metrics.expected_url_recall,
                raw_shortlist_claim_evidence_recall=raw_shortlist_metrics.claim_evidence_recall,
                selected_claim_evidence_recall=selected_metrics.claim_evidence_recall,
                packed_claim_evidence_recall=packed_metrics.claim_evidence_recall,
                raw_shortlist_chunk_support_recall=raw_shortlist_metrics.claim_chunk_support_recall,
                selected_chunk_support_recall=selected_metrics.claim_chunk_support_recall,
                packed_chunk_support_recall=packed_metrics.claim_chunk_support_recall,
                claim_evidence_annotated=packed_metrics.claim_evidence_annotated,
                claim_chunk_support_annotated=packed_metrics.claim_chunk_support_annotated,
                first_expected_raw_shortlist_rank=_first_expected_rank(case, raw_shortlist_chunks),
                first_expected_selected_rank=_first_expected_rank(case, selected_chunks),
                first_expected_packed_rank=_first_expected_rank(case, packed_chunks),
                top_raw_shortlist=_summarize_chunks(raw_shortlist_chunks, limit=top_chunk_limit),
                top_selected=_summarize_chunks(selected_chunks, limit=top_chunk_limit),
                top_packed=_summarize_chunks(packed_chunks, limit=top_chunk_limit),
            )
        )

    run_name = build_run_name(f"{profile.name}_retrieval_{query_mode}")
    return RetrievalBenchmarkRun(
        run_name=run_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        profile_name=profile.name,
        eval_path=str(eval_path),
        query_mode=query_mode,
        index_namespace=resolved_index_namespace,
        case_results=case_results,
        overall_metrics=_compute_overall_metrics(case_results),
    )


def write_retrieval_benchmark_artifacts(
    settings: Settings,
    run: RetrievalBenchmarkRun,
) -> tuple[Path, Path]:
    output_dir = settings.resolve(Path("datasets/runs/retrieval_benchmark"))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    md_path = output_dir / f"{run.run_name}.md"

    json_path.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")

    lines = [
        f"# Retrieval Benchmark: {run.profile_name} ({run.query_mode})",
        "",
        f"- run_name: {run.run_name}",
        f"- created_at: {run.created_at}",
        f"- eval_path: {run.eval_path}",
        f"- index_namespace: {run.index_namespace}",
        "",
        "## Overall Metrics",
        "",
    ]
    for key, value in run.overall_metrics.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Cases", ""])
    for case in run.case_results:
        lines.extend(
            [
                f"### {case.case_id}",
                f"- question: {case.question}",
                f"- retrieval_queries: {case.retrieval_queries}",
                f"- first_expected_raw_shortlist_rank: {case.first_expected_raw_shortlist_rank}",
                f"- first_expected_selected_rank: {case.first_expected_selected_rank}",
                f"- first_expected_packed_rank: {case.first_expected_packed_rank}",
                f"- raw_shortlist_expected_url_recall: {case.raw_shortlist_expected_url_recall}",
                f"- selected_expected_url_recall: {case.selected_expected_url_recall}",
                f"- packed_expected_url_recall: {case.packed_expected_url_recall}",
                f"- raw_shortlist_claim_evidence_recall: {case.raw_shortlist_claim_evidence_recall}",
                f"- selected_claim_evidence_recall: {case.selected_claim_evidence_recall}",
                f"- packed_claim_evidence_recall: {case.packed_claim_evidence_recall}",
                f"- total_case_seconds: {case.total_case_seconds:.3f}",
                "- top_packed:",
            ]
        )
        for chunk in case.top_packed:
            lines.append(f"  - {chunk.rank}. {chunk.chunk_id} | {chunk.canonical_url} | {chunk.text_preview}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
