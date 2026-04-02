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
from bgrag.indexing.elastic import build_es_client, require_es_available
from bgrag.indexing.embedder import CohereEmbedder, read_embedding_store
from bgrag.manifests import build_run_name, load_index_manifest
from bgrag.profiles.loader import load_profile
from bgrag.retrieval.query_expansion import CohereQueryExpander
from bgrag.retrieval.retriever import HybridRetriever
from bgrag.types import ChunkRecord, EvalCase


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
    candidate_primary_url_hit: bool
    packed_primary_url_hit: bool
    candidate_expected_url_recall: float
    packed_expected_url_recall: float
    candidate_claim_evidence_recall: float
    packed_claim_evidence_recall: float
    claim_evidence_annotated: bool
    first_expected_candidate_rank: int | None
    first_expected_packed_rank: int | None
    top_candidates: list[RankedChunkRef]
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


def _prefix_match(doc_id: str, prefixes: list[str]) -> bool:
    return any(doc_id.startswith(prefix.strip()) for prefix in prefixes if prefix.strip())


def _chunk_matches_case(chunk: ChunkRecord, case: EvalCase) -> bool:
    normalized_url = chunk.canonical_url.strip().rstrip("/")
    expected_urls = _normalized_urls(case.primary_urls + case.supporting_urls)
    if normalized_url in expected_urls:
        return True
    return _prefix_match(chunk.doc_id, case.expected_doc_prefixes + case.supporting_doc_prefixes)


def _first_expected_rank(case: EvalCase, chunks: list[ChunkRecord]) -> int | None:
    for index, chunk in enumerate(chunks, start=1):
        if _chunk_matches_case(chunk, case):
            return index
    return None


def _summarize_chunks(chunks: list[ChunkRecord], case: EvalCase, *, limit: int) -> list[RankedChunkRef]:
    refs: list[RankedChunkRef] = []
    for index, chunk in enumerate(chunks[:limit], start=1):
        preview = chunk.text[:220].replace("\n", " ")
        refs.append(
            RankedChunkRef(
                rank=index,
                chunk_id=chunk.chunk_id,
                canonical_url=chunk.canonical_url,
                heading=chunk.heading,
                text_preview=preview,
            )
        )
    return refs


def _compute_overall_metrics(case_results: list[RetrievalBenchmarkCaseResult]) -> dict[str, float | int | str | None]:
    candidate_hit_ranks = [result.first_expected_candidate_rank for result in case_results if result.first_expected_candidate_rank]
    packed_hit_ranks = [result.first_expected_packed_rank for result in case_results if result.first_expected_packed_rank]
    annotated_results = [result for result in case_results if result.claim_evidence_annotated]
    return {
        "case_count": len(case_results),
        "mean_query_planning_seconds": mean(result.query_planning_seconds for result in case_results) if case_results else 0.0,
        "mean_query_embedding_seconds": mean(result.query_embedding_seconds for result in case_results) if case_results else 0.0,
        "mean_retrieval_seconds": mean(result.retrieval_seconds for result in case_results) if case_results else 0.0,
        "mean_lexical_search_seconds": (
            mean(result.lexical_search_seconds for result in case_results) if case_results else 0.0
        ),
        "mean_vector_search_seconds": (
            mean(result.vector_search_seconds for result in case_results) if case_results else 0.0
        ),
        "mean_candidate_fusion_seconds": (
            mean(result.candidate_fusion_seconds for result in case_results) if case_results else 0.0
        ),
        "mean_rerank_seconds": mean(result.rerank_seconds for result in case_results) if case_results else 0.0,
        "mean_packing_seconds": mean(result.packing_seconds for result in case_results) if case_results else 0.0,
        "mean_total_case_seconds": mean(result.total_case_seconds for result in case_results) if case_results else 0.0,
        "candidate_primary_url_hit_rate": (
            mean(1.0 if result.candidate_primary_url_hit else 0.0 for result in case_results) if case_results else 0.0
        ),
        "packed_primary_url_hit_rate": (
            mean(1.0 if result.packed_primary_url_hit else 0.0 for result in case_results) if case_results else 0.0
        ),
        "candidate_expected_url_recall_mean": (
            mean(result.candidate_expected_url_recall for result in case_results) if case_results else 0.0
        ),
        "packed_expected_url_recall_mean": (
            mean(result.packed_expected_url_recall for result in case_results) if case_results else 0.0
        ),
        "candidate_claim_evidence_recall_mean_annotated": (
            mean(result.candidate_claim_evidence_recall for result in annotated_results) if annotated_results else 0.0
        ),
        "packed_claim_evidence_recall_mean_annotated": (
            mean(result.packed_claim_evidence_recall for result in annotated_results) if annotated_results else 0.0
        ),
        "candidate_first_expected_rank_mean_hit_only": (
            mean(candidate_hit_ranks) if candidate_hit_ranks else None
        ),
        "packed_first_expected_rank_mean_hit_only": mean(packed_hit_ranks) if packed_hit_ranks else None,
        "candidate_mrr": (
            mean(1.0 / rank for rank in candidate_hit_ranks) if candidate_hit_ranks else 0.0
        ),
        "packed_mrr": mean(1.0 / rank for rank in packed_hit_ranks) if packed_hit_ranks else 0.0,
        "candidate_miss_count": sum(1 for result in case_results if result.first_expected_candidate_rank is None),
        "packed_miss_count": sum(1 for result in case_results if result.first_expected_packed_rank is None),
        "claim_evidence_annotated_case_count": len(annotated_results),
    }


def run_retrieval_benchmark(
    settings: Settings,
    *,
    eval_path: Path,
    profile_name: str,
    query_mode: str = "single",
    top_chunk_limit: int = 8,
) -> RetrievalBenchmarkRun:
    if query_mode not in {"single", "profile"}:
        raise ValueError(f"Unsupported query_mode: {query_mode}")

    profile = load_profile(profile_name, settings)
    index_manifest = load_index_manifest(settings, None)
    index_namespace = str(index_manifest["namespace"])
    chunks = read_chunks(settings.resolve(Path("datasets/corpus/chunks.jsonl")))
    documents = read_normalized_documents(settings.resolve(Path("datasets/corpus/documents")))
    embedding_store = read_embedding_store(settings.resolve(Path(f"datasets/index/{index_namespace}/chunk_embeddings.json")))
    if not embedding_store:
        raise RuntimeError("Retrieval benchmark requires a populated embedding store. Run `bgrag build-index` first.")

    elastic = build_es_client(settings)
    require_es_available(elastic, settings.elastic_url)
    settings.require_cohere_key("Retrieval benchmark")
    embedder = CohereEmbedder(settings)
    retriever = HybridRetriever(settings, elastic=elastic, index_namespace=index_namespace, documents=documents)
    query_expander = CohereQueryExpander(settings) if profile.retrieval.enable_query_decomposition else None

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
            top_k=profile.retrieval.top_k,
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
            enable_document_seed_retrieval=profile.retrieval.enable_document_seed_retrieval,
            document_seed_ranking_mode=profile.retrieval.document_seed_ranking_mode,
            document_seed_scope=profile.retrieval.document_seed_scope,
            document_seed_scope_docs=profile.retrieval.document_seed_scope_docs,
            document_seed_docs=profile.retrieval.document_seed_docs,
            document_seed_intro_max_order=profile.retrieval.document_seed_intro_max_order,
            document_seed_intro_chunks=profile.retrieval.document_seed_intro_chunks,
            document_seed_candidate_k=profile.retrieval.document_seed_candidate_k,
            document_seed_max_chars=profile.retrieval.document_seed_max_chars,
        )
        after_retrieve = perf_counter()
        packed_metrics = compute_retrieval_metrics(case, evidence.packed_chunks)
        candidate_metrics = compute_retrieval_metrics(case, [candidate.chunk for candidate in evidence.candidates])
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
                candidate_primary_url_hit=candidate_metrics.primary_url_hit,
                packed_primary_url_hit=packed_metrics.primary_url_hit,
                candidate_expected_url_recall=candidate_metrics.expected_url_recall,
                packed_expected_url_recall=packed_metrics.expected_url_recall,
                candidate_claim_evidence_recall=candidate_metrics.claim_evidence_recall,
                packed_claim_evidence_recall=packed_metrics.claim_evidence_recall,
                claim_evidence_annotated=packed_metrics.claim_evidence_annotated,
                first_expected_candidate_rank=_first_expected_rank(
                    case,
                    [candidate.chunk for candidate in evidence.candidates],
                ),
                first_expected_packed_rank=_first_expected_rank(case, evidence.packed_chunks),
                top_candidates=_summarize_chunks([candidate.chunk for candidate in evidence.candidates], case, limit=top_chunk_limit),
                top_packed=_summarize_chunks(evidence.packed_chunks, case, limit=top_chunk_limit),
            )
        )

    run_name = build_run_name(f"{profile_name}_retrieval_{query_mode}")
    return RetrievalBenchmarkRun(
        run_name=run_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        profile_name=profile_name,
        eval_path=str(eval_path),
        query_mode=query_mode,
        index_namespace=index_namespace,
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
                f"- first_expected_candidate_rank: {case.first_expected_candidate_rank}",
                f"- first_expected_packed_rank: {case.first_expected_packed_rank}",
                f"- candidate_expected_url_recall: {case.candidate_expected_url_recall}",
                f"- packed_expected_url_recall: {case.packed_expected_url_recall}",
                f"- candidate_claim_evidence_recall: {case.candidate_claim_evidence_recall}",
                f"- packed_claim_evidence_recall: {case.packed_claim_evidence_recall}",
                f"- total_case_seconds: {case.total_case_seconds:.3f}",
                "- top_packed:",
            ]
        )
        for chunk in case.top_packed:
            lines.append(
                f"  - {chunk.rank}. {chunk.chunk_id} | {chunk.canonical_url} | {chunk.text_preview}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
