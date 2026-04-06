"""Bounded official-site escalation loop built on PydanticAI."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import cohere
import httpx
from pydantic import BaseModel, Field
try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local install state
    Agent = Any  # type: ignore[assignment]
    RunContext = Any  # type: ignore[assignment]
    CohereModel = Any  # type: ignore[assignment]
    CohereProvider = Any  # type: ignore[assignment]
    _PYDANTIC_AI_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _PYDANTIC_AI_IMPORT_ERROR = None

from bgrag.chunking.chunkers import section_chunker
from bgrag.collect.collector import BROWSER_UA, fetch_url
from bgrag.config import Settings
from bgrag.indexing.embedder import CohereEmbedder
from bgrag.normalize.normalizer import normalize_document
from bgrag.benchmarks.official_site import SiteInventoryEntry, build_site_inventory_entries
from bgrag.retrieval.retriever import HybridRetriever
from bgrag.serving.assessment import assess_retrieval
from bgrag.types import (
    AnswerResult,
    ChunkRecord,
    ConversationState,
    EvidenceBundle,
    NormalizedDocument,
    RetrievalAssessment,
    RetrievalCandidate,
    ToolTraceStep,
)


def _require_pydantic_ai() -> None:
    if _PYDANTIC_AI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PydanticAI is required for official-site escalation. "
            "Install the repo dependencies before using the strict agentic path."
        ) from _PYDANTIC_AI_IMPORT_ERROR


class AgenticEscalationOutcome(BaseModel):
    stop_reason: str = Field(min_length=1)


@dataclass
class AgenticEscalationDeps:
    settings: Settings
    runtime_settings: Settings
    question: str
    answer_strategy: Callable[[Settings, str, EvidenceBundle], AnswerResult]
    planner_model_name: str
    answer_top_k: int
    max_steps: int
    max_live_pages: int
    max_live_chunks: int
    max_rerank_chunks: int
    max_indexed_chunks: int
    max_rerank_calls: int
    inventory_entries: list[SiteInventoryEntry]
    inventory_by_url: dict[str, SiteInventoryEntry]
    retriever: HybridRetriever
    chunks: list[ChunkRecord]
    chunk_embeddings: dict[str, list[float]]
    documents: list[NormalizedDocument]
    dense_retrieval_backend: str
    retrieval_mode: str
    hybrid_fusion_mode: str
    hybrid_rrf_k: int
    retrieval_alpha: float
    candidate_k: int
    rerank_top_n: int
    es_knn_num_candidates: int
    conversation_state: ConversationState | None = None
    rerank_client: cohere.ClientV2 | None = None
    live_doc_cache: dict[str, NormalizedDocument] = field(default_factory=dict)
    search_scores: dict[str, float] = field(default_factory=dict)
    pooled_candidates: dict[str, RetrievalCandidate] = field(default_factory=dict)
    tool_trace: list[ToolTraceStep] = field(default_factory=list)
    current_assessment: RetrievalAssessment | None = None
    final_answer: AnswerResult | None = None
    search_count: int = 0
    fetch_count: int = 0
    rerank_count: int = 0
    step_count: int = 0

    def record_step(
        self,
        *,
        tool: str,
        query_or_url: str | None,
        started: float,
        result_count: int | None = None,
        evidence_origin: str | None = None,
        stop_reason: str | None = None,
    ) -> None:
        self.step_count += 1
        self.tool_trace.append(
            ToolTraceStep(
                tool=tool,
                query_or_url=query_or_url,
                elapsed_seconds=perf_counter() - started,
                result_count=result_count,
                evidence_origin=evidence_origin,
                stop_reason=stop_reason,
            )
        )

    def _upsert_candidates(self, candidates: list[RetrievalCandidate]) -> None:
        for candidate in candidates:
            existing = self.pooled_candidates.get(candidate.chunk.chunk_id)
            if existing is None or candidate.blended_score > existing.blended_score:
                self.pooled_candidates[candidate.chunk.chunk_id] = candidate

    def build_evidence_bundle(self, *, note: str) -> EvidenceBundle:
        ranked = sorted(self.pooled_candidates.values(), key=lambda item: item.blended_score, reverse=True)
        selected = ranked[: self.answer_top_k]
        return EvidenceBundle(
            query=self.question,
            raw_shortlist=ranked,
            candidates=selected,
            selected_candidates=selected,
            packed_chunks=[candidate.chunk for candidate in selected],
            retrieval_queries=[self.question],
            notes=[note, "official_site_browse_escalation"],
            retrieval_assessment=self.current_assessment,
            tool_trace=list(self.tool_trace),
        )

    def refresh_assessment(self) -> RetrievalAssessment | None:
        if not self.pooled_candidates:
            self.current_assessment = None
            return None
        evidence = self.build_evidence_bundle(note="assessment_snapshot")
        self.current_assessment = assess_retrieval(
            self.runtime_settings,
            model_name=self.planner_model_name,
            question=self.question,
            evidence=evidence,
            conversation_state=self.conversation_state,
            max_chunks=min(self.answer_top_k, 8),
        )
        return self.current_assessment


def _tool_prompt(deps: AgenticEscalationDeps) -> str:
    assessment = deps.current_assessment.model_dump(mode="json") if deps.current_assessment else None
    return (
        "You are a bounded official-site procurement-policy escalation agent.\n"
        "Use tools only. Prefer the fewest steps that could still answer correctly.\n"
        "Do not invent unsupported specifics.\n"
        "If the latest retrieval assessment says sufficient_for_answer is true, use answer_from_evidence.\n"
        "If evidence still looks incomplete, search or retrieve before answering.\n"
        f"Current state: question={deps.question!r}, steps_used={deps.step_count}, "
        f"fetches_used={deps.fetch_count}, reranks_used={deps.rerank_count}, "
        f"current_assessment={assessment!r}"
    )


def _ensure_budget(deps: AgenticEscalationDeps, *, tool: str) -> None:
    if deps.step_count >= deps.max_steps:
        raise RuntimeError(f"{tool} refused: max_steps exceeded")


def _compact(text: str, *, max_chars: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _render_inventory_hits(entries: list[tuple[SiteInventoryEntry, float]]) -> str:
    if not entries:
        return "No inventory hits."
    return "\n".join(
        f"- {entry.title} | {entry.canonical_url} | score={score:.3f}"
        for entry, score in entries
    )


def build_agentic_escalation_runner(
    *,
    settings: Settings,
    runtime_settings: Settings,
    question: str,
    answer_strategy: Callable[[Settings, str, EvidenceBundle], AnswerResult],
    planner_model_name: str,
    answer_top_k: int,
    max_steps: int,
    max_live_pages: int,
    max_live_chunks: int,
    max_rerank_chunks: int,
    max_indexed_chunks: int,
    max_rerank_calls: int,
    retriever: HybridRetriever,
    chunks: list[ChunkRecord],
    chunk_embeddings: dict[str, list[float]],
    documents: list[NormalizedDocument],
    dense_retrieval_backend: str,
    retrieval_mode: str,
    hybrid_fusion_mode: str,
    hybrid_rrf_k: int,
    retrieval_alpha: float,
    candidate_k: int,
    rerank_top_n: int,
    es_knn_num_candidates: int,
    conversation_state: ConversationState | None = None,
) -> AgenticEscalationDeps:
    inventory_entries = build_site_inventory_entries(documents)
    return AgenticEscalationDeps(
        settings=settings,
        runtime_settings=runtime_settings,
        question=question,
        answer_strategy=answer_strategy,
        planner_model_name=planner_model_name,
        answer_top_k=answer_top_k,
        max_steps=max_steps,
        max_live_pages=max_live_pages,
        max_live_chunks=max_live_chunks,
        max_rerank_chunks=max_rerank_chunks,
        max_indexed_chunks=max_indexed_chunks,
        max_rerank_calls=max_rerank_calls,
        inventory_entries=inventory_entries,
        inventory_by_url={entry.canonical_url: entry for entry in inventory_entries},
        retriever=retriever,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        documents=documents,
        dense_retrieval_backend=dense_retrieval_backend,
        retrieval_mode=retrieval_mode,
        hybrid_fusion_mode=hybrid_fusion_mode,
        hybrid_rrf_k=hybrid_rrf_k,
        retrieval_alpha=retrieval_alpha,
        candidate_k=candidate_k,
        rerank_top_n=rerank_top_n,
        es_knn_num_candidates=es_knn_num_candidates,
        conversation_state=conversation_state,
        rerank_client=cohere.ClientV2(settings.cohere_api_key) if settings.cohere_api_key else None,
    )


def run_agentic_official_browse(
    deps: AgenticEscalationDeps,
) -> tuple[AnswerResult, list[ToolTraceStep]]:
    _require_pydantic_ai()
    deps.settings.require_cohere_key("Official-site escalation")

    agent = Agent(
        CohereModel(
            deps.planner_model_name,
            provider=CohereProvider(api_key=deps.settings.cohere_api_key),
        ),
        deps_type=AgenticEscalationDeps,
        output_type=AgenticEscalationOutcome,
        instructions=lambda ctx: _tool_prompt(ctx.deps),
        defer_model_check=True,
        retries=1,
        output_retries=1,
        end_strategy="early",
    )

    @agent.tool
    def search_official_inventory(ctx: RunContext[AgenticEscalationDeps], query: str, top_k: int = 6) -> str:
        deps = ctx.deps
        _ensure_budget(deps, tool="search_official_inventory")
        started = perf_counter()
        if deps.rerank_client is None:
            raise RuntimeError("Cohere rerank client is unavailable")
        top_n = min(max(1, top_k), deps.max_live_pages)
        response = deps.rerank_client.rerank(
            model=deps.settings.cohere_rerank_model,
            query=query,
            documents=[entry.inventory_text for entry in deps.inventory_entries],
            top_n=top_n,
        )
        hits = [(deps.inventory_entries[int(item.index)], float(item.relevance_score)) for item in response.results]
        deps.search_scores = {entry.canonical_url: score for entry, score in hits}
        deps.search_count += 1
        deps.record_step(
            tool="search_official_inventory",
            query_or_url=query,
            started=started,
            result_count=len(hits),
            evidence_origin="inventory",
        )
        return _render_inventory_hits(hits)

    @agent.tool
    def fetch_official_page(ctx: RunContext[AgenticEscalationDeps], url: str) -> str:
        deps = ctx.deps
        _ensure_budget(deps, tool="fetch_official_page")
        if deps.fetch_count >= deps.max_live_pages:
            raise RuntimeError("fetch_official_page refused: max_live_pages exceeded")
        started = perf_counter()
        cached = deps.live_doc_cache.get(url)
        cache_hit = cached is not None
        if cached is None:
            with httpx.Client(headers={"User-Agent": BROWSER_UA}) as client:
                fetched = fetch_url(client, url)
            cached = normalize_document(fetched.document)
            deps.live_doc_cache[url] = cached
        chunks = section_chunker(cached, enrichers=["authority_metadata", "lineage_metadata", "source_topology_metadata"])
        base_score = deps.search_scores.get(url, 0.0) or 0.01
        deps._upsert_candidates(
            [
                RetrievalCandidate(chunk=chunk, blended_score=base_score, rerank_score=base_score)
                for chunk in chunks
            ]
        )
        deps.fetch_count += 1
        deps.refresh_assessment()
        deps.record_step(
            tool="fetch_official_page",
            query_or_url=url,
            started=started,
            result_count=len(chunks),
            evidence_origin="official_live_page",
            stop_reason="cache_hit" if cache_hit else None,
        )
        return (
            f"Fetched {cached.title} ({cached.canonical_url}) with {len(chunks)} chunk(s). "
            f"Assessment: {deps.current_assessment.model_dump(mode='json') if deps.current_assessment else '<none>'}"
        )

    @agent.tool
    def retrieve_indexed_chunks(ctx: RunContext[AgenticEscalationDeps], query: str, top_k: int = 8) -> str:
        deps = ctx.deps
        _ensure_budget(deps, tool="retrieve_indexed_chunks")
        started = perf_counter()
        top_k = min(max(1, top_k), deps.max_indexed_chunks)
        query_embedding = CohereEmbedder(deps.settings).embed_texts([query], input_type="search_query")[0]
        evidence = deps.retriever.retrieve(
            question=query,
            chunks=deps.chunks,
            query_embedding=query_embedding,
            chunk_embeddings=deps.chunk_embeddings,
            source_topology="ranked_passthrough",
            top_k=top_k,
            candidate_k=max(deps.candidate_k, top_k * 4),
            retrieval_mode=deps.retrieval_mode,
            hybrid_fusion_mode=deps.hybrid_fusion_mode,
            hybrid_rrf_k=deps.hybrid_rrf_k,
            retrieval_alpha=deps.retrieval_alpha,
            rerank_top_n=min(max(deps.rerank_top_n, top_k), deps.max_indexed_chunks),
            dense_retrieval_backend=deps.dense_retrieval_backend,
            es_knn_num_candidates=deps.es_knn_num_candidates,
            retrieval_queries=[query],
            query_embeddings=[query_embedding],
            per_query_candidate_k=max(top_k, min(max(deps.candidate_k, top_k), deps.max_indexed_chunks)),
        )
        deps._upsert_candidates(evidence.selected_candidates or evidence.candidates)
        deps.refresh_assessment()
        deps.record_step(
            tool="retrieve_indexed_chunks",
            query_or_url=query,
            started=started,
            result_count=len(evidence.packed_chunks),
            evidence_origin="indexed_rag",
        )
        return (
            f"Retrieved {len(evidence.packed_chunks)} indexed chunk(s). "
            f"Assessment: {deps.current_assessment.model_dump(mode='json') if deps.current_assessment else '<none>'}"
        )

    @agent.tool
    def rerank_passages(ctx: RunContext[AgenticEscalationDeps], query: str, top_k: int = 8) -> str:
        deps = ctx.deps
        _ensure_budget(deps, tool="rerank_passages")
        if deps.rerank_count >= deps.max_rerank_calls:
            raise RuntimeError("rerank_passages refused: max_rerank_calls exceeded")
        if deps.rerank_client is None:
            raise RuntimeError("Cohere rerank client is unavailable")
        started = perf_counter()
        pool = sorted(deps.pooled_candidates.values(), key=lambda item: item.blended_score, reverse=True)
        if not pool:
            raise RuntimeError("rerank_passages refused: no current passages")
        top_n = min(max(1, top_k), deps.max_rerank_chunks, len(pool))
        response = deps.rerank_client.rerank(
            model=deps.settings.cohere_rerank_model,
            query=query,
            documents=[
                "\n".join(
                    [
                        f"title: {candidate.chunk.title}",
                        f"heading: {candidate.chunk.heading or ''}",
                        f"url: {candidate.chunk.canonical_url}",
                        f"text: {_compact(candidate.chunk.text, max_chars=900)}",
                    ]
                )
                for candidate in pool
            ],
            top_n=top_n,
        )
        reranked: list[RetrievalCandidate] = []
        for item in response.results:
            original = pool[int(item.index)]
            score = float(item.relevance_score)
            reranked.append(
                RetrievalCandidate(
                    chunk=original.chunk,
                    dense_score=original.dense_score,
                    lexical_score=original.lexical_score,
                    rerank_score=score,
                    blended_score=score,
                )
            )
        deps._upsert_candidates(reranked)
        deps.rerank_count += 1
        deps.refresh_assessment()
        deps.record_step(
            tool="rerank_passages",
            query_or_url=query,
            started=started,
            result_count=len(reranked),
            evidence_origin="mixed_pool",
        )
        return (
            f"Reranked {len(reranked)} passage(s). "
            f"Assessment: {deps.current_assessment.model_dump(mode='json') if deps.current_assessment else '<none>'}"
        )

    @agent.tool
    def answer_from_evidence(ctx: RunContext[AgenticEscalationDeps], question: str) -> str:
        deps = ctx.deps
        _ensure_budget(deps, tool="answer_from_evidence")
        started = perf_counter()
        evidence = deps.build_evidence_bundle(note="official_browse_final")
        answer = deps.answer_strategy(deps.runtime_settings, question, evidence)
        deps.final_answer = answer
        deps.record_step(
            tool="answer_from_evidence",
            query_or_url=question,
            started=started,
            result_count=len(evidence.packed_chunks),
            evidence_origin="official_browse_final",
            stop_reason="final_answer_generated",
        )
        return _compact(answer.answer_text, max_chars=500)

    result = agent.run_sync(deps.question, deps=deps)
    if deps.final_answer is not None:
        return deps.final_answer, deps.tool_trace
    raise RuntimeError(
        "Official-site escalation ended without producing a final answer: "
        f"{result.output.stop_reason}"
    )
