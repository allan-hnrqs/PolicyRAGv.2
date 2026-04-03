"""Bounded tool-using official-site agent baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cohere
import cohere.types as ct
import httpx

import bgrag.answering.strategies  # Ensure answer strategies are registered.
from bgrag.chunking.chunkers import section_chunker
from bgrag.collect.collector import BROWSER_UA, fetch_url
from bgrag.config import Settings
from bgrag.eval.judge import CohereJudge
from bgrag.eval.loader import load_eval_cases
from bgrag.eval.retrieval_metrics import compute_retrieval_metrics
from bgrag.eval.run_composition import compute_overall_metrics
from bgrag.indexing.corpus_store import read_normalized_documents
from bgrag.manifests import (
    build_run_name,
    code_fingerprint,
    file_sha256,
    profile_sha256,
    repo_relative_path,
    tree_sha256,
    workspace_fingerprint,
)
from bgrag.normalize.normalizer import normalize_document
from bgrag.official_site_baseline import SiteInventoryEntry, build_site_inventory_entries
from bgrag.profiles.loader import load_profile
from bgrag.profiles.runtime import build_runtime_settings
from bgrag.registry import answer_strategy_registry
from bgrag.types import AnswerResult, ChunkRecord, EvalCase, EvalCaseResult, EvalRunResult, EvidenceBundle, NormalizedDocument, RetrievalCandidate


@dataclass(frozen=True)
class ToolAgentStep:
    step: int
    action: str
    query: str | None = None
    url: str | None = None
    top_k: int | None = None
    reason: str | None = None
    observation: str | None = None


@dataclass(frozen=True)
class ToolAgentVisitedPage:
    canonical_url: str
    title: str
    source_family: str
    fetch_seconds: float
    chunk_count: int
    word_count: int
    cache_hit: bool


def _extract_text_from_chat_response(response: object) -> str:
    message = getattr(response, "message", None)
    contents = getattr(message, "content", None)
    if not contents:
        return ""
    parts: list[str] = []
    for item in contents:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
    return "".join(parts).strip()


def _compact_text(text: str, *, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _tool_agent_prompt(
    *,
    question: str,
    step_index: int,
    max_steps: int,
    steps: list[ToolAgentStep],
    fetched_pages: list[ToolAgentVisitedPage],
    top_search_hits: list[dict[str, object]],
    top_chunk_hits: list[dict[str, object]],
) -> str:
    prior_steps = [
        {
            "step": step.step,
            "action": step.action,
            "query": step.query,
            "url": step.url,
            "top_k": step.top_k,
            "reason": step.reason,
            "observation": step.observation,
        }
        for step in steps
    ]
    fetched = [asdict(page) for page in fetched_pages]
    tool_state = {
        "question": question,
        "step_index": step_index,
        "steps_remaining": max_steps - step_index + 1,
        "prior_steps": prior_steps,
        "fetched_pages": fetched,
        "latest_search_hits": top_search_hits,
        "latest_chunk_hits": top_chunk_hits,
    }
    return (
        "You are controlling a bounded official-site procurement-policy browsing agent.\n"
        "Choose exactly one next action. Return JSON only.\n"
        "Available actions:\n"
        '1. {"action":"search_inventory","query":"refined search query","top_k":1-8,"reason":"why this search helps"}\n'
        '2. {"action":"fetch_page","url":"exact canonical url from a search result or fetched page list","reason":"why this page is worth reading"}\n'
        '3. {"action":"rerank_fetched_chunks","query":"focused sub-question","top_k":1-12,"reason":"why chunk reranking is needed now"}\n'
        '4. {"action":"answer","reason":"why the gathered evidence is sufficient","final_focus_query":"optional focused query for final chunk selection"}\n\n'
        "Rules:\n"
        "1. Prefer the smallest number of actions that could still answer correctly.\n"
        "2. Do not fetch a page you already fetched unless you have a clear reason.\n"
        "3. Use search_inventory when you need candidate pages.\n"
        "4. Use fetch_page only on URLs that appear in the latest_search_hits or fetched_pages.\n"
        "5. Use rerank_fetched_chunks when you already have fetched pages but need to isolate the most relevant passages.\n"
        "6. Choose answer only when the fetched pages are sufficient to support a final answer.\n"
        "7. Keep queries short and operationally useful.\n"
        "8. Do not answer the user directly in this step.\n"
        "9. If you are unsure, search before fetching and fetch before answering.\n\n"
        f"Current tool state:\n{json.dumps(tool_state, ensure_ascii=False, indent=2)}"
    )


def _normalize_action(parsed: dict[str, object]) -> dict[str, object]:
    action = str(parsed.get("action", "")).strip()
    if action not in {"search_inventory", "fetch_page", "rerank_fetched_chunks", "answer"}:
        raise ValueError(f"Unsupported tool agent action: {action}")
    normalized: dict[str, object] = {"action": action}
    if action in {"search_inventory", "rerank_fetched_chunks"}:
        query = " ".join(str(parsed.get("query", "")).split()).strip()
        if not query:
            raise ValueError(f"{action} requires a non-empty query")
        top_k = int(parsed.get("top_k", 0))
        if top_k <= 0:
            raise ValueError(f"{action} requires a positive top_k")
        normalized["query"] = query
        normalized["top_k"] = min(top_k, 12)
    if action == "fetch_page":
        url = " ".join(str(parsed.get("url", "")).split()).strip()
        if not url:
            raise ValueError("fetch_page requires a URL")
        normalized["url"] = url
    if action == "answer":
        focus_query = " ".join(str(parsed.get("final_focus_query", "")).split()).strip()
        if focus_query:
            normalized["final_focus_query"] = focus_query
    reason = " ".join(str(parsed.get("reason", "")).split()).strip()
    normalized["reason"] = reason
    return normalized


class ToolUsingOfficialSiteAgentRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        answer_profile_name: str,
        max_steps: int = 5,
        max_live_pages: int = 6,
        max_live_chunks: int = 24,
    ) -> None:
        settings.require_cohere_key("Tool-using official-site agent baseline")
        self.settings = settings
        self.answer_profile = load_profile(answer_profile_name, settings)
        self.runtime_settings = build_runtime_settings(settings, self.answer_profile)
        self.answer_strategy = answer_strategy_registry.get(self.answer_profile.answering.strategy)
        self.judge = CohereJudge(settings)
        self.client = cohere.ClientV2(settings.cohere_api_key)
        self.max_steps = max_steps
        self.max_live_pages = max_live_pages
        self.max_live_chunks = max_live_chunks
        self.inventory_documents = read_normalized_documents(settings.resolve(Path("datasets/corpus/documents")))
        self.inventory_entries = build_site_inventory_entries(self.inventory_documents, preview_chars=700)
        self._inventory_by_url = {entry.canonical_url: entry for entry in self.inventory_entries}
        self._live_doc_cache: dict[str, NormalizedDocument] = {}

    def _search_inventory(self, query: str, *, top_k: int) -> tuple[list[dict[str, object]], float]:
        start = perf_counter()
        response = self.client.rerank(
            model=self.settings.cohere_rerank_model,
            query=query,
            documents=[entry.inventory_text for entry in self.inventory_entries],
            top_n=min(top_k, len(self.inventory_entries)),
        )
        end = perf_counter()
        hits: list[dict[str, object]] = []
        for result in response.results:
            entry = self.inventory_entries[int(result.index)]
            hits.append(
                {
                    "canonical_url": entry.canonical_url,
                    "title": entry.title,
                    "source_family": entry.source_family,
                    "score": float(result.relevance_score),
                }
            )
        return hits, end - start

    def _fetch_live_document(self, client: httpx.Client, url: str) -> tuple[NormalizedDocument, bool, float]:
        cached = self._live_doc_cache.get(url)
        if cached is not None:
            return cached, True, 0.0
        start = perf_counter()
        fetched = fetch_url(client, url)
        normalized = normalize_document(fetched.document)
        end = perf_counter()
        self._live_doc_cache[url] = normalized
        return normalized, False, end - start

    def _rerank_fetched_chunks(
        self,
        query: str,
        fetched_documents: dict[str, NormalizedDocument],
        *,
        top_k: int,
    ) -> tuple[list[dict[str, object]], list[ChunkRecord], float]:
        chunks: list[ChunkRecord] = []
        for document in fetched_documents.values():
            chunks.extend(
                section_chunker(
                    document,
                    enrichers=self.answer_profile.chunking.metadata_enrichers,
                )
            )
        if not chunks:
            return [], [], 0.0
        start = perf_counter()
        response = self.client.rerank(
            model=self.settings.cohere_rerank_model,
            query=query,
            documents=[
                (
                    f"Title: {chunk.title}\n"
                    f"Heading: {' > '.join(chunk.heading_path) if chunk.heading_path else chunk.title}\n"
                    f"URL: {chunk.canonical_url}\n"
                    f"Text: {chunk.text}"
                )
                for chunk in chunks
            ],
            top_n=min(top_k, len(chunks)),
        )
        end = perf_counter()
        top_chunks: list[ChunkRecord] = []
        top_hits: list[dict[str, object]] = []
        for result in response.results:
            chunk = chunks[int(result.index)]
            top_chunks.append(chunk)
            top_hits.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "canonical_url": chunk.canonical_url,
                    "heading": " > ".join(chunk.heading_path) if chunk.heading_path else chunk.title,
                    "score": float(result.relevance_score),
                    "preview": _compact_text(chunk.text, max_chars=180),
                }
            )
        return top_hits, top_chunks, end - start

    def _plan_next_action(
        self,
        *,
        question: str,
        step_index: int,
        steps: list[ToolAgentStep],
        fetched_pages: list[ToolAgentVisitedPage],
        top_search_hits: list[dict[str, object]],
        top_chunk_hits: list[dict[str, object]],
    ) -> dict[str, object]:
        response = self.client.chat(
            model=self.runtime_settings.cohere_query_planner_model,
            messages=[
                ct.UserChatMessageV2(
                    content=_tool_agent_prompt(
                        question=question,
                        step_index=step_index,
                        max_steps=self.max_steps,
                        steps=steps,
                        fetched_pages=fetched_pages,
                        top_search_hits=top_search_hits,
                        top_chunk_hits=top_chunk_hits,
                    )
                )
            ],
            response_format=ct.JsonObjectResponseFormatV2(),
            temperature=0,
            max_tokens=400,
        )
        parsed = json.loads(_extract_text_from_chat_response(response))
        if not isinstance(parsed, dict):
            raise ValueError("Tool agent planner response must be a JSON object")
        return _normalize_action(parsed)

    def answer_case(self, case: EvalCase) -> AnswerResult:
        start = perf_counter()
        fetched_documents: dict[str, NormalizedDocument] = {}
        visited_pages: list[ToolAgentVisitedPage] = []
        steps: list[ToolAgentStep] = []
        top_search_hits: list[dict[str, object]] = []
        top_chunk_hits: list[dict[str, object]] = []
        current_packed_chunks: list[ChunkRecord] = []
        timings = {
            "planning_seconds": 0.0,
            "inventory_rerank_seconds": 0.0,
            "live_fetch_seconds": 0.0,
            "live_chunk_rerank_seconds": 0.0,
        }

        with httpx.Client(headers={"User-Agent": BROWSER_UA}) as http_client:
            for step_index in range(1, self.max_steps + 1):
                plan_start = perf_counter()
                action = self._plan_next_action(
                    question=case.question,
                    step_index=step_index,
                    steps=steps,
                    fetched_pages=visited_pages,
                    top_search_hits=top_search_hits,
                    top_chunk_hits=top_chunk_hits,
                )
                plan_end = perf_counter()
                timings["planning_seconds"] += plan_end - plan_start

                if action["action"] == "search_inventory":
                    hits, elapsed = self._search_inventory(str(action["query"]), top_k=int(action["top_k"]))
                    top_search_hits = hits
                    timings["inventory_rerank_seconds"] += elapsed
                    observation = f"Found {len(hits)} candidate pages."
                    steps.append(
                        ToolAgentStep(
                            step=step_index,
                            action="search_inventory",
                            query=str(action["query"]),
                            top_k=int(action["top_k"]),
                            reason=str(action.get("reason", "")),
                            observation=observation,
                        )
                    )
                    continue

                if action["action"] == "fetch_page":
                    url = str(action["url"])
                    if url not in self._inventory_by_url and url not in fetched_documents:
                        raise ValueError(f"Planner selected URL outside inventory/fetched set: {url}")
                    document, cache_hit, elapsed = self._fetch_live_document(http_client, url)
                    fetched_documents[document.canonical_url] = document
                    timings["live_fetch_seconds"] += elapsed
                    page_chunks = section_chunker(document, enrichers=self.answer_profile.chunking.metadata_enrichers)
                    visited_pages.append(
                        ToolAgentVisitedPage(
                            canonical_url=document.canonical_url,
                            title=document.title,
                            source_family=document.source_family.value,
                            fetch_seconds=elapsed,
                            chunk_count=len(page_chunks),
                            word_count=document.word_count,
                            cache_hit=cache_hit,
                        )
                    )
                    steps.append(
                        ToolAgentStep(
                            step=step_index,
                            action="fetch_page",
                            url=url,
                            reason=str(action.get("reason", "")),
                            observation=f"Fetched {document.title} with {len(page_chunks)} chunks.",
                        )
                    )
                    continue

                if action["action"] == "rerank_fetched_chunks":
                    top_chunk_hits, current_packed_chunks, elapsed = self._rerank_fetched_chunks(
                        str(action["query"]),
                        fetched_documents,
                        top_k=min(int(action["top_k"]), self.max_live_chunks),
                    )
                    timings["live_chunk_rerank_seconds"] += elapsed
                    steps.append(
                        ToolAgentStep(
                            step=step_index,
                            action="rerank_fetched_chunks",
                            query=str(action["query"]),
                            top_k=int(action["top_k"]),
                            reason=str(action.get("reason", "")),
                            observation=f"Ranked {len(current_packed_chunks)} live chunks.",
                        )
                    )
                    continue

                if action["action"] == "answer":
                    focus_query = str(action.get("final_focus_query", "")).strip() or case.question
                    if not current_packed_chunks:
                        top_chunk_hits, current_packed_chunks, elapsed = self._rerank_fetched_chunks(
                            focus_query,
                            fetched_documents,
                            top_k=self.max_live_chunks,
                        )
                        timings["live_chunk_rerank_seconds"] += elapsed
                    steps.append(
                        ToolAgentStep(
                            step=step_index,
                            action="answer",
                            query=focus_query,
                            reason=str(action.get("reason", "")),
                            observation=f"Answering with {len(current_packed_chunks)} packed chunks.",
                        )
                    )
                    break

        evidence = EvidenceBundle(
            query=case.question,
            candidates=[
                RetrievalCandidate(
                    chunk=chunk,
                    blended_score=1.0 / (index + 1),
                )
                for index, chunk in enumerate(current_packed_chunks)
            ],
            packed_chunks=current_packed_chunks,
            retrieval_queries=[case.question],
            notes=[
                "tool_using_official_site_agent_v1",
                "tool_budget_bounded",
            ],
            timings={
                "inventory_rerank_seconds": timings["inventory_rerank_seconds"],
                "live_fetch_seconds": timings["live_fetch_seconds"],
                "live_chunk_rerank_seconds": timings["live_chunk_rerank_seconds"],
                "retrieval_seconds": timings["inventory_rerank_seconds"]
                + timings["live_fetch_seconds"]
                + timings["live_chunk_rerank_seconds"],
            },
        )
        if not current_packed_chunks:
            return AnswerResult(
                question=case.question,
                answer_text="",
                strategy_name=self.answer_profile.answering.strategy,
                model_name=self.runtime_settings.cohere_chat_model,
                evidence_bundle=evidence,
                raw_response={
                    "tool_agent_trace": {
                        "steps": [asdict(step) for step in steps],
                        "visited_pages": [asdict(page) for page in visited_pages],
                        "latest_search_hits": top_search_hits,
                        "latest_chunk_hits": top_chunk_hits,
                    }
                },
                failure_reason="Tool agent baseline ended without any packed evidence.",
                timings={
                    "query_planning_seconds": timings["planning_seconds"],
                    "retrieval_seconds": evidence.timings["retrieval_seconds"],
                    "answer_generation_seconds": 0.0,
                    "total_answer_path_seconds": perf_counter() - start,
                },
            )

        answer_generation_start = perf_counter()
        result = self.answer_strategy(self.runtime_settings, case.question, evidence)
        answer_generation_end = perf_counter()
        existing_raw = result.raw_response if isinstance(result.raw_response, dict) else {}
        result.raw_response = {
            **existing_raw,
            "tool_agent_trace": {
                "steps": [asdict(step) for step in steps],
                "visited_pages": [asdict(page) for page in visited_pages],
                "latest_search_hits": top_search_hits,
                "latest_chunk_hits": top_chunk_hits,
            },
        }
        result.timings.update(
            {
                "query_planning_seconds": timings["planning_seconds"],
                "retrieval_seconds": evidence.timings["retrieval_seconds"],
                "answer_generation_seconds": answer_generation_end - answer_generation_start,
                "total_answer_path_seconds": answer_generation_end - start,
            }
        )
        return result

    def judge_case(self, case: EvalCase, answer: AnswerResult) -> dict[str, object]:
        return self.judge.judge(case, answer)


def build_tool_agent_run_manifest(
    settings: Settings,
    *,
    eval_path: Path,
    answer_profile_name: str,
    max_steps: int,
    max_live_pages: int,
    max_live_chunks: int,
) -> dict[str, object]:
    resolved_eval_path = settings.resolve(eval_path)
    documents_dir = settings.resolve(Path("datasets/corpus/documents"))
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "tool_using_official_site_agent_v1",
        "workspace_fingerprint": workspace_fingerprint(settings),
        "project_root": str(settings.project_root.resolve()),
        "eval_path": repo_relative_path(settings, resolved_eval_path),
        "eval_sha256": file_sha256(resolved_eval_path),
        "answer_profile_name": answer_profile_name,
        "answer_profile_path": repo_relative_path(settings, settings.resolved_profiles_dir / f"{answer_profile_name}.yaml"),
        "answer_profile_sha256": profile_sha256(settings, answer_profile_name),
        "inventory_documents_dir": repo_relative_path(settings, documents_dir),
        "inventory_documents_sha256": tree_sha256(documents_dir.glob("*.json"), settings.project_root),
        "code_sha256": code_fingerprint(settings),
        "planner_model": build_runtime_settings(settings, load_profile(answer_profile_name, settings)).cohere_query_planner_model,
        "answer_model": load_profile(answer_profile_name, settings).answering.model_name,
        "judge_model": settings.cohere_judge_model,
        "rerank_model": settings.cohere_rerank_model,
        "max_steps": max_steps,
        "max_live_pages": max_live_pages,
        "max_live_chunks": max_live_chunks,
    }


def run_tool_agent_baseline_eval(
    settings: Settings,
    *,
    eval_path: Path,
    answer_profile_name: str = "baseline_vector",
    max_steps: int = 5,
    max_live_pages: int = 6,
    max_live_chunks: int = 24,
    case_limit: int = 0,
) -> EvalRunResult:
    runner = ToolUsingOfficialSiteAgentRunner(
        settings,
        answer_profile_name=answer_profile_name,
        max_steps=max_steps,
        max_live_pages=max_live_pages,
        max_live_chunks=max_live_chunks,
    )
    cases = load_eval_cases(eval_path)
    if case_limit > 0:
        cases = cases[:case_limit]

    case_results: list[EvalCaseResult] = []
    for case in cases:
        case_start = perf_counter()
        answer = runner.answer_case(case)
        before_judge = perf_counter()
        judgment = runner.judge_case(case, answer)
        after_judge = perf_counter()
        packed_metrics = compute_retrieval_metrics(case, answer.evidence_bundle.packed_chunks if answer.evidence_bundle else [])
        candidate_metrics = compute_retrieval_metrics(
            case,
            [candidate.chunk for candidate in answer.evidence_bundle.candidates] if answer.evidence_bundle else [],
        )
        case_results.append(
            EvalCaseResult(
                case=case,
                answer=answer,
                judgment=judgment,
                metrics={
                    "required_claim_recall": float(judgment["required_claim_recall"]),
                    "abstained": answer.abstained,
                    "judge_answer_abstains": bool(judgment["answer_abstains"]),
                    "expect_abstain_annotated": case.expect_abstain is not None,
                    "expect_abstain": case.expect_abstain,
                    "abstain_correct": judgment["abstain_correct"],
                    "failed": bool(answer.failure_reason),
                    "forbidden_claims_clean": bool(judgment["forbidden_claims_clean"]),
                    "forbidden_claim_violation_count": int(judgment["forbidden_claim_violation_count"]),
                    "query_embedding_seconds": 0.0,
                    "retrieval_seconds": answer.timings.get("retrieval_seconds", 0.0),
                    "answer_generation_seconds": answer.timings.get("answer_generation_seconds", 0.0),
                    "judge_seconds": after_judge - before_judge,
                    "total_case_seconds": after_judge - case_start,
                    "packed_primary_url_hit": packed_metrics.primary_url_hit,
                    "candidate_primary_url_hit": candidate_metrics.primary_url_hit,
                    "packed_supporting_url_hit": packed_metrics.supporting_url_hit,
                    "candidate_supporting_url_hit": candidate_metrics.supporting_url_hit,
                    "packed_expected_url_recall": packed_metrics.expected_url_recall,
                    "candidate_expected_url_recall": candidate_metrics.expected_url_recall,
                    "packed_claim_evidence_recall": packed_metrics.claim_evidence_recall,
                    "candidate_claim_evidence_recall": candidate_metrics.claim_evidence_recall,
                    "claim_evidence_annotated": packed_metrics.claim_evidence_annotated,
                },
            )
        )

    profile_name = f"tool_agent_{answer_profile_name}"
    return EvalRunResult(
        run_name=build_run_name(profile_name),
        created_at=datetime.now(timezone.utc),
        profile_name=profile_name,
        answer_model=build_runtime_settings(settings, load_profile(answer_profile_name, settings)).cohere_chat_model,
        judge_model=settings.cohere_judge_model,
        run_manifest=build_tool_agent_run_manifest(
            settings,
            eval_path=eval_path,
            answer_profile_name=answer_profile_name,
            max_steps=max_steps,
            max_live_pages=max_live_pages,
            max_live_chunks=max_live_chunks,
        ),
        cases=case_results,
        overall_metrics=compute_overall_metrics(case_results),
        notes=[
            "Bounded tool-using official-site agent baseline.",
            "Planner chooses among search_inventory, fetch_page, rerank_fetched_chunks, and answer.",
            "Final answer still uses the profile answer strategy on gathered live evidence.",
        ],
    )


def render_tool_agent_baseline_markdown(run: EvalRunResult) -> str:
    lines = [
        f"# Tool Agent Baseline: {run.profile_name}",
        "",
        f"- run_name: {run.run_name}",
        f"- created_at: {run.created_at.isoformat()}",
        f"- answer_model: {run.answer_model}",
        f"- judge_model: {run.judge_model}",
        "",
        "## Overall Metrics",
        "",
    ]
    for key, value in run.overall_metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Cases", ""])
    for case in run.cases:
        lines.append(f"### {case.case.id}")
        lines.append(f"- question: {case.case.question}")
        lines.append(f"- required_claim_recall: {case.metrics.get('required_claim_recall')}")
        lines.append(f"- total_case_seconds: {case.metrics.get('total_case_seconds')}")
        raw = case.answer.raw_response or {}
        trace = raw.get("tool_agent_trace") if isinstance(raw, dict) else None
        if isinstance(trace, dict):
            steps = trace.get("steps", [])
            if steps:
                lines.append("- steps:")
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    lines.append(
                        f"  - {step.get('step')}. {step.get('action')} | query={step.get('query')} | url={step.get('url')} | reason={step.get('reason')}"
                    )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_tool_agent_baseline_artifacts(
    settings: Settings,
    run: EvalRunResult,
) -> tuple[Path, Path]:
    output_dir = settings.resolved_runs_dir / "tool_agent_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    md_path = output_dir / f"{run.run_name}.md"
    json_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_tool_agent_baseline_markdown(run), encoding="utf-8")
    return json_path, md_path
