"""Bounded official-site live browsing baseline for eval comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cohere
import httpx

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
from bgrag.profiles.loader import load_profile
from bgrag.profiles.runtime import build_runtime_settings
from bgrag.registry import answer_strategy_registry
from bgrag.types import AnswerResult, ChunkRecord, EvalCase, EvalCaseResult, EvalRunResult, EvidenceBundle, NormalizedDocument, RetrievalCandidate


@dataclass(frozen=True)
class SiteInventoryEntry:
    canonical_url: str
    title: str
    source_family: str
    inventory_text: str


@dataclass(frozen=True)
class OfficialSiteBrowseBudget:
    inventory_preview_chars: int = 700
    max_live_pages: int = 6
    max_live_chunks: int = 24


@dataclass(frozen=True)
class VisitedLivePage:
    canonical_url: str
    title: str
    source_family: str
    page_selection_score: float
    fetch_seconds: float
    chunk_count: int
    word_count: int


def _compact_text(text: str, *, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def build_site_inventory_entries(
    documents: list[NormalizedDocument],
    *,
    preview_chars: int = 700,
) -> list[SiteInventoryEntry]:
    entries: list[SiteInventoryEntry] = []
    for document in documents:
        breadcrumbs = " > ".join(link.title for link in document.breadcrumbs[:4] if link.title)
        preview = _compact_text(document.raw_text, max_chars=preview_chars)
        inventory_text = (
            f"Title: {document.title}\n"
            f"URL: {document.canonical_url}\n"
            f"Source family: {document.source_family.value}\n"
            f"Breadcrumbs: {breadcrumbs or '<none>'}\n"
            f"Preview: {preview}"
        )
        entries.append(
            SiteInventoryEntry(
                canonical_url=document.canonical_url,
                title=document.title,
                source_family=document.source_family.value,
                inventory_text=inventory_text,
            )
        )
    return entries


def _rerank_texts(
    client: cohere.ClientV2,
    *,
    model: str,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[tuple[int, float]]:
    response = client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
    )
    return [(int(item.index), float(item.relevance_score)) for item in response.results]


class OfficialSiteBaselineRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        answer_profile_name: str,
        budget: OfficialSiteBrowseBudget | None = None,
        inventory_preview_chars: int = 700,
        max_live_pages: int = 6,
        max_live_chunks: int = 24,
    ) -> None:
        settings.require_cohere_key("Official-site live baseline")
        resolved_budget = budget or OfficialSiteBrowseBudget(
            inventory_preview_chars=inventory_preview_chars,
            max_live_pages=max_live_pages,
            max_live_chunks=max_live_chunks,
        )
        self.settings = settings
        self.budget = resolved_budget
        self.answer_profile = load_profile(answer_profile_name, settings)
        self.runtime_settings = build_runtime_settings(settings, self.answer_profile)
        self.answer_strategy = answer_strategy_registry.get(self.answer_profile.answering.strategy)
        self.judge = CohereJudge(settings)
        self.rerank_client = cohere.ClientV2(settings.cohere_api_key)
        self.max_live_pages = resolved_budget.max_live_pages
        self.max_live_chunks = resolved_budget.max_live_chunks
        self.inventory_documents = read_normalized_documents(settings.resolve(Path("datasets/corpus/documents")))
        self.inventory_entries = build_site_inventory_entries(
            self.inventory_documents,
            preview_chars=resolved_budget.inventory_preview_chars,
        )
        self._live_doc_cache: dict[str, NormalizedDocument] = {}

    def _select_live_pages(self, question: str) -> list[tuple[SiteInventoryEntry, float]]:
        ranked = _rerank_texts(
            self.rerank_client,
            model=self.settings.cohere_rerank_model,
            query=question,
            documents=[entry.inventory_text for entry in self.inventory_entries],
            top_n=self.max_live_pages,
        )
        return [(self.inventory_entries[index], score) for index, score in ranked]

    def _fetch_live_document(self, client: httpx.Client, url: str) -> NormalizedDocument:
        cached = self._live_doc_cache.get(url)
        if cached is not None:
            return cached
        fetched = fetch_url(client, url)
        normalized = normalize_document(fetched.document)
        self._live_doc_cache[url] = normalized
        return normalized

    def _build_live_evidence(
        self,
        question: str,
    ) -> tuple[EvidenceBundle, dict[str, object], dict[str, float]]:
        selection_start = perf_counter()
        selected_pages = self._select_live_pages(question)
        selection_end = perf_counter()

        live_chunks: list[ChunkRecord] = []
        visited_pages: list[VisitedLivePage] = []
        fetch_start = perf_counter()
        with httpx.Client(headers={"User-Agent": BROWSER_UA}) as client:
            for entry, score in selected_pages:
                page_fetch_start = perf_counter()
                live_document = self._fetch_live_document(client, entry.canonical_url)
                page_fetch_end = perf_counter()
                chunks = section_chunker(
                    live_document,
                    enrichers=self.answer_profile.chunking.metadata_enrichers,
                )
                live_chunks.extend(chunks)
                visited_pages.append(
                    VisitedLivePage(
                        canonical_url=live_document.canonical_url,
                        title=live_document.title,
                        source_family=live_document.source_family.value,
                        page_selection_score=score,
                        fetch_seconds=page_fetch_end - page_fetch_start,
                        chunk_count=len(chunks),
                        word_count=live_document.word_count,
                    )
                )
        fetch_end = perf_counter()

        chunk_rerank_start = perf_counter()
        ranked_chunk_refs = _rerank_texts(
            self.rerank_client,
            model=self.settings.cohere_rerank_model,
            query=question,
            documents=[
                (
                    f"Title: {chunk.title}\n"
                    f"Heading: {' > '.join(chunk.heading_path) if chunk.heading_path else chunk.title}\n"
                    f"URL: {chunk.canonical_url}\n"
                    f"Text: {chunk.text}"
                )
                for chunk in live_chunks
            ],
            top_n=min(self.max_live_chunks, len(live_chunks)),
        ) if live_chunks else []
        chunk_rerank_end = perf_counter()

        ranked_candidates: list[RetrievalCandidate] = []
        packed_chunks: list[ChunkRecord] = []
        for index, score in ranked_chunk_refs:
            chunk = live_chunks[index]
            ranked_candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    rerank_score=score,
                    blended_score=score,
                )
            )
            packed_chunks.append(chunk)

        evidence = EvidenceBundle(
            query=question,
            candidates=ranked_candidates,
            packed_chunks=packed_chunks,
            retrieval_queries=[question],
            notes=[
                "official_site_live_browse",
                "inventory_rerank_applied",
                "live_chunk_rerank_applied",
            ],
            timings={
                "inventory_rerank_seconds": selection_end - selection_start,
                "live_fetch_seconds": fetch_end - fetch_start,
                "live_chunk_rerank_seconds": chunk_rerank_end - chunk_rerank_start,
                "retrieval_seconds": (selection_end - selection_start)
                + (fetch_end - fetch_start)
                + (chunk_rerank_end - chunk_rerank_start),
            },
        )
        trace = {
            "visited_pages": [asdict(page) for page in visited_pages],
            "live_chunk_count": len(live_chunks),
            "packed_chunk_count": len(packed_chunks),
        }
        timings = {
            "inventory_rerank_seconds": selection_end - selection_start,
            "live_fetch_seconds": fetch_end - fetch_start,
            "live_chunk_rerank_seconds": chunk_rerank_end - chunk_rerank_start,
        }
        return evidence, trace, timings

    def answer_case(self, case: EvalCase) -> AnswerResult:
        answer_start = perf_counter()
        try:
            evidence, trace, browse_timings = self._build_live_evidence(case.question)
            if not evidence.packed_chunks:
                return AnswerResult(
                    question=case.question,
                    answer_text="",
                    strategy_name=self.answer_profile.answering.strategy,
                    model_name=self.runtime_settings.cohere_chat_model,
                    evidence_bundle=evidence,
                    raw_response=trace,
                    failure_reason="No live evidence was retrieved from the official site baseline.",
                    timings={
                        **browse_timings,
                        "retrieval_seconds": sum(browse_timings.values()),
                        "answer_generation_seconds": 0.0,
                    },
                )
            strategy = self.answer_strategy
            answer_generation_start = perf_counter()
            result = strategy(self.runtime_settings, case.question, evidence)
            answer_generation_end = perf_counter()
            existing_raw = result.raw_response if isinstance(result.raw_response, dict) else {}
            result.raw_response = {**existing_raw, "official_site_live_browse": trace}
            result.timings.update(
                {
                    **browse_timings,
                    "retrieval_seconds": sum(browse_timings.values()),
                    "answer_generation_seconds": answer_generation_end - answer_generation_start,
                    "total_answer_path_seconds": answer_generation_end - answer_start,
                }
            )
            return result
        except Exception as exc:
            return AnswerResult(
                question=case.question,
                answer_text="",
                strategy_name=self.answer_profile.answering.strategy,
                model_name=self.runtime_settings.cohere_chat_model,
                evidence_bundle=EvidenceBundle(query=case.question),
                raw_response={"official_site_live_browse": {"error": repr(exc)}},
                failure_reason=repr(exc),
                timings={
                    "answer_generation_seconds": 0.0,
                    "total_answer_path_seconds": perf_counter() - answer_start,
                },
            )

    def judge_case(self, case: EvalCase, answer: AnswerResult) -> dict[str, object]:
        return self.judge.judge(case, answer)


def build_official_site_run_manifest(
    settings: Settings,
    *,
    eval_path: Path,
    answer_profile_name: str,
    budget: OfficialSiteBrowseBudget | None = None,
    max_live_pages: int | None = None,
    max_live_chunks: int | None = None,
    inventory_preview_chars: int | None = None,
) -> dict[str, object]:
    resolved_budget = budget or OfficialSiteBrowseBudget(
        inventory_preview_chars=inventory_preview_chars or 700,
        max_live_pages=max_live_pages or 6,
        max_live_chunks=max_live_chunks or 24,
    )
    resolved_eval_path = settings.resolve(eval_path)
    documents_dir = settings.resolve(Path("datasets/corpus/documents"))
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "official_site_live_browse_v1",
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
        "answer_model": load_profile(answer_profile_name, settings).answering.model_name,
        "judge_model": settings.cohere_judge_model,
        "rerank_model": settings.cohere_rerank_model,
        "inventory_preview_chars": resolved_budget.inventory_preview_chars,
        "max_live_pages": resolved_budget.max_live_pages,
        "max_live_chunks": resolved_budget.max_live_chunks,
    }


def run_official_site_baseline_eval(
    settings: Settings,
    *,
    eval_path: Path,
    answer_profile_name: str = "baseline_vector",
    budget: OfficialSiteBrowseBudget | None = None,
    max_live_pages: int = 6,
    max_live_chunks: int = 24,
    case_limit: int = 0,
) -> EvalRunResult:
    resolved_budget = budget or OfficialSiteBrowseBudget(
        max_live_pages=max_live_pages,
        max_live_chunks=max_live_chunks,
    )
    runner = OfficialSiteBaselineRunner(
        settings,
        answer_profile_name=answer_profile_name,
        budget=resolved_budget,
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

    profile_name = f"official_site_live_{answer_profile_name}"
    return EvalRunResult(
        run_name=build_run_name(profile_name),
        created_at=datetime.now(timezone.utc),
        profile_name=profile_name,
        answer_model=runner.runtime_settings.cohere_chat_model,
        judge_model=settings.cohere_judge_model,
        run_manifest=build_official_site_run_manifest(
            settings,
            eval_path=eval_path,
            answer_profile_name=answer_profile_name,
            budget=resolved_budget,
        ),
        cases=case_results,
        overall_metrics=compute_overall_metrics(case_results),
        notes=[
            "Bounded official-site live browsing baseline.",
            "Page selection uses local URL inventory plus Cohere rerank, then fetches live official pages before answering.",
        ],
    )


def render_official_site_baseline_markdown(run: EvalRunResult) -> str:
    lines = [
        f"# Official-Site Live Baseline: {run.profile_name}",
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
        visited_pages = []
        raw = case.answer.raw_response or {}
        browse = raw.get("official_site_live_browse") if isinstance(raw, dict) else None
        if isinstance(browse, dict):
            visited_pages = browse.get("visited_pages", [])
        if visited_pages:
            lines.append("- visited_pages:")
            for page in visited_pages:
                if not isinstance(page, dict):
                    continue
                lines.append(
                    f"  - {page.get('canonical_url')} | score={page.get('page_selection_score')} | chunks={page.get('chunk_count')}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_official_site_baseline_artifacts(
    settings: Settings,
    run: EvalRunResult,
) -> tuple[Path, Path]:
    output_dir = settings.resolved_runs_dir / "official_site_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    md_path = output_dir / f"{run.run_name}.md"
    json_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_official_site_baseline_markdown(run), encoding="utf-8")
    return json_path, md_path
