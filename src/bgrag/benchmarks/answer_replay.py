"""Answer-only replay benchmark for isolating final answer-path behavior."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from pydantic import BaseModel, Field

from bgrag.answering import strategies as _answering_strategies  # noqa: F401
from bgrag.config import Settings
from bgrag.eval.judge import CohereJudge
from bgrag.manifests import build_run_name
from bgrag.registry import answer_strategy_registry
from bgrag.types import AnswerResult, EvalCaseResult, EvalRunResult


class AnswerReplayCaseResult(BaseModel):
    case_id: str
    question: str
    strategy_name: str
    answer_model: str
    source_profile_name: str
    citation_count: int = 0
    answer_generation_seconds: float = 0.0
    required_claim_recall: float = 0.0
    forbidden_claims_clean: bool = True
    answer_abstains: bool = False
    abstain_correct: bool | None = None
    answer_text: str
    raw_response: dict[str, object] | None = None
    error: str | None = None


class AnswerReplayBenchmarkRun(BaseModel):
    run_name: str
    created_at: datetime
    source_run_path: str
    source_profile_name: str
    strategy_name: str
    answer_model: str
    case_filter_ids: list[str] = Field(default_factory=list)
    case_results: list[AnswerReplayCaseResult]
    summary: dict[str, object] = Field(default_factory=dict)


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def _summarize(run: AnswerReplayBenchmarkRun) -> dict[str, object]:
    successful = [case for case in run.case_results if case.error is None]
    answer_times = [case.answer_generation_seconds for case in successful]
    citation_counts = [case.citation_count for case in successful]
    abstain_annotated = [case for case in successful if case.abstain_correct is not None]
    return {
        "case_count": len(run.case_results),
        "successful_case_count": len(successful),
        "error_case_count": len(run.case_results) - len(successful),
        "required_claim_recall_mean": mean(case.required_claim_recall for case in successful) if successful else 0.0,
        "forbidden_claims_clean_rate": (
            mean(1.0 if case.forbidden_claims_clean else 0.0 for case in successful) if successful else 0.0
        ),
        "answer_abstains_rate": (
            mean(1.0 if case.answer_abstains else 0.0 for case in successful) if successful else 0.0
        ),
        "abstain_correct_rate_annotated": (
            mean(1.0 if case.abstain_correct else 0.0 for case in abstain_annotated) if abstain_annotated else None
        ),
        "citation_count_mean": mean(citation_counts) if citation_counts else 0.0,
        "nonzero_citation_rate": (
            mean(1.0 if count > 0 else 0.0 for count in citation_counts) if citation_counts else 0.0
        ),
        "answer_generation_seconds_mean": mean(answer_times) if answer_times else 0.0,
        "answer_generation_seconds_p50": _quantile(answer_times, 0.50),
        "answer_generation_seconds_p95": _quantile(answer_times, 0.95),
    }


def run_answer_replay_benchmark(
    settings: Settings,
    *,
    source_run_path: Path,
    strategy_name: str,
    answer_model: str | None = None,
    case_filter_ids: list[str] | None = None,
) -> AnswerReplayBenchmarkRun:
    resolved_source = settings.resolve(source_run_path)
    source_run = EvalRunResult.model_validate_json(resolved_source.read_text(encoding="utf-8"))
    model_name = answer_model or source_run.answer_model
    runtime_settings = settings.model_copy(update={"cohere_chat_model": model_name})
    answer_strategy = answer_strategy_registry.get(strategy_name)
    judge = CohereJudge(settings)
    selected_case_ids = [case_id.strip() for case_id in (case_filter_ids or []) if case_id.strip()]
    selected_case_id_set = set(selected_case_ids)

    case_results: list[AnswerReplayCaseResult] = []
    for case_result in source_run.cases:
        if selected_case_id_set and case_result.case.id not in selected_case_id_set:
            continue
        if case_result.answer.evidence_bundle is None:
            case_results.append(
                AnswerReplayCaseResult(
                    case_id=case_result.case.id,
                    question=case_result.case.question,
                    strategy_name=strategy_name,
                    answer_model=model_name,
                    source_profile_name=source_run.profile_name,
                    answer_text="",
                    error="missing_evidence_bundle",
                )
            )
            continue

        answer_start = perf_counter()
        try:
            replayed_answer: AnswerResult = answer_strategy(
                runtime_settings,
                case_result.case.question,
                case_result.answer.evidence_bundle,
            )
        except Exception as exc:
            case_results.append(
                AnswerReplayCaseResult(
                    case_id=case_result.case.id,
                    question=case_result.case.question,
                    strategy_name=strategy_name,
                    answer_model=model_name,
                    source_profile_name=source_run.profile_name,
                    answer_text="",
                    error=repr(exc),
                )
            )
            continue
        answer_seconds = perf_counter() - answer_start
        judgment = judge.judge(case_result.case, replayed_answer)
        abstain_correct = judgment.get("abstain_correct")
        case_results.append(
            AnswerReplayCaseResult(
                case_id=case_result.case.id,
                question=case_result.case.question,
                strategy_name=strategy_name,
                answer_model=model_name,
                source_profile_name=source_run.profile_name,
                citation_count=len(replayed_answer.citations),
                answer_generation_seconds=answer_seconds,
                required_claim_recall=float(judgment["required_claim_recall"]),
                forbidden_claims_clean=bool(judgment["forbidden_claims_clean"]),
                answer_abstains=bool(judgment["answer_abstains"]),
                abstain_correct=bool(abstain_correct) if abstain_correct is not None else None,
                answer_text=replayed_answer.answer_text,
                raw_response=replayed_answer.raw_response,
            )
        )

    run = AnswerReplayBenchmarkRun(
        run_name=build_run_name(f"answer_replay_{strategy_name}"),
        created_at=datetime.now(timezone.utc),
        source_run_path=str(resolved_source),
        source_profile_name=source_run.profile_name,
        strategy_name=strategy_name,
        answer_model=model_name,
        case_filter_ids=selected_case_ids,
        case_results=case_results,
    )
    run.summary = _summarize(run)
    return run


def render_answer_replay_markdown(run: AnswerReplayBenchmarkRun) -> str:
    lines = [
        f"# Answer Replay Benchmark: {run.strategy_name}",
        "",
        f"- run_name: {run.run_name}",
        f"- source_run_path: {run.source_run_path}",
        f"- source_profile_name: {run.source_profile_name}",
        f"- answer_model: {run.answer_model}",
        f"- case_filter_ids: {', '.join(run.case_filter_ids) if run.case_filter_ids else '<all>'}",
        f"- case_count: {run.summary.get('case_count')}",
        f"- successful_case_count: {run.summary.get('successful_case_count')}",
        f"- error_case_count: {run.summary.get('error_case_count')}",
        f"- required_claim_recall_mean: {run.summary.get('required_claim_recall_mean', 0.0):.4f}",
        f"- forbidden_claims_clean_rate: {run.summary.get('forbidden_claims_clean_rate', 0.0):.4f}",
        f"- citation_count_mean: {run.summary.get('citation_count_mean', 0.0):.4f}",
        f"- nonzero_citation_rate: {run.summary.get('nonzero_citation_rate', 0.0):.4f}",
        (
            f"- answer_generation_seconds: mean={run.summary.get('answer_generation_seconds_mean', 0.0):.3f}s "
            f"p50={run.summary.get('answer_generation_seconds_p50', 0.0) or 0.0:.3f}s "
            f"p95={run.summary.get('answer_generation_seconds_p95', 0.0) or 0.0:.3f}s"
        ),
        "",
        "## Cases",
        "",
    ]
    for case in run.case_results:
        lines.append(f"### {case.case_id}")
        lines.append(f"- required_claim_recall: {case.required_claim_recall:.4f}")
        lines.append(f"- citation_count: {case.citation_count}")
        lines.append(f"- answer_generation_seconds: {case.answer_generation_seconds:.3f}")
        if case.error:
            lines.append(f"- error: {case.error}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_answer_replay_artifacts(
    settings: Settings,
    run: AnswerReplayBenchmarkRun,
) -> tuple[Path, Path]:
    output_dir = settings.resolved_runs_dir / "answer_replay_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    md_path = output_dir / f"{run.run_name}.md"
    json_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")
    md_path.write_text(render_answer_replay_markdown(run), encoding="utf-8")
    return json_path, md_path
