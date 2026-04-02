"""Product-serving benchmark helpers for chat-style RAG checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from bgrag.config import Settings
from bgrag.demo_server import run_demo_query
from bgrag.manifests import build_run_name


class ProductChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ProductBenchmarkCase(BaseModel):
    id: str
    question: str
    description: str | None = None
    messages: list[ProductChatTurn] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    quality_focus: list[str] = Field(default_factory=list)
    notes: str | None = None


class ProductBenchmarkManifest(BaseModel):
    name: str
    version: int
    description: str
    cases: list[ProductBenchmarkCase]
    notes: list[str] = Field(default_factory=list)


class ProductBenchmarkCaseResult(BaseModel):
    case_id: str
    question: str
    description: str | None = None
    resolved_question: str | None = None
    answer_text: str | None = None
    citation_count: int = 0
    response_mode: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    quality_focus: list[str] = Field(default_factory=list)
    error: str | None = None


class ProductBenchmarkRunResult(BaseModel):
    run_name: str
    created_at: datetime
    profile_name: str
    manifest_name: str
    manifest_path: str
    cases: list[ProductBenchmarkCaseResult]
    summary: dict[str, object] = Field(default_factory=dict)


def load_product_benchmark_manifest(path: Path) -> ProductBenchmarkManifest:
    return ProductBenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def evaluate_product_case(
    settings: Settings,
    profile_name: str,
    case: ProductBenchmarkCase,
) -> ProductBenchmarkCaseResult:
    wall_start = perf_counter()
    try:
        payload = run_demo_query(
            settings,
            case.question,
            profile_name=profile_name,
            messages=[turn.model_dump() for turn in case.messages],
        )
        elapsed_wall_seconds = perf_counter() - wall_start
    except Exception as exc:
        return ProductBenchmarkCaseResult(
            case_id=case.id,
            question=case.question,
            description=case.description,
            tags=case.tags,
            quality_focus=case.quality_focus,
            error=repr(exc),
            timings={"elapsed_wall_seconds": perf_counter() - wall_start},
        )

    resolved_question = str(payload.get("resolved_question", "")).strip()
    answer_text = str(payload.get("answer_text", "")).strip()
    response_mode = str(payload.get("response_mode", "")).strip() or None
    citation_count = len(payload.get("citations", []))
    timings = {
        key: float(value)
        for key, value in dict(payload.get("timings", {})).items()
        if isinstance(value, (int, float))
    }
    timings["elapsed_wall_seconds"] = elapsed_wall_seconds

    return ProductBenchmarkCaseResult(
        case_id=case.id,
        question=case.question,
        description=case.description,
        resolved_question=resolved_question or None,
        answer_text=answer_text or None,
        citation_count=citation_count,
        response_mode=response_mode,
        timings=timings,
        notes=[str(item) for item in payload.get("notes", [])],
        tags=case.tags,
        quality_focus=case.quality_focus,
    )


def summarize_product_benchmark(results: list[ProductBenchmarkCaseResult]) -> dict[str, object]:
    successful = [result for result in results if result.error is None]
    failed = [result for result in results if result.error is not None]

    def timing_stats(key: str) -> dict[str, float] | None:
        values = [result.timings[key] for result in successful if key in result.timings]
        if not values:
            return None
        return {
            "mean": mean(values),
            "p50": _quantile(values, 0.50) or 0.0,
            "p95": _quantile(values, 0.95) or 0.0,
            "max": max(values),
        }

    return {
        "case_count": len(results),
        "successful_case_count": len(successful),
        "error_case_count": len(failed),
        "error_case_ids": [result.case_id for result in failed],
        "stage_timings": {
            key: timing_stats(key)
            for key in (
                "contextualization_seconds",
                "query_planning_seconds",
                "query_embedding_seconds",
                "retrieval_seconds",
                "answer_generation_seconds",
                "total_answer_path_seconds",
                "total_request_seconds",
                "elapsed_wall_seconds",
            )
        },
    }


def run_product_benchmark(
    settings: Settings,
    manifest_path: Path,
    profile_name: str,
) -> ProductBenchmarkRunResult:
    manifest = load_product_benchmark_manifest(manifest_path)
    results = [evaluate_product_case(settings, profile_name, case) for case in manifest.cases]
    return ProductBenchmarkRunResult(
        run_name=build_run_name(f"product_benchmark_{profile_name}"),
        created_at=datetime.now(timezone.utc),
        profile_name=profile_name,
        manifest_name=manifest.name,
        manifest_path=str(manifest_path),
        cases=results,
        summary=summarize_product_benchmark(results),
    )


def _answer_preview(text: str | None, max_chars: int = 240) -> str:
    if not text:
        return "<empty>"
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def render_product_benchmark_markdown(run: ProductBenchmarkRunResult) -> str:
    summary = run.summary
    lines = [
        f"# Product Benchmark: {run.profile_name}",
        "",
        f"- run_name: {run.run_name}",
        f"- created_at: {run.created_at.isoformat()}",
        f"- manifest: {run.manifest_name}",
        f"- case_count: {summary.get('case_count')}",
        f"- successful_case_count: {summary.get('successful_case_count')}",
        f"- error_case_count: {summary.get('error_case_count')}",
        f"- error_case_ids: {', '.join(summary.get('error_case_ids', [])) or '<none>'}",
        "",
        "## Stage Timings",
        "",
    ]
    stage_timings = summary.get("stage_timings", {})
    if isinstance(stage_timings, dict):
        for key, stats in stage_timings.items():
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"- {key}: mean={stats['mean']:.3f}s p50={stats['p50']:.3f}s p95={stats['p95']:.3f}s max={stats['max']:.3f}s"
            )

    lines.extend(["", "## Cases", ""])
    for case in run.cases:
        lines.append(f"### {case.case_id}")
        lines.append(f"- question: {case.question}")
        if case.description:
            lines.append(f"- description: {case.description}")
        if case.tags:
            lines.append(f"- tags: {', '.join(case.tags)}")
        if case.quality_focus:
            lines.append(f"- quality_focus: {', '.join(case.quality_focus)}")
        if case.resolved_question:
            lines.append(f"- resolved_question: {case.resolved_question}")
        if case.error:
            lines.append(f"- error: {case.error}")
        else:
            lines.append(f"- response_mode: {case.response_mode or '<none>'}")
            lines.append(f"- citation_count: {case.citation_count}")
            lines.append(
                f"- total_request_seconds: {case.timings.get('total_request_seconds', case.timings.get('elapsed_wall_seconds', 0.0)):.3f}"
            )
            lines.append(f"- answer_preview: {_answer_preview(case.answer_text)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_product_benchmark_artifacts(
    settings: Settings,
    run: ProductBenchmarkRunResult,
) -> tuple[Path, Path]:
    output_dir = settings.resolved_runs_dir / "product_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    md_path = output_dir / f"{run.run_name}.md"
    json_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")
    md_path.write_text(render_product_benchmark_markdown(run), encoding="utf-8")
    return json_path, md_path
