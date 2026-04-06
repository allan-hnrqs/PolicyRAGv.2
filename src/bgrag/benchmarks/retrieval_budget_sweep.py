"""Bounded retrieval-budget sweep helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from bgrag.benchmarks.retrieval import run_retrieval_benchmark, write_retrieval_benchmark_artifacts
from bgrag.config import Settings
from bgrag.eval.runner import run_eval
from bgrag.manifests import build_run_name
from bgrag.pipeline import build_answer_callback
from bgrag.profiles.loader import load_profile
from bgrag.profiles.models import RuntimeProfile


class RetrievalBudgetVariant(BaseModel):
    label: str
    top_k: int
    candidate_k: int
    rerank_top_n: int
    per_query_candidate_k: int
    max_expanded_queries: int = 2


class RetrievalBudgetSweepVariantResult(BaseModel):
    label: str
    effective_profile_name: str
    retrieval_overrides: dict[str, int]
    retrieval_json_path: str
    retrieval_markdown_path: str
    eval_json_path: str
    eval_overall_metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    retrieval_overall_metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class RetrievalBudgetSweepRun(BaseModel):
    run_name: str
    created_at: datetime
    base_profile_name: str
    eval_path: str
    query_mode: str
    variants: list[RetrievalBudgetSweepVariantResult] = Field(default_factory=list)
    summary: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


def default_budget_variants() -> list[RetrievalBudgetVariant]:
    return [
        RetrievalBudgetVariant(label="top16_c48_r48_p24", top_k=16, candidate_k=48, rerank_top_n=48, per_query_candidate_k=24),
        RetrievalBudgetVariant(label="top16_c64_r64_p32", top_k=16, candidate_k=64, rerank_top_n=64, per_query_candidate_k=32),
        RetrievalBudgetVariant(label="top16_c96_r96_p48", top_k=16, candidate_k=96, rerank_top_n=96, per_query_candidate_k=48),
        RetrievalBudgetVariant(label="top24_c64_r64_p32", top_k=24, candidate_k=64, rerank_top_n=64, per_query_candidate_k=32),
        RetrievalBudgetVariant(label="top24_c96_r96_p48", top_k=24, candidate_k=96, rerank_top_n=96, per_query_candidate_k=48),
    ]


def build_variant_profile(base_profile: RuntimeProfile, variant: RetrievalBudgetVariant) -> RuntimeProfile:
    profile = base_profile.model_copy(deep=True)
    profile.name = f"{base_profile.name}_{variant.label}"
    profile.retrieval.top_k = variant.top_k
    profile.retrieval.candidate_k = variant.candidate_k
    profile.retrieval.rerank_top_n = variant.rerank_top_n
    profile.retrieval.per_query_candidate_k = variant.per_query_candidate_k
    profile.retrieval.max_expanded_queries = variant.max_expanded_queries
    return profile


def _write_eval_json(settings: Settings, *, run_name: str, payload: dict[str, object]) -> Path:
    output_dir = settings.resolve(Path("datasets/runs/retrieval_budget_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _variant_summary(variant: RetrievalBudgetSweepVariantResult) -> tuple[float, float, float]:
    quality = float(variant.eval_overall_metrics.get("required_claim_recall_mean", 0.0) or 0.0)
    speed = float(variant.eval_overall_metrics.get("mean_case_seconds", 0.0) or 0.0)
    retrieval = float(variant.retrieval_overall_metrics.get("packed_expected_url_recall_mean", 0.0) or 0.0)
    return quality, speed, retrieval


def _summarize_variants(variants: list[RetrievalBudgetSweepVariantResult]) -> dict[str, float | int | str | bool | None]:
    if not variants:
        return {"variant_count": 0}
    best_quality = max(variants, key=lambda item: _variant_summary(item)[0])
    fastest = min(variants, key=lambda item: _variant_summary(item)[1])
    best_retrieval = max(variants, key=lambda item: _variant_summary(item)[2])
    return {
        "variant_count": len(variants),
        "best_quality_label": best_quality.label,
        "best_quality_required_claim_recall_mean": _variant_summary(best_quality)[0],
        "fastest_label": fastest.label,
        "fastest_mean_case_seconds": _variant_summary(fastest)[1],
        "best_retrieval_label": best_retrieval.label,
        "best_retrieval_packed_expected_url_recall_mean": _variant_summary(best_retrieval)[2],
    }


def run_retrieval_budget_sweep(
    settings: Settings,
    *,
    base_profile_name: str,
    eval_path: Path,
    query_mode: str = "profile",
    index_namespace: str | None = None,
    variants: list[RetrievalBudgetVariant] | None = None,
) -> RetrievalBudgetSweepRun:
    resolved_eval_path = settings.resolve(eval_path)
    base_profile = load_profile(base_profile_name, settings)
    effective_variants = variants or default_budget_variants()

    results: list[RetrievalBudgetSweepVariantResult] = []
    for variant in effective_variants:
        variant_profile = build_variant_profile(base_profile, variant)
        retrieval_run = run_retrieval_benchmark(
            settings,
            eval_path=resolved_eval_path,
            profile_name=base_profile_name,
            query_mode=query_mode,
            index_namespace=index_namespace,
            runtime_profile=variant_profile,
        )
        retrieval_json_path, retrieval_markdown_path = write_retrieval_benchmark_artifacts(settings, retrieval_run)
        answer_callback = build_answer_callback(
            settings,
            base_profile_name,
            index_namespace=index_namespace,
            runtime_profile=variant_profile,
        )
        eval_run = run_eval(
            settings,
            variant_profile,
            resolved_eval_path,
            answer_callback,
            run_manifest={
                "base_profile_name": base_profile_name,
                "variant_label": variant.label,
                "retrieval_overrides": variant.model_dump(mode="json"),
            },
        )
        eval_json_path = _write_eval_json(
            settings,
            run_name=eval_run.run_name,
            payload=eval_run.model_dump(mode="json"),
        )
        results.append(
            RetrievalBudgetSweepVariantResult(
                label=variant.label,
                effective_profile_name=variant_profile.name,
                retrieval_overrides={
                    "top_k": variant.top_k,
                    "candidate_k": variant.candidate_k,
                    "rerank_top_n": variant.rerank_top_n,
                    "per_query_candidate_k": variant.per_query_candidate_k,
                    "max_expanded_queries": variant.max_expanded_queries,
                },
                retrieval_json_path=str(retrieval_json_path),
                retrieval_markdown_path=str(retrieval_markdown_path),
                eval_json_path=str(eval_json_path),
                eval_overall_metrics=dict(eval_run.overall_metrics),
                retrieval_overall_metrics=dict(retrieval_run.overall_metrics),
            )
        )

    return RetrievalBudgetSweepRun(
        run_name=build_run_name(f"{base_profile_name}_budget_sweep"),
        created_at=datetime.now(timezone.utc),
        base_profile_name=base_profile_name,
        eval_path=str(resolved_eval_path),
        query_mode=query_mode,
        variants=results,
        summary=_summarize_variants(results),
    )


def render_retrieval_budget_sweep_markdown(run: RetrievalBudgetSweepRun) -> str:
    lines = [
        "# Retrieval Budget Sweep",
        "",
        f"- run_name: {run.run_name}",
        f"- created_at: {run.created_at.isoformat()}",
        f"- base_profile_name: {run.base_profile_name}",
        f"- eval_path: {run.eval_path}",
        f"- query_mode: {run.query_mode}",
        "",
        "## Summary",
        "",
    ]
    for key, value in run.summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Variants", ""])
    for variant in run.variants:
        lines.extend(
            [
                f"### {variant.label}",
                f"- effective_profile_name: {variant.effective_profile_name}",
                f"- retrieval_overrides: {variant.retrieval_overrides}",
                f"- retrieval_json_path: {variant.retrieval_json_path}",
                f"- retrieval_markdown_path: {variant.retrieval_markdown_path}",
                f"- eval_json_path: {variant.eval_json_path}",
                f"- eval_required_claim_recall_mean: {variant.eval_overall_metrics.get('required_claim_recall_mean')}",
                f"- eval_mean_case_seconds: {variant.eval_overall_metrics.get('mean_case_seconds')}",
                f"- retrieval_packed_expected_url_recall_mean: {variant.retrieval_overall_metrics.get('packed_expected_url_recall_mean')}",
                "- retrieval_packed_claim_evidence_recall_mean_annotated: "
                f"{variant.retrieval_overall_metrics.get('packed_claim_evidence_recall_mean_annotated')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_retrieval_budget_sweep_artifacts(settings: Settings, run: RetrievalBudgetSweepRun) -> tuple[Path, Path]:
    output_dir = settings.resolve(Path("datasets/runs/retrieval_budget_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    markdown_path = output_dir / f"{run.run_name}.md"
    json_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")
    markdown_path.write_text(render_retrieval_budget_sweep_markdown(run), encoding="utf-8")
    return json_path, markdown_path
