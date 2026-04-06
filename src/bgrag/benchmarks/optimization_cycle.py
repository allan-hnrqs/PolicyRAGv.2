"""Persistent optimization-loop scaffolding and cycle runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from bgrag.benchmarks.product import (
    ProductBenchmarkRunResult,
    run_product_benchmark,
    write_product_benchmark_artifacts,
)
from bgrag.benchmarks.retrieval import (
    RetrievalBenchmarkRun,
    run_retrieval_benchmark,
    write_retrieval_benchmark_artifacts,
)
from bgrag.config import Settings
from bgrag.demo_server import build_demo_settings
from bgrag.eval.loader import load_eval_cases
from bgrag.eval.runner import run_eval
from bgrag.manifests import (
    build_eval_run_manifest,
    build_run_name,
    derive_index_namespace,
    load_index_manifest,
    repo_relative_path,
    write_run_artifact_manifest,
)
from bgrag.pipeline import build_answer_callback
from bgrag.profiles.loader import load_profile
from bgrag.types import EvalRunResult

CycleKind = Literal["large", "small"]
CycleClassification = Literal["pending", "promising", "mixed", "rejected"]


class EvalSurfaceManifest(BaseModel):
    name: str
    purpose: str
    source_eval_paths: dict[str, str] = Field(default_factory=dict)
    target_eval_paths: dict[str, str] = Field(default_factory=dict)
    source_artifacts: list[str] = Field(default_factory=list)
    dev_case_ids: list[str] = Field(default_factory=list)
    holdout_case_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationCycleArtifactRef(BaseModel):
    label: str
    artifact_path: str
    details_path: str | None = None
    summary: dict[str, object] = Field(default_factory=dict)


class ExternalComparatorRef(BaseModel):
    label: str
    artifact_path: str
    notes: str | None = None


class OptimizationCycleResult(BaseModel):
    run_name: str
    created_at: datetime
    cycle_id: str
    cycle_kind: CycleKind
    hypothesis_id: str
    profile_name: str
    control_profile_name: str
    classification: CycleClassification = "pending"
    failure_surface_manifest_path: str | None = None
    materialized_surfaces: dict[str, str] = Field(default_factory=dict)
    benchmark_artifacts: list[OptimizationCycleArtifactRef] = Field(default_factory=list)
    external_comparators: list[ExternalComparatorRef] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def load_eval_surface_manifest(path: Path) -> EvalSurfaceManifest:
    return EvalSurfaceManifest.model_validate_json(path.read_text(encoding="utf-8"))


def materialize_eval_surface_from_manifest(settings: Settings, manifest_path: Path) -> dict[str, Path]:
    manifest = load_eval_surface_manifest(manifest_path)
    written: dict[str, Path] = {}
    for split, case_ids in (("dev", manifest.dev_case_ids), ("holdout", manifest.holdout_case_ids)):
        source = manifest.source_eval_paths.get(split)
        target = manifest.target_eval_paths.get(split)
        if not source or not target:
            continue
        source_path = settings.resolve(Path(source))
        target_path = settings.resolve(Path(target))
        cases = load_eval_cases(source_path)
        by_id = {case.id: case for case in cases}
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"Surface manifest {manifest.name} is missing {split} case ids: {missing}")
        selected = [by_id[case_id] for case_id in case_ids]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            "\n".join(json.dumps(case.model_dump(mode="json", exclude_none=True)) for case in selected) + "\n",
            encoding="utf-8",
        )
        written[split] = target_path
    return written


def _write_eval_run_artifacts(settings: Settings, run: EvalRunResult) -> tuple[Path, Path]:
    json_path = settings.resolved_runs_dir / f"{run.run_name}.json"
    json_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    manifest_path = write_run_artifact_manifest(
        settings,
        run_name=run.run_name,
        run_kind="eval",
        run_artifact_path=json_path,
        run_manifest=run.run_manifest,
    )
    return json_path, manifest_path


def _summarize_eval_run(run: EvalRunResult) -> dict[str, object]:
    keys = (
        "required_claim_recall_mean",
        "forbidden_claim_violation_count",
        "answer_failure_count",
        "mean_case_seconds",
        "packed_expected_url_recall_mean",
        "packed_claim_evidence_recall_mean",
        "abstain_accuracy",
    )
    return {key: run.overall_metrics.get(key) for key in keys if key in run.overall_metrics}


def _summarize_retrieval_run(run: RetrievalBenchmarkRun) -> dict[str, object]:
    keys = (
        "packed_expected_url_recall_mean",
        "packed_claim_evidence_recall_mean_annotated",
        "packed_primary_url_hit_rate",
        "packed_mrr",
        "packed_miss_count",
        "mean_total_case_seconds",
    )
    return {key: run.overall_metrics.get(key) for key in keys if key in run.overall_metrics}


def _summarize_product_run(run: ProductBenchmarkRunResult) -> dict[str, object]:
    stage_timings = run.summary.get("stage_timings", {}) if isinstance(run.summary, dict) else {}
    total_request = stage_timings.get("total_request_seconds", {}) if isinstance(stage_timings, dict) else {}
    elapsed_wall = stage_timings.get("elapsed_wall_seconds", {}) if isinstance(stage_timings, dict) else {}
    return {
        "case_count": run.summary.get("case_count"),
        "error_case_count": run.summary.get("error_case_count"),
        "total_request_mean_seconds": total_request.get("mean") if isinstance(total_request, dict) else None,
        "elapsed_wall_mean_seconds": elapsed_wall.get("mean") if isinstance(elapsed_wall, dict) else None,
    }


def _run_eval_surface(settings: Settings, profile_name: str, eval_path: Path) -> tuple[EvalRunResult, Path, Path]:
    profile = load_profile(profile_name, settings)
    index_namespace = derive_index_namespace(settings, profile_name)
    index_manifest = load_index_manifest(settings, index_namespace)
    answer_callback = build_answer_callback(settings, profile_name, index_namespace=str(index_manifest["namespace"]))
    run = run_eval(
        settings,
        profile,
        eval_path,
        answer_callback,
        run_manifest=build_eval_run_manifest(settings, profile, eval_path, index_manifest),
    )
    json_path, manifest_path = _write_eval_run_artifacts(settings, run)
    return run, json_path, manifest_path


def _append_artifact(
    refs: list[OptimizationCycleArtifactRef],
    *,
    settings: Settings,
    label: str,
    artifact_path: Path,
    details_path: Path | None = None,
    summary: dict[str, object] | None = None,
) -> None:
    refs.append(
        OptimizationCycleArtifactRef(
            label=label,
            artifact_path=repo_relative_path(settings, artifact_path),
            details_path=repo_relative_path(settings, details_path) if details_path is not None else None,
            summary=summary or {},
        )
    )


def render_optimization_cycle_markdown(run: OptimizationCycleResult) -> str:
    lines = [
        f"# Optimization Cycle: {run.cycle_id}",
        "",
        f"- run_name: {run.run_name}",
        f"- created_at: {run.created_at.isoformat()}",
        f"- cycle_kind: {run.cycle_kind}",
        f"- hypothesis_id: {run.hypothesis_id}",
        f"- profile_name: {run.profile_name}",
        f"- control_profile_name: {run.control_profile_name}",
        f"- classification: {run.classification}",
    ]
    if run.failure_surface_manifest_path:
        lines.append(f"- failure_surface_manifest_path: {run.failure_surface_manifest_path}")
    if run.notes:
        lines.append(f"- notes: {' | '.join(run.notes)}")
    if run.materialized_surfaces:
        lines.extend(["", "## Materialized Surfaces", ""])
        for split, path in run.materialized_surfaces.items():
            lines.append(f"- {split}: {path}")
    if run.benchmark_artifacts:
        lines.extend(["", "## Benchmark Artifacts", ""])
        for ref in run.benchmark_artifacts:
            lines.append(f"### {ref.label}")
            lines.append(f"- artifact_path: {ref.artifact_path}")
            if ref.details_path:
                lines.append(f"- details_path: {ref.details_path}")
            for key, value in ref.summary.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
    if run.external_comparators:
        lines.extend(["", "## External Comparator References", ""])
        for ref in run.external_comparators:
            lines.append(f"- {ref.label}: {ref.artifact_path}")
            if ref.notes:
                lines.append(f"  notes: {ref.notes}")
    return "\n".join(lines).rstrip() + "\n"


def write_optimization_cycle_artifacts(settings: Settings, run: OptimizationCycleResult) -> tuple[Path, Path]:
    output_dir = settings.resolved_runs_dir / "optimization_cycles"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    md_path = output_dir / f"{run.run_name}.md"
    json_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")
    md_path.write_text(render_optimization_cycle_markdown(run), encoding="utf-8")
    return json_path, md_path


def run_optimization_cycle(
    settings: Settings,
    *,
    cycle_id: str,
    hypothesis_id: str,
    cycle_kind: CycleKind,
    profile_name: str,
    control_profile_name: str = "baseline_vector_rerank_shortlist",
    classification: CycleClassification = "pending",
    dev_eval_path: Path = Path("datasets/eval/dev/parity19_dev.jsonl"),
    holdout_eval_path: Path = Path("datasets/eval/holdout/parity19_holdout.jsonl"),
    failure_surface_manifest_path: Path | None = None,
    product_manifest_path: Path = Path("datasets/eval/manifests/product_serving_benchmark_v1.json"),
    multiturn_manifest_path: Path = Path("datasets/eval/manifests/multiturn_benchmark_v1.json"),
    run_canonical: bool | None = None,
    run_failure_surfaces: bool = True,
    run_product: bool | None = None,
    run_multiturn: bool = False,
    external_comparators: list[ExternalComparatorRef] | None = None,
    notes: list[str] | None = None,
) -> OptimizationCycleResult:
    if run_canonical is None:
        run_canonical = cycle_kind == "large"
    if run_product is None:
        run_product = cycle_kind == "large"

    artifact_refs: list[OptimizationCycleArtifactRef] = []
    materialized_surfaces: dict[str, str] = {}
    if failure_surface_manifest_path is not None:
        written = materialize_eval_surface_from_manifest(settings, failure_surface_manifest_path)
        materialized_surfaces = {split: repo_relative_path(settings, path) for split, path in written.items()}

    if run_canonical:
        for split, eval_path in (("dev", dev_eval_path), ("holdout", holdout_eval_path)):
            retrieval_run = run_retrieval_benchmark(
                settings,
                eval_path=settings.resolve(eval_path),
                profile_name=profile_name,
                query_mode="profile",
            )
            retrieval_json, retrieval_md = write_retrieval_benchmark_artifacts(settings, retrieval_run)
            _append_artifact(
                artifact_refs,
                settings=settings,
                label=f"retrieval_{split}",
                artifact_path=retrieval_json,
                details_path=retrieval_md,
                summary=_summarize_retrieval_run(retrieval_run),
            )

            eval_run, eval_json, eval_manifest = _run_eval_surface(settings, profile_name, settings.resolve(eval_path))
            _append_artifact(
                artifact_refs,
                settings=settings,
                label=f"eval_{split}",
                artifact_path=eval_json,
                details_path=eval_manifest,
                summary=_summarize_eval_run(eval_run),
            )

    if run_failure_surfaces and failure_surface_manifest_path is not None:
        surface_manifest = load_eval_surface_manifest(settings.resolve(failure_surface_manifest_path))
        for split, rel_path in surface_manifest.target_eval_paths.items():
            if split not in materialized_surfaces:
                continue
            resolved_path = settings.resolve(Path(rel_path))
            retrieval_run = run_retrieval_benchmark(
                settings,
                eval_path=resolved_path,
                profile_name=profile_name,
                query_mode="profile",
            )
            retrieval_json, retrieval_md = write_retrieval_benchmark_artifacts(settings, retrieval_run)
            _append_artifact(
                artifact_refs,
                settings=settings,
                label=f"failure_surface_retrieval_{split}",
                artifact_path=retrieval_json,
                details_path=retrieval_md,
                summary=_summarize_retrieval_run(retrieval_run),
            )

            eval_run, eval_json, eval_manifest = _run_eval_surface(settings, profile_name, resolved_path)
            _append_artifact(
                artifact_refs,
                settings=settings,
                label=f"failure_surface_eval_{split}",
                artifact_path=eval_json,
                details_path=eval_manifest,
                summary=_summarize_eval_run(eval_run),
            )

    if run_product or run_multiturn:
        demo_settings = build_demo_settings(settings.project_root)
        if run_product:
            product_run = run_product_benchmark(
                demo_settings,
                manifest_path=demo_settings.resolve(product_manifest_path),
                profile_name=profile_name,
            )
            product_json, product_md = write_product_benchmark_artifacts(demo_settings, product_run)
            _append_artifact(
                artifact_refs,
                settings=demo_settings,
                label="product_benchmark",
                artifact_path=product_json,
                details_path=product_md,
                summary=_summarize_product_run(product_run),
            )
        if run_multiturn:
            multiturn_run = run_product_benchmark(
                demo_settings,
                manifest_path=demo_settings.resolve(multiturn_manifest_path),
                profile_name=profile_name,
            )
            multiturn_json, multiturn_md = write_product_benchmark_artifacts(demo_settings, multiturn_run)
            _append_artifact(
                artifact_refs,
                settings=demo_settings,
                label="multiturn_benchmark",
                artifact_path=multiturn_json,
                details_path=multiturn_md,
                summary=_summarize_product_run(multiturn_run),
            )

    run = OptimizationCycleResult(
        run_name=build_run_name(f"optimization_cycle_{cycle_id}"),
        created_at=datetime.now(timezone.utc),
        cycle_id=cycle_id,
        cycle_kind=cycle_kind,
        hypothesis_id=hypothesis_id,
        profile_name=profile_name,
        control_profile_name=control_profile_name,
        classification=classification,
        failure_surface_manifest_path=(
            repo_relative_path(settings, settings.resolve(failure_surface_manifest_path))
            if failure_surface_manifest_path is not None
            else None
        ),
        materialized_surfaces=materialized_surfaces,
        benchmark_artifacts=artifact_refs,
        external_comparators=external_comparators or [],
        notes=notes or [],
    )
    return run
