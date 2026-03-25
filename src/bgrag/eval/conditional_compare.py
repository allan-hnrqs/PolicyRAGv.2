"""Helpers for evaluating conditional answer profiles against a preserved baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bgrag.config import Settings
from bgrag.eval.pairwise import compare_pairwise_runs
from bgrag.eval.run_composition import compose_eval_run, intervention_selected
from bgrag.eval.runner import run_eval
from bgrag.manifests import (
    build_eval_run_manifest,
    build_pairwise_run_manifest,
    build_run_name,
    load_index_manifest,
    write_run_artifact_manifest,
)
from bgrag.pipeline import build_answer_callback
from bgrag.profiles.loader import load_profile
from bgrag.types import EvalRunResult, PairwiseRunResult


@dataclass(frozen=True)
class EvalRunArtifact:
    result: EvalRunResult
    path: Path


@dataclass(frozen=True)
class CompositeRunArtifact:
    result: EvalRunResult
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class PairwiseRunArtifact:
    result: PairwiseRunResult
    path: Path


@dataclass(frozen=True)
class ConditionalCompareArtifacts:
    control: EvalRunArtifact
    candidate: EvalRunArtifact
    composite: CompositeRunArtifact
    summary_json_path: Path
    summary_markdown_path: Path
    pairwise: PairwiseRunArtifact | None = None
    pairwise_error: str | None = None


def resolve_cli_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def write_eval_result_artifact(
    settings: Settings,
    result: EvalRunResult,
    *,
    run_kind: str = "eval",
) -> Path:
    output_path = settings.resolved_runs_dir / f"{result.run_name}.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    write_run_artifact_manifest(
        settings,
        run_name=result.run_name,
        run_kind=run_kind,
        run_artifact_path=output_path,
        run_manifest=result.run_manifest,
    )
    return output_path


def run_profile_eval(
    *,
    settings: Settings,
    profile_name: str,
    eval_path: Path,
    index_namespace: str | None,
) -> EvalRunArtifact:
    runtime_profile = load_profile(profile_name, settings)
    index_manifest = load_index_manifest(settings, index_namespace)
    answer_callback = build_answer_callback(settings, profile_name, index_namespace=str(index_manifest["namespace"]))
    result = run_eval(
        settings,
        runtime_profile,
        eval_path,
        answer_callback,
        run_manifest=build_eval_run_manifest(settings, runtime_profile, eval_path, index_manifest),
    )
    return EvalRunArtifact(result=result, path=write_eval_result_artifact(settings, result))


def render_composite_markdown(
    *,
    control_run: EvalRunResult,
    candidate_run: EvalRunResult,
    composite_run: EvalRunResult,
) -> str:
    composed_from = composite_run.run_manifest.get("composed_from", {})
    selected_case_ids = composed_from.get("selected_case_ids", [])
    non_selected_changed_case_ids = composed_from.get("non_selected_changed_case_ids", [])
    lines = [
        "# Conditional Profile Comparison",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- control_run: {control_run.run_name}",
        f"- candidate_run: {candidate_run.run_name}",
        f"- composite_run: {composite_run.run_name}",
        f"- selected_case_count: {len(selected_case_ids)}",
        f"- selected_case_ids: {', '.join(selected_case_ids) if selected_case_ids else 'none'}",
        f"- non_selected_changed_case_count: {len(non_selected_changed_case_ids)}",
        (
            f"- non_selected_changed_case_ids: "
            f"{', '.join(non_selected_changed_case_ids) if non_selected_changed_case_ids else 'none'}"
        ),
        "",
        "## Required Claim Recall",
        "",
        f"- control: {float(control_run.overall_metrics['required_claim_recall_mean']):.6f}",
        f"- candidate: {float(candidate_run.overall_metrics['required_claim_recall_mean']):.6f}",
        f"- composite: {float(composite_run.overall_metrics['required_claim_recall_mean']):.6f}",
        "",
        "## Abstention",
        "",
        f"- control_abstain_accuracy: {float(control_run.overall_metrics.get('abstain_accuracy', 0.0)):.6f}",
        f"- candidate_abstain_accuracy: {float(candidate_run.overall_metrics.get('abstain_accuracy', 0.0)):.6f}",
        f"- composite_abstain_accuracy: {float(composite_run.overall_metrics.get('abstain_accuracy', 0.0)):.6f}",
        "",
        "## Safety",
        "",
        f"- control_forbidden_claim_violations: {int(control_run.overall_metrics['forbidden_claim_violation_count'])}",
        f"- candidate_forbidden_claim_violations: {int(candidate_run.overall_metrics['forbidden_claim_violation_count'])}",
        f"- composite_forbidden_claim_violations: {int(composite_run.overall_metrics['forbidden_claim_violation_count'])}",
        "",
        "## Artifact Timing",
        "",
        f"- control_mean_case_seconds: {float(control_run.overall_metrics['mean_case_seconds']):.3f}",
        f"- candidate_mean_case_seconds: {float(candidate_run.overall_metrics['mean_case_seconds']):.3f}",
        f"- composite_artifact_mean_case_seconds: {float(composite_run.overall_metrics['mean_case_seconds']):.3f}",
        "- note: composite artifact timing is stitched from parent runs and should not be read as end-to-end deployed latency.",
    ]
    return "\n".join(lines)


def build_conditional_compare_summary(
    *,
    eval_path: Path,
    index_namespace: str,
    control_profile: str,
    candidate_profile: str,
    intervention_paths: set[str],
    control_artifact: EvalRunArtifact,
    candidate_artifact: EvalRunArtifact,
    composite_artifact: CompositeRunArtifact,
    pairwise_artifact: PairwiseRunArtifact | None,
    pairwise_error: str | None = None,
) -> dict[str, object]:
    composed_from = composite_artifact.result.run_manifest.get("composed_from", {})
    selected_case_ids = composed_from.get("selected_case_ids", [])
    non_selected_changed_case_ids = composed_from.get("non_selected_changed_case_ids", [])
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_path": str(eval_path),
        "index_namespace": index_namespace,
        "control_profile": control_profile,
        "candidate_profile": candidate_profile,
        "intervention_paths": sorted(intervention_paths),
        "selected_case_ids": list(selected_case_ids),
        "selected_case_count": len(selected_case_ids),
        "non_selected_changed_case_ids": list(non_selected_changed_case_ids),
        "non_selected_changed_case_count": len(non_selected_changed_case_ids),
        "non_selected_preserved_baseline": bool(composed_from.get("non_selected_preserved_baseline", False)),
        "control_run_path": str(control_artifact.path),
        "candidate_run_path": str(candidate_artifact.path),
        "composite_run_path": str(composite_artifact.json_path),
        "composite_summary_path": str(composite_artifact.markdown_path),
        "control_required_claim_recall": float(control_artifact.result.overall_metrics["required_claim_recall_mean"]),
        "candidate_required_claim_recall": float(
            candidate_artifact.result.overall_metrics["required_claim_recall_mean"]
        ),
        "composite_required_claim_recall": float(
            composite_artifact.result.overall_metrics["required_claim_recall_mean"]
        ),
        "control_forbidden_claim_violations": int(
            control_artifact.result.overall_metrics["forbidden_claim_violation_count"]
        ),
        "candidate_forbidden_claim_violations": int(
            candidate_artifact.result.overall_metrics["forbidden_claim_violation_count"]
        ),
        "composite_forbidden_claim_violations": int(
            composite_artifact.result.overall_metrics["forbidden_claim_violation_count"]
        ),
        "control_answer_failure_count": int(control_artifact.result.overall_metrics.get("answer_failure_count", 0)),
        "candidate_answer_failure_count": int(candidate_artifact.result.overall_metrics.get("answer_failure_count", 0)),
        "composite_answer_failure_count": int(composite_artifact.result.overall_metrics.get("answer_failure_count", 0)),
        "control_abstain_accuracy": float(control_artifact.result.overall_metrics.get("abstain_accuracy", 0.0)),
        "candidate_abstain_accuracy": float(candidate_artifact.result.overall_metrics.get("abstain_accuracy", 0.0)),
        "composite_abstain_accuracy": float(composite_artifact.result.overall_metrics.get("abstain_accuracy", 0.0)),
        "pairwise_run_path": str(pairwise_artifact.path) if pairwise_artifact is not None else None,
        "pairwise_error": pairwise_error,
    }
    if pairwise_artifact is not None:
        summary["pairwise_metrics"] = dict(pairwise_artifact.result.overall_metrics)
    return summary


def render_conditional_compare_summary_markdown(summary: dict[str, object]) -> str:
    pairwise_metrics = summary.get("pairwise_metrics", {})
    lines = [
        "# Conditional Profile Evaluation Summary",
        "",
        f"- generated_at: {summary['created_at']}",
        f"- eval_path: {summary['eval_path']}",
        f"- index_namespace: {summary['index_namespace']}",
        f"- control_profile: {summary['control_profile']}",
        f"- candidate_profile: {summary['candidate_profile']}",
        f"- intervention_paths: {', '.join(summary['intervention_paths'])}",
        f"- selected_case_count: {summary['selected_case_count']}",
        f"- selected_case_ids: {', '.join(summary['selected_case_ids']) if summary['selected_case_ids'] else 'none'}",
        f"- non_selected_changed_case_count: {summary['non_selected_changed_case_count']}",
        (
            f"- non_selected_changed_case_ids: "
            f"{', '.join(summary['non_selected_changed_case_ids']) if summary['non_selected_changed_case_ids'] else 'none'}"
        ),
        f"- non_selected_preserved_baseline: {summary['non_selected_preserved_baseline']}",
        "",
        "## Scalar Metrics",
        "",
        f"- control_required_claim_recall: {float(summary['control_required_claim_recall']):.6f}",
        f"- candidate_required_claim_recall: {float(summary['candidate_required_claim_recall']):.6f}",
        f"- composite_required_claim_recall: {float(summary['composite_required_claim_recall']):.6f}",
        f"- control_forbidden_claim_violations: {int(summary['control_forbidden_claim_violations'])}",
        f"- candidate_forbidden_claim_violations: {int(summary['candidate_forbidden_claim_violations'])}",
        f"- composite_forbidden_claim_violations: {int(summary['composite_forbidden_claim_violations'])}",
        f"- control_answer_failure_count: {int(summary['control_answer_failure_count'])}",
        f"- candidate_answer_failure_count: {int(summary['candidate_answer_failure_count'])}",
        f"- composite_answer_failure_count: {int(summary['composite_answer_failure_count'])}",
        f"- control_abstain_accuracy: {float(summary['control_abstain_accuracy']):.6f}",
        f"- candidate_abstain_accuracy: {float(summary['candidate_abstain_accuracy']):.6f}",
        f"- composite_abstain_accuracy: {float(summary['composite_abstain_accuracy']):.6f}",
        "",
        "## Artifacts",
        "",
        f"- control_run_path: {summary['control_run_path']}",
        f"- candidate_run_path: {summary['candidate_run_path']}",
        f"- composite_run_path: {summary['composite_run_path']}",
        f"- composite_summary_path: {summary['composite_summary_path']}",
        f"- pairwise_run_path: {summary['pairwise_run_path'] or 'not_run'}",
        f"- pairwise_error: {summary['pairwise_error'] or 'none'}",
    ]
    if pairwise_metrics:
        lines.extend(
            [
                "",
                "## Pairwise",
                "",
                f"- control_win_count: {int(pairwise_metrics['control_win_count'])}",
                f"- candidate_win_count: {int(pairwise_metrics['candidate_win_count'])}",
                f"- tie_count: {int(pairwise_metrics['tie_count'])}",
                f"- candidate_win_rate_non_tie: {float(pairwise_metrics['candidate_win_rate_non_tie']):.6f}",
                f"- cache_hit_count: {int(pairwise_metrics['cache_hit_count'])}",
            ]
        )
    return "\n".join(lines)


def write_conditional_compare_summary(
    settings: Settings,
    summary: dict[str, object],
) -> tuple[Path, Path]:
    run_name = build_run_name("conditional_compare_summary")
    json_path = settings.resolved_runs_dir / f"{run_name}.json"
    markdown_path = settings.resolved_runs_dir / f"{run_name}.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(render_conditional_compare_summary_markdown(summary), encoding="utf-8")
    write_run_artifact_manifest(
        settings,
        run_name=run_name,
        run_kind="conditional_compare_summary",
        run_artifact_path=json_path,
        run_manifest=summary,
    )
    return json_path, markdown_path


def compose_conditional_run(
    *,
    settings: Settings,
    control_artifact: EvalRunArtifact,
    candidate_artifact: EvalRunArtifact,
    intervention_paths: set[str],
) -> CompositeRunArtifact:
    composite_run = compose_eval_run(
        control_run=control_artifact.result,
        candidate_run=candidate_artifact.result,
        choose_candidate_case=lambda case: intervention_selected(case, intervention_paths=intervention_paths),
        composite_run_name=build_run_name(f"{candidate_artifact.result.profile_name}_intervention_only"),
        notes=[f"Intervention paths: {', '.join(sorted(intervention_paths))}"],
    )
    json_path = write_eval_result_artifact(settings, composite_run, run_kind="composite_eval")
    markdown_path = settings.resolved_runs_dir / f"{composite_run.run_name}.md"
    markdown_path.write_text(
        render_composite_markdown(
            control_run=control_artifact.result,
            candidate_run=candidate_artifact.result,
            composite_run=composite_run,
        ),
        encoding="utf-8",
    )
    return CompositeRunArtifact(result=composite_run, json_path=json_path, markdown_path=markdown_path)


def write_pairwise_run_artifact(settings: Settings, result: PairwiseRunResult) -> Path:
    output_path = settings.resolved_runs_dir / f"{result.run_name}.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    write_run_artifact_manifest(
        settings,
        run_name=result.run_name,
        run_kind="pairwise",
        run_artifact_path=output_path,
        run_manifest=result.run_manifest,
    )
    return output_path


def run_conditional_compare(
    *,
    settings: Settings,
    eval_path: Path,
    control_profile: str,
    candidate_profile: str,
    index_namespace: str | None,
    intervention_paths: set[str],
    include_pairwise: bool,
) -> ConditionalCompareArtifacts:
    resolved_index_manifest = load_index_manifest(settings, index_namespace)
    resolved_index_namespace = str(resolved_index_manifest["namespace"])
    control_artifact = run_profile_eval(
        settings=settings,
        profile_name=control_profile,
        eval_path=eval_path,
        index_namespace=resolved_index_namespace,
    )
    candidate_artifact = run_profile_eval(
        settings=settings,
        profile_name=candidate_profile,
        eval_path=eval_path,
        index_namespace=resolved_index_namespace,
    )
    composite_artifact = compose_conditional_run(
        settings=settings,
        control_artifact=control_artifact,
        candidate_artifact=candidate_artifact,
        intervention_paths=intervention_paths,
    )

    pairwise_artifact: PairwiseRunArtifact | None = None
    pairwise_error: str | None = None
    if include_pairwise:
        try:
            pairwise_result = compare_pairwise_runs(
                settings,
                control_artifact.path,
                composite_artifact.json_path,
                run_manifest=build_pairwise_run_manifest(settings, control_artifact.path, composite_artifact.json_path),
            )
            pairwise_artifact = PairwiseRunArtifact(
                result=pairwise_result,
                path=write_pairwise_run_artifact(settings, pairwise_result),
            )
        except Exception as exc:  # pragma: no cover - exercised by live API/auth/runtime only
            pairwise_error = str(exc)

    summary = build_conditional_compare_summary(
        eval_path=eval_path,
        index_namespace=resolved_index_namespace,
        control_profile=control_profile,
        candidate_profile=candidate_profile,
        intervention_paths=intervention_paths,
        control_artifact=control_artifact,
        candidate_artifact=candidate_artifact,
        composite_artifact=composite_artifact,
        pairwise_artifact=pairwise_artifact,
        pairwise_error=pairwise_error,
    )
    summary_json_path, summary_markdown_path = write_conditional_compare_summary(settings, summary)
    return ConditionalCompareArtifacts(
        control=control_artifact,
        candidate=candidate_artifact,
        composite=composite_artifact,
        pairwise=pairwise_artifact,
        pairwise_error=pairwise_error,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
    )
