from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from bgrag.config import Settings
from bgrag.eval.runner import run_eval
from bgrag.manifests import build_eval_run_manifest, build_run_name, load_index_manifest, write_run_artifact_manifest
from bgrag.pipeline import build_answer_callback
from bgrag.profiles.loader import load_profile


@dataclass
class RunSummary:
    run_file: str
    required_claim_recall: float
    mean_case_seconds: float
    answer_failures: int
    forbidden_claim_violations: int
    packed_primary_url_hit_rate: float
    candidate_primary_url_hit_rate: float
    packed_expected_url_recall: float
    candidate_expected_url_recall: float
    packed_claim_evidence_recall: float
    candidate_claim_evidence_recall: float
    packed_claim_evidence_recall_annotated: float
    candidate_claim_evidence_recall_annotated: float
    annotated_case_count: int
    abstain_count: int
    route_counts: dict[str, int]


@dataclass
class RepeatSummary:
    profile: str
    eval_path: str
    repeats: int
    run_files: list[str]
    mean_required_claim_recall: float
    min_required_claim_recall: float
    max_required_claim_recall: float
    mean_case_seconds: float
    answer_failures: int
    forbidden_claim_violations: int
    mean_packed_primary_url_hit_rate: float
    mean_candidate_primary_url_hit_rate: float
    mean_packed_expected_url_recall: float
    mean_candidate_expected_url_recall: float
    mean_packed_claim_evidence_recall: float
    mean_candidate_claim_evidence_recall: float
    mean_packed_claim_evidence_recall_annotated: float
    mean_candidate_claim_evidence_recall_annotated: float
    annotated_case_count: float
    run_summaries: list[RunSummary]


def _run_once(
    *,
    settings: Settings,
    profile_name: str,
    eval_path: Path,
    index_namespace: str,
) -> RunSummary:
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
    output_path = settings.resolved_runs_dir / f"{result.run_name}.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    write_run_artifact_manifest(
        settings,
        run_name=result.run_name,
        run_kind="eval",
        run_artifact_path=output_path,
        run_manifest=result.run_manifest,
    )
    route_counts: dict[str, int] = {}
    for case in result.cases:
        raw_response = case.answer.raw_response or {}
        route = raw_response.get("selected_path")
        if not route:
            answer_plan = raw_response.get("answer_plan", {})
            route = answer_plan.get("answer_mode")
        if not route:
            route = case.answer.strategy_name
        route_key = str(route)
        route_counts[route_key] = route_counts.get(route_key, 0) + 1
    overall_metrics = result.overall_metrics
    return RunSummary(
        run_file=str(output_path),
        required_claim_recall=float(overall_metrics["required_claim_recall_mean"]),
        mean_case_seconds=float(overall_metrics["mean_case_seconds"]),
        answer_failures=int(overall_metrics["answer_failure_count"]),
        forbidden_claim_violations=int(overall_metrics["forbidden_claim_violation_count"]),
        packed_primary_url_hit_rate=float(overall_metrics["packed_primary_url_hit_rate"]),
        candidate_primary_url_hit_rate=float(overall_metrics["candidate_primary_url_hit_rate"]),
        packed_expected_url_recall=float(overall_metrics["packed_expected_url_recall_mean"]),
        candidate_expected_url_recall=float(overall_metrics["candidate_expected_url_recall_mean"]),
        packed_claim_evidence_recall=float(overall_metrics["packed_claim_evidence_recall_mean"]),
        candidate_claim_evidence_recall=float(overall_metrics["candidate_claim_evidence_recall_mean"]),
        packed_claim_evidence_recall_annotated=float(overall_metrics["packed_claim_evidence_recall_mean_annotated"]),
        candidate_claim_evidence_recall_annotated=float(
            overall_metrics["candidate_claim_evidence_recall_mean_annotated"]
        ),
        annotated_case_count=int(overall_metrics["claim_evidence_annotated_case_count"]),
        abstain_count=sum(1 for case in result.cases if case.answer.abstained),
        route_counts=route_counts,
    )


def _render_markdown_summary(eval_path: Path, summaries: list[RepeatSummary]) -> str:
    lines = [
        "# Profile Comparison Summary",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- eval_path: {eval_path}",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary.profile}",
                "",
                f"- repeats: {summary.repeats}",
                f"- mean_required_claim_recall: {summary.mean_required_claim_recall:.6f}",
                f"- min_required_claim_recall: {summary.min_required_claim_recall:.6f}",
                f"- max_required_claim_recall: {summary.max_required_claim_recall:.6f}",
                f"- mean_case_seconds: {summary.mean_case_seconds:.3f}",
                f"- answer_failures: {summary.answer_failures}",
                f"- forbidden_claim_violations: {summary.forbidden_claim_violations}",
                f"- mean_packed_primary_url_hit_rate: {summary.mean_packed_primary_url_hit_rate:.6f}",
                f"- mean_candidate_primary_url_hit_rate: {summary.mean_candidate_primary_url_hit_rate:.6f}",
                f"- mean_packed_expected_url_recall: {summary.mean_packed_expected_url_recall:.6f}",
                f"- mean_candidate_expected_url_recall: {summary.mean_candidate_expected_url_recall:.6f}",
                f"- mean_packed_claim_evidence_recall: {summary.mean_packed_claim_evidence_recall:.6f}",
                f"- mean_candidate_claim_evidence_recall: {summary.mean_candidate_claim_evidence_recall:.6f}",
                f"- mean_packed_claim_evidence_recall_annotated: {summary.mean_packed_claim_evidence_recall_annotated:.6f}",
                (
                    "- mean_candidate_claim_evidence_recall_annotated: "
                    f"{summary.mean_candidate_claim_evidence_recall_annotated:.6f}"
                ),
                f"- annotated_case_count: {summary.annotated_case_count:.1f}",
                "",
                "### Runs",
                "",
            ]
        )
        for index, run_summary in enumerate(summary.run_summaries, start=1):
            route_bits = ", ".join(
                f"{route}={count}" for route, count in sorted(run_summary.route_counts.items())
            )
            lines.extend(
                [
                    f"{index}. `{run_summary.run_file}`",
                    f"   - required_claim_recall: {run_summary.required_claim_recall:.6f}",
                    f"   - mean_case_seconds: {run_summary.mean_case_seconds:.3f}",
                    f"   - answer_failures: {run_summary.answer_failures}",
                    f"   - forbidden_claim_violations: {run_summary.forbidden_claim_violations}",
                    f"   - abstain_count: {run_summary.abstain_count}",
                    f"   - routes: {route_bits}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated profile comparisons on the same eval set.")
    parser.add_argument("eval_path", help="Path to eval jsonl relative to repo root or absolute")
    parser.add_argument("--profiles", nargs="+", required=True, help="Profile names to compare")
    parser.add_argument("--repeats", type=int, default=2, help="Number of repeats per profile")
    parser.add_argument("--index-namespace", default=None, help="Explicit index namespace")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings = Settings(_env_file=repo_root / ".env", project_root=repo_root)
    settings.ensure_directories()
    eval_path = Path(args.eval_path)
    if not eval_path.is_absolute():
        eval_path = settings.resolve(eval_path)

    summaries: list[RepeatSummary] = []
    for profile_name in args.profiles:
        run_summaries: list[RunSummary] = []
        recalls: list[float] = []
        mean_case_seconds: list[float] = []
        failure_counts: list[int] = []
        forbidden_counts: list[int] = []
        packed_primary_hit_rates: list[float] = []
        candidate_primary_hit_rates: list[float] = []
        packed_expected_url_recalls: list[float] = []
        candidate_expected_url_recalls: list[float] = []
        packed_claim_recalls: list[float] = []
        candidate_claim_recalls: list[float] = []
        packed_claim_recalls_annotated: list[float] = []
        candidate_claim_recalls_annotated: list[float] = []
        annotated_case_counts: list[float] = []
        for _ in range(args.repeats):
            run_summary = _run_once(
                settings=settings,
                profile_name=profile_name,
                eval_path=eval_path,
                index_namespace=args.index_namespace,
            )
            run_summaries.append(run_summary)
            recalls.append(run_summary.required_claim_recall)
            mean_case_seconds.append(run_summary.mean_case_seconds)
            failure_counts.append(run_summary.answer_failures)
            forbidden_counts.append(run_summary.forbidden_claim_violations)
            packed_primary_hit_rates.append(run_summary.packed_primary_url_hit_rate)
            candidate_primary_hit_rates.append(run_summary.candidate_primary_url_hit_rate)
            packed_expected_url_recalls.append(run_summary.packed_expected_url_recall)
            candidate_expected_url_recalls.append(run_summary.candidate_expected_url_recall)
            packed_claim_recalls.append(run_summary.packed_claim_evidence_recall)
            candidate_claim_recalls.append(run_summary.candidate_claim_evidence_recall)
            packed_claim_recalls_annotated.append(run_summary.packed_claim_evidence_recall_annotated)
            candidate_claim_recalls_annotated.append(run_summary.candidate_claim_evidence_recall_annotated)
            annotated_case_counts.append(float(run_summary.annotated_case_count))
        summaries.append(
            RepeatSummary(
                profile=profile_name,
                eval_path=str(eval_path),
                repeats=args.repeats,
                run_files=[run_summary.run_file for run_summary in run_summaries],
                mean_required_claim_recall=statistics.mean(recalls),
                min_required_claim_recall=min(recalls),
                max_required_claim_recall=max(recalls),
                mean_case_seconds=statistics.mean(mean_case_seconds),
                answer_failures=sum(failure_counts),
                forbidden_claim_violations=sum(forbidden_counts),
                mean_packed_primary_url_hit_rate=statistics.mean(packed_primary_hit_rates),
                mean_candidate_primary_url_hit_rate=statistics.mean(candidate_primary_hit_rates),
                mean_packed_expected_url_recall=statistics.mean(packed_expected_url_recalls),
                mean_candidate_expected_url_recall=statistics.mean(candidate_expected_url_recalls),
                mean_packed_claim_evidence_recall=statistics.mean(packed_claim_recalls),
                mean_candidate_claim_evidence_recall=statistics.mean(candidate_claim_recalls),
                mean_packed_claim_evidence_recall_annotated=statistics.mean(packed_claim_recalls_annotated),
                mean_candidate_claim_evidence_recall_annotated=statistics.mean(candidate_claim_recalls_annotated),
                annotated_case_count=statistics.mean(annotated_case_counts),
                run_summaries=run_summaries,
            )
        )

    run_name = build_run_name("profile_compare")
    json_output_path = settings.resolved_runs_dir / f"{run_name}.json"
    json_output_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "eval_path": str(eval_path),
                "profiles": [asdict(summary) for summary in summaries],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_output_path = settings.resolved_runs_dir / f"{run_name}.md"
    markdown_output_path.write_text(_render_markdown_summary(eval_path, summaries), encoding="utf-8")
    print(json_output_path)
    print(markdown_output_path)
    print(json.dumps([asdict(summary) for summary in summaries], indent=2))


if __name__ == "__main__":
    main()
