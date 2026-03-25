from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import REPO_ROOT
from bgrag.eval.run_composition import compose_eval_run, intervention_selected
from bgrag.types import EvalRunResult


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "composite"


def _load_run(path: Path) -> EvalRunResult:
    return EvalRunResult.model_validate_json(path.read_text(encoding="utf-8"))


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _render_summary(*, control_run: EvalRunResult, candidate_run: EvalRunResult, composite_run: EvalRunResult) -> str:
    selected_case_ids = composite_run.run_manifest.get("composed_from", {}).get("selected_case_ids", [])
    lines = [
        "# Intervention-Only Composite Run",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- control_run: {control_run.run_name}",
        f"- candidate_run: {candidate_run.run_name}",
        f"- composite_run: {composite_run.run_name}",
        "",
        "## Selection",
        "",
        f"- selected_case_count: {len(selected_case_ids)}",
        f"- selected_case_ids: {', '.join(selected_case_ids) if selected_case_ids else 'none'}",
        "",
        "## Required Claim Recall",
        "",
        f"- control: {float(control_run.overall_metrics['required_claim_recall_mean']):.6f}",
        f"- candidate: {float(candidate_run.overall_metrics['required_claim_recall_mean']):.6f}",
        f"- composite: {float(composite_run.overall_metrics['required_claim_recall_mean']):.6f}",
        "",
        "## Safety and Failures",
        "",
        f"- control_forbidden_claim_violations: {int(control_run.overall_metrics['forbidden_claim_violation_count'])}",
        f"- candidate_forbidden_claim_violations: {int(candidate_run.overall_metrics['forbidden_claim_violation_count'])}",
        f"- composite_forbidden_claim_violations: {int(composite_run.overall_metrics['forbidden_claim_violation_count'])}",
        f"- control_answer_failures: {int(control_run.overall_metrics['answer_failure_count'])}",
        f"- candidate_answer_failures: {int(candidate_run.overall_metrics['answer_failure_count'])}",
        f"- composite_answer_failures: {int(composite_run.overall_metrics['answer_failure_count'])}",
        "",
        "## Latency",
        "",
        f"- control_mean_case_seconds: {float(control_run.overall_metrics['mean_case_seconds']):.3f}",
        f"- candidate_mean_case_seconds: {float(candidate_run.overall_metrics['mean_case_seconds']):.3f}",
        f"- composite_mean_case_seconds: {float(composite_run.overall_metrics['mean_case_seconds']):.3f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an intervention-only composite eval run.")
    parser.add_argument("control_run")
    parser.add_argument("candidate_run")
    parser.add_argument(
        "--intervention-path",
        action="append",
        default=None,
        help="Candidate selected_path value that counts as a real intervention. Repeatable.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    control_path = _resolve_path(repo_root, args.control_run)
    candidate_path = _resolve_path(repo_root, args.candidate_run)
    control_run = _load_run(control_path)
    candidate_run = _load_run(candidate_path)

    allowed_paths = set(args.intervention_path or ["rewrite_structured_contract"])
    composite_run = compose_eval_run(
        control_run=control_run,
        candidate_run=candidate_run,
        choose_candidate_case=lambda case: intervention_selected(case, intervention_paths=allowed_paths),
        composite_run_name=f"{candidate_run.profile_name}_intervention_only_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        notes=[f"Intervention paths: {', '.join(sorted(allowed_paths))}"],
    )

    runs_dir = repo_root / "datasets" / "runs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    slug = _slugify(f"{control_run.run_name}_vs_{candidate_run.run_name}_intervention_only")
    json_path = runs_dir / f"{slug}_{timestamp}.json"
    md_path = runs_dir / f"{slug}_{timestamp}.md"
    json_path.write_text(composite_run.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(
        _render_summary(control_run=control_run, candidate_run=candidate_run, composite_run=composite_run),
        encoding="utf-8",
    )
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
