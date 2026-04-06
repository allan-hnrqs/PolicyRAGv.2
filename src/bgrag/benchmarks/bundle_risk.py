"""Audit deterministic bundle-risk signals against eval artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from bgrag.config import Settings
from bgrag.manifests import build_run_name
from bgrag.serving.bundle_risk import BundleRiskAssessment, assess_bundle_risk
from bgrag.types import EvalRunResult


class BundleRiskAuditCaseResult(BaseModel):
    case_id: str
    split: str | None = None
    required_claim_recall: float = 0.0
    low_recall_case: bool = False
    risk: BundleRiskAssessment


class BundleRiskAuditRun(BaseModel):
    run_name: str
    created_at: datetime
    source_run_path: str
    source_profile_name: str
    low_recall_threshold: float
    case_results: list[BundleRiskAuditCaseResult]
    summary: dict[str, object] = Field(default_factory=dict)


def _summarize(run: BundleRiskAuditRun) -> dict[str, object]:
    cases = run.case_results
    low_recall_cases = [case for case in cases if case.low_recall_case]
    flagged_cases = [case for case in cases if case.risk.retry_signal]
    flagged_low_recall = [case for case in low_recall_cases if case.risk.retry_signal]
    return {
        "case_count": len(cases),
        "low_recall_case_count": len(low_recall_cases),
        "flagged_case_count": len(flagged_cases),
        "flagged_low_recall_case_count": len(flagged_low_recall),
        "flag_precision_on_low_recall": (
            len(flagged_low_recall) / len(flagged_cases) if flagged_cases else None
        ),
        "flag_recall_on_low_recall": (
            len(flagged_low_recall) / len(low_recall_cases) if low_recall_cases else None
        ),
        "mean_required_claim_recall": mean(case.required_claim_recall for case in cases) if cases else 0.0,
        "mean_required_claim_recall_flagged": (
            mean(case.required_claim_recall for case in flagged_cases) if flagged_cases else None
        ),
        "high_risk_case_ids": [case.case_id for case in cases if case.risk.risk_level == "high"],
        "flagged_case_ids": [case.case_id for case in flagged_cases],
        "flagged_low_recall_case_ids": [case.case_id for case in flagged_low_recall],
    }


def run_bundle_risk_audit(
    settings: Settings,
    *,
    source_run_path: Path,
    low_recall_threshold: float = 0.75,
) -> BundleRiskAuditRun:
    resolved_source = settings.resolve(source_run_path)
    source_run = EvalRunResult.model_validate_json(resolved_source.read_text(encoding="utf-8"))
    case_results: list[BundleRiskAuditCaseResult] = []
    for case in source_run.cases:
        evidence = case.answer.evidence_bundle
        if evidence is None:
            continue
        recall = float(case.metrics.get("required_claim_recall", 0.0) or 0.0)
        case_results.append(
            BundleRiskAuditCaseResult(
                case_id=case.case.id,
                split=case.case.split,
                required_claim_recall=recall,
                low_recall_case=recall < low_recall_threshold,
                risk=assess_bundle_risk(evidence),
            )
        )
    run = BundleRiskAuditRun(
        run_name=build_run_name("bundle_risk_audit"),
        created_at=datetime.now(timezone.utc),
        source_run_path=str(resolved_source),
        source_profile_name=source_run.profile_name,
        low_recall_threshold=low_recall_threshold,
        case_results=case_results,
    )
    run.summary = _summarize(run)
    return run


def render_bundle_risk_audit_markdown(run: BundleRiskAuditRun) -> str:
    lines = [
        "# Bundle Risk Audit",
        "",
        f"- run_name: {run.run_name}",
        f"- source_run_path: {run.source_run_path}",
        f"- source_profile_name: {run.source_profile_name}",
        f"- low_recall_threshold: {run.low_recall_threshold}",
        f"- case_count: {run.summary.get('case_count')}",
        f"- low_recall_case_count: {run.summary.get('low_recall_case_count')}",
        f"- flagged_case_count: {run.summary.get('flagged_case_count')}",
        f"- flagged_low_recall_case_count: {run.summary.get('flagged_low_recall_case_count')}",
        f"- flag_precision_on_low_recall: {run.summary.get('flag_precision_on_low_recall')}",
        f"- flag_recall_on_low_recall: {run.summary.get('flag_recall_on_low_recall')}",
        "",
        "## Cases",
        "",
    ]
    for case in run.case_results:
        lines.append(f"### {case.case_id}")
        lines.append(f"- split: {case.split or '<none>'}")
        lines.append(f"- required_claim_recall: {case.required_claim_recall:.4f}")
        lines.append(f"- low_recall_case: {case.low_recall_case}")
        lines.append(f"- risk_level: {case.risk.risk_level}")
        lines.append(f"- retry_signal: {case.risk.retry_signal}")
        lines.append(f"- top_document_share: {case.risk.top_document_share:.4f}")
        lines.append(f"- top_heading_share: {case.risk.top_heading_share:.4f}")
        lines.append(f"- lost_document_breadth: {case.risk.lost_document_breadth}")
        lines.append(f"- lost_heading_breadth: {case.risk.lost_heading_breadth}")
        lines.append(f"- lost_source_family_breadth: {case.risk.lost_source_family_breadth}")
        lines.append(f"- reasons: {', '.join(case.risk.reasons) or '<none>'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_bundle_risk_audit_artifacts(settings: Settings, run: BundleRiskAuditRun) -> tuple[Path, Path]:
    output_dir = settings.resolved_runs_dir / "bundle_risk_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    md_path = output_dir / f"{run.run_name}.md"
    json_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")
    md_path.write_text(render_bundle_risk_audit_markdown(run), encoding="utf-8")
    return json_path, md_path
