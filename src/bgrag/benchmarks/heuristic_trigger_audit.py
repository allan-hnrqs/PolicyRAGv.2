"""Audit wording-triggered runtime heuristics against paraphrase pairs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bgrag.config import Settings
from bgrag.manifests import build_run_name
from bgrag.serving.retry_policy import assess_question_risk
from bgrag.retrieval.topology import _requires_authority_support


@dataclass(frozen=True)
class HeuristicTriggerAuditCase:
    source_case_id: str
    paraphrase_case_id: str
    source_question: str
    paraphrase_question: str
    authority_support_original: bool
    authority_support_paraphrase: bool
    question_risk_original: dict[str, object]
    question_risk_paraphrase: dict[str, object]
    changed_signals: list[str]


@dataclass(frozen=True)
class HeuristicTriggerAuditRun:
    run_name: str
    manifest_path: str
    generated_eval_path: str
    case_count: int
    authority_support_changed_count: int
    question_risk_level_changed_count: int
    exactness_changed_count: int
    branch_changed_count: int
    cases: list[HeuristicTriggerAuditCase]


def _load_manifest(settings: Settings, manifest_path: Path) -> dict[str, object]:
    resolved = settings.resolve(manifest_path)
    return json.loads(resolved.read_text(encoding="utf-8"))


def run_heuristic_trigger_audit(settings: Settings, *, manifest_path: Path) -> HeuristicTriggerAuditRun:
    manifest = _load_manifest(settings, manifest_path)
    cases: list[HeuristicTriggerAuditCase] = []
    authority_support_changed_count = 0
    question_risk_level_changed_count = 0
    exactness_changed_count = 0
    branch_changed_count = 0

    for pair in manifest["pairs"]:
        source_question = str(pair.get("source_question") or "")
        paraphrase_question = str(pair["paraphrase_question"])
        if not source_question:
            source_case_id = str(pair["source_case_id"])
            for rel_path in manifest["source_eval_paths"]:
                path = settings.resolve(Path(rel_path))
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    case = json.loads(line)
                    if str(case["id"]) == source_case_id:
                        source_question = str(case["question"])
                        break
                if source_question:
                    break
            if not source_question:
                raise RuntimeError(f"Could not resolve source question for {source_case_id}")

        authority_original = _requires_authority_support(source_question)
        authority_paraphrase = _requires_authority_support(paraphrase_question)
        risk_original = assess_question_risk(source_question).model_dump(mode="json")
        risk_paraphrase = assess_question_risk(paraphrase_question).model_dump(mode="json")

        changed_signals: list[str] = []
        if authority_original != authority_paraphrase:
            authority_support_changed_count += 1
            changed_signals.append("authority_support")
        if risk_original["risk_level"] != risk_paraphrase["risk_level"]:
            question_risk_level_changed_count += 1
            changed_signals.append("question_risk_level")
        if risk_original["exactness_sensitive"] != risk_paraphrase["exactness_sensitive"]:
            exactness_changed_count += 1
            changed_signals.append("exactness_sensitive")
        if risk_original["branch_sensitive"] != risk_paraphrase["branch_sensitive"]:
            branch_changed_count += 1
            changed_signals.append("branch_sensitive")

        cases.append(
            HeuristicTriggerAuditCase(
                source_case_id=str(pair["source_case_id"]),
                paraphrase_case_id=str(pair["paraphrase_case_id"]),
                source_question=source_question,
                paraphrase_question=paraphrase_question,
                authority_support_original=authority_original,
                authority_support_paraphrase=authority_paraphrase,
                question_risk_original=risk_original,
                question_risk_paraphrase=risk_paraphrase,
                changed_signals=changed_signals,
            )
        )

    return HeuristicTriggerAuditRun(
        run_name=build_run_name("heuristic_trigger_audit"),
        manifest_path=str(settings.resolve(manifest_path)),
        generated_eval_path=str(settings.resolve(Path(str(manifest["output_eval_path"])))),
        case_count=len(cases),
        authority_support_changed_count=authority_support_changed_count,
        question_risk_level_changed_count=question_risk_level_changed_count,
        exactness_changed_count=exactness_changed_count,
        branch_changed_count=branch_changed_count,
        cases=cases,
    )


def render_heuristic_trigger_audit_markdown(run: HeuristicTriggerAuditRun) -> str:
    lines = [
        f"# Heuristic Trigger Audit: {run.run_name}",
        "",
        f"- manifest_path: {run.manifest_path}",
        f"- generated_eval_path: {run.generated_eval_path}",
        f"- case_count: {run.case_count}",
        f"- authority_support_changed_count: {run.authority_support_changed_count}",
        f"- question_risk_level_changed_count: {run.question_risk_level_changed_count}",
        f"- exactness_changed_count: {run.exactness_changed_count}",
        f"- branch_changed_count: {run.branch_changed_count}",
        "",
        "## Cases",
        "",
    ]
    for case in run.cases:
        lines.extend(
            [
                f"### {case.source_case_id} -> {case.paraphrase_case_id}",
                f"- changed_signals: {', '.join(case.changed_signals) or '<none>'}",
                f"- authority_support: {case.authority_support_original} -> {case.authority_support_paraphrase}",
                f"- question_risk_level: {case.question_risk_original['risk_level']} -> {case.question_risk_paraphrase['risk_level']}",
                f"- exactness_sensitive: {case.question_risk_original['exactness_sensitive']} -> {case.question_risk_paraphrase['exactness_sensitive']}",
                f"- branch_sensitive: {case.question_risk_original['branch_sensitive']} -> {case.question_risk_paraphrase['branch_sensitive']}",
                f"- source_question: {case.source_question}",
                f"- paraphrase_question: {case.paraphrase_question}",
                "",
            ]
        )
    return "\n".join(lines)


def write_heuristic_trigger_audit_artifacts(
    settings: Settings, run: HeuristicTriggerAuditRun
) -> tuple[Path, Path]:
    output_dir = settings.resolved_runs_dir / "heuristic_trigger_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run.run_name}.json"
    md_path = output_dir / f"{run.run_name}.md"
    json_path.write_text(
        json.dumps(
            {
                "run_name": run.run_name,
                "manifest_path": run.manifest_path,
                "generated_eval_path": run.generated_eval_path,
                "case_count": run.case_count,
                "authority_support_changed_count": run.authority_support_changed_count,
                "question_risk_level_changed_count": run.question_risk_level_changed_count,
                "exactness_changed_count": run.exactness_changed_count,
                "branch_changed_count": run.branch_changed_count,
                "cases": [asdict(case) for case in run.cases],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(render_heuristic_trigger_audit_markdown(run), encoding="utf-8")
    return json_path, md_path
