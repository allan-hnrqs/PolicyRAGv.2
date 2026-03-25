"""Compose evaluation runs to isolate intervention effects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from bgrag.types import EvalCaseResult, EvalRunResult


def compute_overall_metrics(case_results: list[EvalCaseResult]) -> dict[str, float | int | bool | str | None]:
    recalls = [float(result.metrics["required_claim_recall"]) for result in case_results]
    failures = sum(1 for result in case_results if result.answer.failure_reason)
    forbidden_violations = sum(int(result.metrics["forbidden_claim_violation_count"]) for result in case_results)
    total_case_times = [float(result.metrics["total_case_seconds"]) for result in case_results]
    packed_primary_hits = [1.0 if result.metrics["packed_primary_url_hit"] else 0.0 for result in case_results]
    candidate_primary_hits = [1.0 if result.metrics["candidate_primary_url_hit"] else 0.0 for result in case_results]
    packed_expected_url_recalls = [float(result.metrics["packed_expected_url_recall"]) for result in case_results]
    candidate_expected_url_recalls = [float(result.metrics["candidate_expected_url_recall"]) for result in case_results]
    packed_claim_recalls = [float(result.metrics["packed_claim_evidence_recall"]) for result in case_results]
    candidate_claim_recalls = [float(result.metrics["candidate_claim_evidence_recall"]) for result in case_results]
    annotated_results = [result for result in case_results if bool(result.metrics["claim_evidence_annotated"])]
    annotated_packed_claim_recalls = [float(result.metrics["packed_claim_evidence_recall"]) for result in annotated_results]
    annotated_candidate_claim_recalls = [float(result.metrics["candidate_claim_evidence_recall"]) for result in annotated_results]
    abstention_annotated_results = [result for result in case_results if bool(result.metrics.get("expect_abstain_annotated"))]
    abstention_correct_results = [
        result for result in abstention_annotated_results if result.metrics.get("abstain_correct") is True
    ]
    abstention_abstained_results = [
        result for result in abstention_annotated_results if bool(result.metrics.get("judge_answer_abstains"))
    ]
    return {
        "required_claim_recall_mean": sum(recalls) / len(recalls) if recalls else 0.0,
        "answer_failure_count": failures,
        "forbidden_claim_violation_count": forbidden_violations,
        "case_count": len(case_results),
        "mean_case_seconds": sum(total_case_times) / len(total_case_times) if total_case_times else 0.0,
        "packed_primary_url_hit_rate": sum(packed_primary_hits) / len(packed_primary_hits) if packed_primary_hits else 0.0,
        "candidate_primary_url_hit_rate": (
            sum(candidate_primary_hits) / len(candidate_primary_hits) if candidate_primary_hits else 0.0
        ),
        "packed_expected_url_recall_mean": (
            sum(packed_expected_url_recalls) / len(packed_expected_url_recalls) if packed_expected_url_recalls else 0.0
        ),
        "candidate_expected_url_recall_mean": (
            sum(candidate_expected_url_recalls) / len(candidate_expected_url_recalls)
            if candidate_expected_url_recalls
            else 0.0
        ),
        "packed_claim_evidence_recall_mean": (
            sum(packed_claim_recalls) / len(packed_claim_recalls) if packed_claim_recalls else 0.0
        ),
        "candidate_claim_evidence_recall_mean": (
            sum(candidate_claim_recalls) / len(candidate_claim_recalls) if candidate_claim_recalls else 0.0
        ),
        "claim_evidence_annotated_case_count": len(annotated_results),
        "packed_claim_evidence_recall_mean_annotated": (
            sum(annotated_packed_claim_recalls) / len(annotated_packed_claim_recalls)
            if annotated_packed_claim_recalls
            else 0.0
        ),
        "candidate_claim_evidence_recall_mean_annotated": (
            sum(annotated_candidate_claim_recalls) / len(annotated_candidate_claim_recalls)
            if annotated_candidate_claim_recalls
            else 0.0
        ),
        "expect_abstain_annotated_case_count": len(abstention_annotated_results),
        "judge_answer_abstain_count_annotated": len(abstention_abstained_results),
        "abstain_correct_count": len(abstention_correct_results),
        "abstain_accuracy": (
            len(abstention_correct_results) / len(abstention_annotated_results) if abstention_annotated_results else 0.0
        ),
    }


def intervention_selected(
    case_result: EvalCaseResult,
    *,
    intervention_paths: set[str] | None = None,
) -> bool:
    allowed = intervention_paths or {"rewrite_structured_contract"}
    raw_response = case_result.answer.raw_response or {}
    selected_path = raw_response.get("selected_path")
    return isinstance(selected_path, str) and selected_path in allowed


def compose_eval_run(
    *,
    control_run: EvalRunResult,
    candidate_run: EvalRunResult,
    choose_candidate_case: Callable[[EvalCaseResult], bool],
    composite_run_name: str | None = None,
    notes: list[str] | None = None,
) -> EvalRunResult:
    candidate_by_id = {case.case.id: case for case in candidate_run.cases}
    composite_cases: list[EvalCaseResult] = []
    selected_case_ids: list[str] = []
    for control_case in control_run.cases:
        candidate_case = candidate_by_id.get(control_case.case.id)
        if candidate_case and choose_candidate_case(candidate_case):
            composite_cases.append(candidate_case)
            selected_case_ids.append(candidate_case.case.id)
        else:
            composite_cases.append(control_case)

    composite_notes = list(control_run.notes)
    composite_notes.extend(notes or [])
    composite_notes.append(
        "Composed from a control run plus candidate-case substitutions for intervention-selected cases only."
    )
    if selected_case_ids:
        composite_notes.append(f"Selected candidate cases: {', '.join(selected_case_ids)}")
    else:
        composite_notes.append("Selected candidate cases: none")

    run_name = composite_run_name or f"{candidate_run.profile_name}_intervention_only_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_manifest = dict(control_run.run_manifest)
    run_manifest["composed_from"] = {
        "control_run_name": control_run.run_name,
        "candidate_run_name": candidate_run.run_name,
        "selected_case_ids": selected_case_ids,
    }

    return EvalRunResult(
        run_name=run_name,
        created_at=datetime.now(timezone.utc),
        profile_name=f"{candidate_run.profile_name}__intervention_only",
        answer_model=candidate_run.answer_model,
        judge_model=candidate_run.judge_model,
        run_manifest=run_manifest,
        cases=composite_cases,
        overall_metrics=compute_overall_metrics(composite_cases),
        notes=composite_notes,
    )
