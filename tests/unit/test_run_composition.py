from datetime import datetime, timezone

import pytest

from bgrag.eval.run_composition import compose_eval_run, intervention_selected
from bgrag.types import AnswerResult, EvalCase, EvalCaseResult, EvalRunResult


def _case_result(case_id: str, *, recall: float, selected_path: str, seconds: float = 10.0) -> EvalCaseResult:
    return EvalCaseResult(
        case=EvalCase(id=case_id, question=f"Question {case_id}"),
        answer=AnswerResult(
            question=f"Question {case_id}",
            answer_text=f"Answer {case_id}",
            strategy_name="baseline",
            model_name="command-a",
            raw_response={"selected_path": selected_path},
        ),
        judgment={
            "required_claim_recall": recall,
            "forbidden_claim_violation_count": 0,
            "forbidden_claims_clean": True,
        },
        metrics={
            "required_claim_recall": recall,
            "abstained": False,
            "failed": False,
            "forbidden_claims_clean": True,
            "forbidden_claim_violation_count": 0,
            "judge_answer_abstains": False,
            "expect_abstain_annotated": False,
            "expect_abstain": None,
            "abstain_correct": None,
            "query_embedding_seconds": 0.1,
            "retrieval_seconds": 0.2,
            "answer_generation_seconds": 0.3,
            "judge_seconds": 0.4,
            "total_case_seconds": seconds,
            "packed_primary_url_hit": True,
            "candidate_primary_url_hit": True,
            "packed_supporting_url_hit": False,
            "candidate_supporting_url_hit": False,
            "packed_expected_url_recall": 1.0,
            "candidate_expected_url_recall": 1.0,
            "packed_claim_evidence_recall": 1.0,
            "candidate_claim_evidence_recall": 1.0,
            "claim_evidence_annotated": True,
        },
    )


def _run(name: str, cases: list[EvalCaseResult]) -> EvalRunResult:
    return EvalRunResult(
        run_name=name,
        created_at=datetime.now(timezone.utc),
        profile_name=name,
        answer_model="command-a",
        judge_model="command-a",
        run_manifest={"eval_path": "datasets/eval/parity/parity19.jsonl", "index_namespace": "baseline_ns"},
        cases=cases,
        overall_metrics={},
    )


def test_intervention_selected_uses_selected_path() -> None:
    keep_case = _case_result("HR_001", recall=0.5, selected_path="baseline_keep")
    rewrite_case = _case_result("HR_002", recall=1.0, selected_path="rewrite_structured_contract")

    assert intervention_selected(keep_case) is False
    assert intervention_selected(rewrite_case) is True


def test_compose_eval_run_uses_candidate_only_for_intervened_cases() -> None:
    control_cases = [
        _case_result("HR_001", recall=0.5, selected_path="inline_evidence_chat", seconds=10.0),
        _case_result("HR_002", recall=0.4, selected_path="inline_evidence_chat", seconds=12.0),
    ]
    candidate_cases = [
        _case_result("HR_001", recall=0.9, selected_path="rewrite_structured_contract", seconds=20.0),
        _case_result("HR_002", recall=0.1, selected_path="baseline_keep", seconds=25.0),
    ]
    composite = compose_eval_run(
        control_run=_run("control", control_cases),
        candidate_run=_run("candidate", candidate_cases),
        choose_candidate_case=intervention_selected,
    )

    assert [case.case.id for case in composite.cases] == ["HR_001", "HR_002"]
    assert composite.cases[0].metrics["required_claim_recall"] == 0.9
    assert composite.cases[1].metrics["required_claim_recall"] == 0.4
    assert composite.overall_metrics["required_claim_recall_mean"] == 0.65
    assert composite.overall_metrics["mean_case_seconds"] == 16.0
    assert "Selected candidate cases: HR_001" in composite.notes
    composed_from = composite.run_manifest["composed_from"]
    assert composed_from["non_selected_preserved_baseline"] is False
    assert composed_from["non_selected_changed_case_count"] == 1
    assert composed_from["non_selected_changed_case_ids"] == ["HR_002"]


def test_compose_eval_run_tracks_non_selected_preserved_baseline_when_unchanged() -> None:
    control_case = _case_result("HR_003", recall=0.7, selected_path="baseline_keep")
    candidate_case = _case_result("HR_003", recall=0.7, selected_path="baseline_keep")

    composite = compose_eval_run(
        control_run=_run("control", [control_case]),
        candidate_run=_run("candidate", [candidate_case]),
        choose_candidate_case=intervention_selected,
    )

    composed_from = composite.run_manifest["composed_from"]
    assert composed_from["non_selected_preserved_baseline"] is True
    assert composed_from["non_selected_changed_case_count"] == 0
    assert composed_from["non_selected_changed_case_ids"] == []


def test_compose_eval_run_tracks_abstention_accuracy() -> None:
    control_case = _case_result("HR_010", recall=1.0, selected_path="baseline_keep")
    control_case.metrics["expect_abstain_annotated"] = True
    control_case.metrics["expect_abstain"] = True
    control_case.metrics["judge_answer_abstains"] = True
    control_case.metrics["abstain_correct"] = True

    composite = compose_eval_run(
        control_run=_run("control", [control_case]),
        candidate_run=_run("candidate", [control_case]),
        choose_candidate_case=intervention_selected,
    )

    assert composite.overall_metrics["expect_abstain_annotated_case_count"] == 1
    assert composite.overall_metrics["judge_answer_abstain_count_annotated"] == 1
    assert composite.overall_metrics["abstain_correct_count"] == 1
    assert composite.overall_metrics["abstain_accuracy"] == 1.0


def test_compose_eval_run_rejects_mismatched_case_sets() -> None:
    control = _run("control", [_case_result("HR_001", recall=0.5, selected_path="baseline_keep")])
    candidate = _run("candidate", [_case_result("HR_002", recall=1.0, selected_path="rewrite_structured_contract")])

    with pytest.raises(RuntimeError, match="identical case IDs"):
        compose_eval_run(
            control_run=control,
            candidate_run=candidate,
            choose_candidate_case=intervention_selected,
        )


def test_compose_eval_run_rejects_mismatched_eval_provenance() -> None:
    control = _run("control", [_case_result("HR_001", recall=0.5, selected_path="baseline_keep")])
    candidate = _run("candidate", [_case_result("HR_001", recall=1.0, selected_path="rewrite_structured_contract")])
    candidate.run_manifest["eval_path"] = "datasets/eval/holdout/parity19_holdout.jsonl"

    with pytest.raises(RuntimeError, match="matching parent-run provenance"):
        compose_eval_run(
            control_run=control,
            candidate_run=candidate,
            choose_candidate_case=intervention_selected,
        )
