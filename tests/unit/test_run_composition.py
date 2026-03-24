from datetime import datetime, timezone

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
