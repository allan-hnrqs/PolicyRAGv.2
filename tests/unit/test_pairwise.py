from pathlib import Path

from bgrag.config import Settings
from bgrag.eval import pairwise
from bgrag.types import AnswerResult, EvalCase, EvalCaseResult, EvalRunResult, PairwiseJudgeVerdict


def _case_result(case_id: str, answer_text: str) -> EvalCaseResult:
    return EvalCaseResult(
        case=EvalCase.model_validate(
            {
                "id": case_id,
                "question": f"Question {case_id}?",
                "split": "dev",
                "reference_answer": "Reference.",
                "required_claims": ["A"],
                "forbidden_claims": ["B"],
            }
        ),
        answer=AnswerResult(
            question=f"Question {case_id}?",
            answer_text=answer_text,
            strategy_name="baseline",
            model_name="command-a-03-2025",
        ),
        judgment={"required_claim_recall": 1.0},
        metrics={"required_claim_recall": 1.0},
    )


def test_stable_order_is_deterministic() -> None:
    first = pairwise._stable_order("HR_001", "control", "candidate")
    second = pairwise._stable_order("HR_001", "control", "candidate")
    assert first == second


def test_winner_to_source_maps_ab_tie() -> None:
    assert pairwise._winner_to_source("A", "control", "candidate") == "control"
    assert pairwise._winner_to_source("B", "control", "candidate") == "candidate"
    assert pairwise._winner_to_source("answer_a", "control", "candidate") == "control"
    assert pairwise._winner_to_source("answer_b", "control", "candidate") == "candidate"
    assert pairwise._winner_to_source("Tie", "control", "candidate") == "tie"


def test_compare_pairwise_runs_uses_cache_and_maps_winner(monkeypatch, tmp_path: Path) -> None:
    control_run = EvalRunResult(
        run_name="control_run",
        created_at="2026-03-24T00:00:00Z",
        profile_name="baseline",
        answer_model="command-a-03-2025",
        judge_model="command-a-03-2025",
        cases=[_case_result("HR_001", "control answer")],
    )
    candidate_run = EvalRunResult(
        run_name="candidate_run",
        created_at="2026-03-24T00:00:00Z",
        profile_name="candidate",
        answer_model="command-a-03-2025",
        judge_model="command-a-03-2025",
        cases=[_case_result("HR_001", "candidate answer")],
    )
    control_path = tmp_path / "control.json"
    candidate_path = tmp_path / "candidate.json"
    control_path.write_text(control_run.model_dump_json(indent=2), encoding="utf-8")
    candidate_path.write_text(candidate_run.model_dump_json(indent=2), encoding="utf-8")

    calls = {"count": 0}

    class FakeJudge:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def judge(self, case_result_a: EvalCaseResult, case_result_b: EvalCaseResult):
            calls["count"] += 1
            assert case_result_a.case.id == "HR_001"
            return (
                PairwiseJudgeVerdict(
                    winner="answer_a",
                    confidence="high",
                    coverage_winner="answer_a",
                    faithfulness_winner="tie",
                    safety_winner="tie",
                    rationale="A is more complete.",
                ),
                False,
            )

    monkeypatch.setattr(pairwise, "PairwiseOpenAIJudge", FakeJudge)
    settings = Settings(project_root=tmp_path, openai_api_key="test-key")

    result = pairwise.compare_pairwise_runs(settings, control_path, candidate_path)

    assert calls["count"] == 1
    assert result.overall_metrics["case_count"] == 1
    assert result.overall_metrics["control_win_count"] + result.overall_metrics["candidate_win_count"] == 1
    assert result.cases[0].overall_winner in {"control", "candidate"}
