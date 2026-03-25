from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bgrag.config import Settings
from bgrag.eval.conditional_compare import (
    CompositeRunArtifact,
    ConditionalCompareArtifacts,
    EvalRunArtifact,
    PairwiseRunArtifact,
    build_conditional_compare_summary,
    compose_conditional_run,
    render_composite_markdown,
    render_conditional_compare_summary_markdown,
    run_conditional_compare,
    write_conditional_compare_summary,
)
from bgrag.eval.run_composition import compute_overall_metrics
from bgrag.types import AnswerResult, EvalCase, EvalCaseResult, EvalRunResult, PairwiseRunResult


def _eval_run(name: str, *, recall: float, forbidden: int, abstain_accuracy: float = 0.0) -> EvalRunResult:
    return EvalRunResult(
        run_name=name,
        created_at=datetime.now(timezone.utc),
        profile_name=name,
        answer_model="command-a",
        judge_model="command-a",
        run_manifest={},
        cases=[],
        overall_metrics={
            "required_claim_recall_mean": recall,
            "forbidden_claim_violation_count": forbidden,
            "mean_case_seconds": 12.5,
            "abstain_accuracy": abstain_accuracy,
            "answer_failure_count": 0,
        },
    )


def _case_result(case_id: str, *, recall: float, selected_path: str) -> EvalCaseResult:
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
            "total_case_seconds": 1.0,
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


def _eval_run_with_cases(name: str, cases: list[EvalCaseResult]) -> EvalRunResult:
    return EvalRunResult(
        run_name=name,
        created_at=datetime.now(timezone.utc),
        profile_name=name,
        answer_model="command-a",
        judge_model="command-a",
        run_manifest={},
        cases=cases,
        overall_metrics=compute_overall_metrics(cases),
    )


def _pairwise_run(name: str) -> PairwiseRunResult:
    return PairwiseRunResult(
        run_name=name,
        created_at=datetime.now(timezone.utc),
        control_run_path="control.json",
        candidate_run_path="composite.json",
        judge_model="gpt-5.4",
        overall_metrics={
            "control_win_count": 2,
            "candidate_win_count": 4,
            "tie_count": 1,
            "candidate_win_rate_non_tie": 2 / 3,
            "cache_hit_count": 7,
        },
    )


def test_render_composite_markdown_shows_selected_cases() -> None:
    control = _eval_run("baseline_run", recall=0.7, forbidden=1)
    candidate = _eval_run("candidate_run", recall=0.8, forbidden=0)
    composite = _eval_run("candidate_intervention_only_run", recall=0.85, forbidden=0, abstain_accuracy=1.0)
    composite.run_manifest = {
        "composed_from": {
            "selected_case_ids": ["HR_016", "HR_038"],
            "non_selected_changed_case_ids": ["HR_005"],
            "non_selected_preserved_baseline": False,
        }
    }

    markdown = render_composite_markdown(
        control_run=control,
        candidate_run=candidate,
        composite_run=composite,
    )

    assert "- selected_case_count: 2" in markdown
    assert "- selected_case_ids: HR_016, HR_038" in markdown
    assert "- non_selected_changed_case_count: 1" in markdown
    assert "- non_selected_changed_case_ids: HR_005" in markdown
    assert "- composite: 0.850000" in markdown
    assert "- composite_abstain_accuracy: 1.000000" in markdown
    assert "- composite_artifact_mean_case_seconds: 12.500" in markdown
    assert "should not be read as end-to-end deployed latency" in markdown


def test_build_conditional_compare_summary_includes_pairwise_metrics() -> None:
    control = EvalRunArtifact(result=_eval_run("baseline_run", recall=0.7, forbidden=1), path=Path("control.json"))
    candidate = EvalRunArtifact(result=_eval_run("candidate_run", recall=0.8, forbidden=0), path=Path("candidate.json"))
    composite_result = _eval_run("candidate_intervention_only_run", recall=0.9, forbidden=0)
    composite_result.run_manifest = {
        "composed_from": {
            "selected_case_ids": ["HR_016"],
            "non_selected_changed_case_ids": ["HR_002"],
            "non_selected_preserved_baseline": False,
        }
    }
    composite = CompositeRunArtifact(
        result=composite_result,
        json_path=Path("composite.json"),
        markdown_path=Path("composite.md"),
    )
    pairwise = PairwiseRunArtifact(result=_pairwise_run("pairwise_run"), path=Path("pairwise.json"))

    summary = build_conditional_compare_summary(
        eval_path=Path("datasets/eval/holdout/parity19_holdout.jsonl"),
        index_namespace="baseline_ns",
        control_profile="baseline",
        candidate_profile="narrow_contract_slot_coverage_verifier_gated_structured_contract_answering",
        intervention_paths={"rewrite_structured_contract"},
        control_artifact=control,
        candidate_artifact=candidate,
        composite_artifact=composite,
        pairwise_artifact=pairwise,
        pairwise_error=None,
    )

    assert summary["selected_case_count"] == 1
    assert summary["selected_case_ids"] == ["HR_016"]
    assert summary["non_selected_changed_case_ids"] == ["HR_002"]
    assert summary["non_selected_preserved_baseline"] is False
    assert summary["composite_required_claim_recall"] == 0.9
    assert summary["pairwise_metrics"]["candidate_win_count"] == 4

    markdown = render_conditional_compare_summary_markdown(summary)
    assert "- non_selected_changed_case_count: 1" in markdown
    assert "- pairwise_run_path: pairwise.json" in markdown
    assert "- candidate_win_count: 4" in markdown


def test_write_conditional_compare_summary_writes_manifest(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_path": "datasets/eval/generated/missing_detail_focus.jsonl",
        "index_namespace": "baseline_ns",
        "control_profile": "baseline",
        "candidate_profile": "candidate",
        "intervention_paths": ["rewrite_structured_contract"],
        "selected_case_ids": ["HR_016"],
        "selected_case_count": 1,
        "non_selected_changed_case_ids": [],
        "non_selected_changed_case_count": 0,
        "non_selected_preserved_baseline": True,
        "control_run_path": "datasets/runs/control.json",
        "candidate_run_path": "datasets/runs/candidate.json",
        "composite_run_path": "datasets/runs/composite.json",
        "composite_summary_path": "datasets/runs/composite.md",
        "pairwise_run_path": None,
        "pairwise_error": "authentication failed",
        "control_required_claim_recall": 0.7,
        "candidate_required_claim_recall": 0.8,
        "composite_required_claim_recall": 0.85,
        "control_forbidden_claim_violations": 1,
        "candidate_forbidden_claim_violations": 0,
        "composite_forbidden_claim_violations": 0,
        "control_answer_failure_count": 0,
        "candidate_answer_failure_count": 0,
        "composite_answer_failure_count": 0,
        "control_abstain_accuracy": 0.0,
        "candidate_abstain_accuracy": 0.0,
        "composite_abstain_accuracy": 0.0,
    }

    json_path, markdown_path = write_conditional_compare_summary(settings, summary)
    manifest_path = tmp_path / "datasets" / "runs" / "manifests" / f"{json_path.stem}.manifest.json"

    assert json_path.exists()
    assert markdown_path.exists()
    assert "pairwise_error: authentication failed" in markdown_path.read_text(encoding="utf-8")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "conditional_compare_summary"
    assert manifest["run_artifact_path"].endswith(json_path.name)


def test_run_conditional_compare_records_resolved_index_namespace(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    control = EvalRunArtifact(result=_eval_run("baseline_run", recall=0.7, forbidden=1), path=Path("control.json"))
    candidate = EvalRunArtifact(result=_eval_run("candidate_run", recall=0.8, forbidden=0), path=Path("candidate.json"))
    composite_result = _eval_run("candidate_intervention_only_run", recall=0.9, forbidden=0)
    composite_result.run_manifest = {"composed_from": {"selected_case_ids": ["HR_016"]}}
    composite = CompositeRunArtifact(
        result=composite_result,
        json_path=Path("composite.json"),
        markdown_path=Path("composite.md"),
    )
    captured_summary: dict[str, object] = {}

    monkeypatch.setattr(
        "bgrag.eval.conditional_compare.load_index_manifest",
        lambda settings, namespace: {"namespace": "resolved_ns"},
    )

    def fake_run_profile_eval(*, settings, profile_name, eval_path, index_namespace, progress=None):
        assert index_namespace == "resolved_ns"
        return control if profile_name == "baseline" else candidate

    monkeypatch.setattr("bgrag.eval.conditional_compare.run_profile_eval", fake_run_profile_eval)
    monkeypatch.setattr(
        "bgrag.eval.conditional_compare.compose_conditional_run",
        lambda **kwargs: composite,
    )

    def fake_write_summary(settings: Settings, summary: dict[str, object]) -> tuple[Path, Path]:
        captured_summary.update(summary)
        return Path("summary.json"), Path("summary.md")

    monkeypatch.setattr(
        "bgrag.eval.conditional_compare.write_conditional_compare_summary",
        fake_write_summary,
    )

    artifacts = run_conditional_compare(
        settings=settings,
        eval_path=Path("datasets/eval/generated/missing_detail_focus.jsonl"),
        control_profile="baseline",
        candidate_profile="candidate",
        index_namespace=None,
        intervention_paths={"rewrite_structured_contract"},
        include_pairwise=False,
    )

    assert captured_summary["index_namespace"] == "resolved_ns"
    assert isinstance(artifacts, ConditionalCompareArtifacts)


def test_compose_conditional_run_rejects_profile_name_used_as_intervention_selector(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    control = EvalRunArtifact(
        result=_eval_run_with_cases(
            "baseline_run",
            [
                _case_result("HR_016", recall=0.5, selected_path="baseline_keep"),
                _case_result("HR_037", recall=0.5, selected_path="baseline_keep"),
            ],
        ),
        path=Path("control.json"),
    )
    candidate = EvalRunArtifact(
        result=_eval_run_with_cases(
            "missing_detail_exactness_verifier_gated_structured_contract_answering",
            [
                _case_result("HR_016", recall=1.0, selected_path="rewrite_structured_contract"),
                _case_result("HR_037", recall=0.5, selected_path="baseline_keep"),
            ],
        ),
        path=Path("candidate.json"),
    )

    with pytest.raises(RuntimeError, match="Use raw selected_path labels"):
        compose_conditional_run(
            settings=settings,
            control_artifact=control,
            candidate_artifact=candidate,
            intervention_paths={"missing_detail_exactness_verifier_gated_structured_contract_answering"},
        )


def test_compose_conditional_run_allows_zero_selected_cases_when_candidate_never_intervenes(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    control = EvalRunArtifact(
        result=_eval_run_with_cases(
            "baseline_run",
            [_case_result("HR_017", recall=1.0, selected_path="baseline_keep")],
        ),
        path=Path("control.json"),
    )
    candidate = EvalRunArtifact(
        result=_eval_run_with_cases(
            "missing_detail_exactness_verifier_gated_structured_contract_answering",
            [_case_result("HR_017", recall=1.0, selected_path="baseline_keep")],
        ),
        path=Path("candidate.json"),
    )

    composite = compose_conditional_run(
        settings=settings,
        control_artifact=control,
        candidate_artifact=candidate,
        intervention_paths={"missing_detail_exactness_verifier_gated_structured_contract_answering"},
    )

    assert composite.result.run_manifest["composed_from"]["selected_case_ids"] == []


def test_run_conditional_compare_emits_progress(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    control = EvalRunArtifact(result=_eval_run("baseline_run", recall=0.7, forbidden=1), path=Path("control.json"))
    candidate = EvalRunArtifact(result=_eval_run("candidate_run", recall=0.8, forbidden=0), path=Path("candidate.json"))
    composite_result = _eval_run("candidate_intervention_only_run", recall=0.9, forbidden=0)
    composite_result.run_manifest = {
        "composed_from": {
            "selected_case_ids": ["HR_016"],
            "non_selected_changed_case_ids": [],
            "non_selected_preserved_baseline": True,
        }
    }
    composite = CompositeRunArtifact(
        result=composite_result,
        json_path=Path("composite.json"),
        markdown_path=Path("composite.md"),
    )
    messages: list[str] = []

    monkeypatch.setattr(
        "bgrag.eval.conditional_compare.load_index_manifest",
        lambda settings, namespace: {"namespace": "resolved_ns"},
    )
    monkeypatch.setattr(
        "bgrag.eval.conditional_compare.run_profile_eval",
        lambda **kwargs: (
            messages.append(f"run_profile_eval:{kwargs['profile_name']}") or (
                control if kwargs["profile_name"] == "baseline" else candidate
            )
        ),
    )
    monkeypatch.setattr(
        "bgrag.eval.conditional_compare.compose_conditional_run",
        lambda **kwargs: (messages.append("compose_conditional_run") or composite),
    )
    monkeypatch.setattr(
        "bgrag.eval.conditional_compare.write_conditional_compare_summary",
        lambda settings, summary: (Path("summary.json"), Path("summary.md")),
    )

    run_conditional_compare(
        settings=settings,
        eval_path=Path("datasets/eval/generated/missing_detail_focus.jsonl"),
        control_profile="baseline",
        candidate_profile="candidate",
        index_namespace=None,
        intervention_paths={"rewrite_structured_contract"},
        include_pairwise=False,
        progress=messages.append,
    )

    assert messages[0] == "resolved_index_namespace=resolved_ns"
    assert "run_profile_eval:baseline" in messages
    assert "run_profile_eval:candidate" in messages
    assert "compose_conditional_run" in messages
    assert any(message.startswith("finished_summary json=summary.json") for message in messages)
