from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bgrag.config import Settings
from bgrag.tool_agent_baseline import (
    _normalize_action,
    build_tool_agent_run_manifest,
    render_tool_agent_baseline_markdown,
)
from bgrag.types import EvalCase, EvalCaseResult, EvalRunResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_normalize_action_validates_required_fields() -> None:
    search = _normalize_action(
        {"action": "search_inventory", "query": "late offers proof", "top_k": 5, "reason": "need pages"}
    )
    answer = _normalize_action(
        {"action": "answer", "reason": "enough evidence", "final_focus_query": "late offers proof"}
    )

    assert search["action"] == "search_inventory"
    assert search["top_k"] == 5
    assert answer["action"] == "answer"
    assert answer["final_focus_query"] == "late offers proof"


def test_build_tool_agent_run_manifest_records_limits() -> None:
    settings = Settings(project_root=REPO_ROOT)
    manifest = build_tool_agent_run_manifest(
        settings,
        eval_path=Path("datasets/eval/dev/parity19_dev.jsonl"),
        answer_profile_name="baseline_vector",
        max_steps=5,
        max_live_pages=6,
        max_live_chunks=24,
    )

    assert manifest["mode"] == "tool_using_official_site_agent_v1"
    assert manifest["answer_profile_name"] == "baseline_vector"
    assert manifest["max_steps"] == 5
    assert manifest["max_live_pages"] == 6
    assert manifest["max_live_chunks"] == 24


def test_render_tool_agent_baseline_markdown_lists_steps() -> None:
    case = EvalCase(id="HR_001", question="What is a standing offer?")
    case_result = EvalCaseResult(
        case=case,
        answer={
            "question": case.question,
            "answer_text": "A standing offer is not a contract until a call-up is issued.",
            "strategy_name": "inline_evidence_chat",
            "model_name": "command-a-03-2025",
            "raw_response": {
                "tool_agent_trace": {
                    "steps": [
                        {
                            "step": 1,
                            "action": "search_inventory",
                            "query": "standing offer binding",
                            "url": None,
                            "reason": "find candidate pages",
                        }
                    ]
                }
            },
        },
        judgment={"required_claim_recall": 1.0},
        metrics={"required_claim_recall": 1.0, "total_case_seconds": 5.1},
    )
    run = EvalRunResult(
        run_name="tool_agent_baseline_test",
        created_at=datetime.now(timezone.utc),
        profile_name="tool_agent_baseline_vector",
        answer_model="command-a-03-2025",
        judge_model="command-a-03-2025",
        cases=[case_result],
        overall_metrics={"required_claim_recall_mean": 1.0},
    )

    markdown = render_tool_agent_baseline_markdown(run)

    assert "# Tool Agent Baseline" in markdown
    assert "search_inventory" in markdown
    assert "standing offer binding" in markdown
