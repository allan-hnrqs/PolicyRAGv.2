from __future__ import annotations

from pathlib import Path

from bgrag.config import Settings
from bgrag.product_benchmark import (
    ProductBenchmarkCase,
    ProductBenchmarkCaseResult,
    ProductChatTurn,
    evaluate_product_case,
    load_product_benchmark_manifest,
    summarize_product_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_product_benchmark_manifest() -> None:
    manifest = load_product_benchmark_manifest(
        REPO_ROOT / "datasets" / "eval" / "manifests" / "product_serving_benchmark_v1.json"
    )
    assert manifest.name == "product_serving_benchmark_v1"
    assert len(manifest.cases) >= 8
    assert any(case.id == "SO_002" for case in manifest.cases)


def test_evaluate_product_case_records_raw_output(monkeypatch) -> None:
    def fake_run_demo_query(settings, question, profile_name="demo", messages=None):
        return {
            "resolved_question": "Under Canadian procurement policy, when does a standing offer or a supply arrangement become binding?",
            "answer_text": "Canada becomes bound when a call-up or contract is issued.",
            "citations": [{"chunk_id": "c1", "canonical_url": "https://example.test", "snippet": None}],
            "timings": {"total_request_seconds": 2.5, "retrieval_seconds": 0.5},
            "notes": ["conversation_route:history_contextualizer"],
            "response_mode": "rag",
        }

    monkeypatch.setattr("bgrag.product_benchmark.run_demo_query", fake_run_demo_query)

    case = ProductBenchmarkCase(
        id="SO_002",
        question="What about under each one, when is Canada legally bound?",
        messages=[
            ProductChatTurn(role="user", content="What is the difference between a standing offer and a supply arrangement?"),
            ProductChatTurn(role="assistant", content="They are different methods of supply."),
            ProductChatTurn(role="user", content="What about under each one, when is Canada legally bound?"),
        ],
        tags=["follow_up"],
        quality_focus=["follow_up_resolution"],
    )

    result = evaluate_product_case(Settings(project_root=REPO_ROOT), "demo", case)

    assert result.error is None
    assert result.case_id == "SO_002"
    assert result.citation_count == 1
    assert result.response_mode == "rag"
    assert result.timings["total_request_seconds"] == 2.5
    assert result.notes == ["conversation_route:history_contextualizer"]


def test_summarize_product_benchmark_counts_errors_and_timings() -> None:
    ok = ProductBenchmarkCaseResult(
        case_id="A",
        question="q",
        timings={"total_request_seconds": 2.0, "retrieval_seconds": 1.0},
    )
    errored = ProductBenchmarkCaseResult(
        case_id="B",
        question="q",
        error="RuntimeError('boom')",
        timings={"elapsed_wall_seconds": 1.0},
    )

    summary = summarize_product_benchmark([ok, errored])

    assert summary["case_count"] == 2
    assert summary["successful_case_count"] == 1
    assert summary["error_case_count"] == 1
    assert summary["error_case_ids"] == ["B"]
    assert summary["stage_timings"]["total_request_seconds"]["mean"] == 2.0
