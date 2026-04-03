from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bgrag.config import Settings
from bgrag.official_site_baseline import (
    build_official_site_run_manifest,
    build_site_inventory_entries,
    render_official_site_baseline_markdown,
)
from bgrag.types import EvalCase, EvalCaseResult, EvalRunResult, NormalizedDocument, SourceFamily, SourceGraph

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_site_inventory_entries_includes_title_url_and_preview() -> None:
    document = NormalizedDocument(
        doc_id="doc1",
        title="Standing offers",
        source_url="https://example.test/source",
        canonical_url="https://example.test/standing-offers",
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        fetched_at=datetime.now(timezone.utc),
        content_hash="abc",
        word_count=100,
        extraction_method="unit_test",
        graph=SourceGraph(),
        raw_text="Standing offers let departments issue call-ups without creating a new solicitation each time.",
    )

    entries = build_site_inventory_entries([document], preview_chars=80)

    assert len(entries) == 1
    assert entries[0].title == "Standing offers"
    assert "https://example.test/standing-offers" in entries[0].inventory_text
    assert "Standing offers let departments issue call-ups" in entries[0].inventory_text


def test_build_official_site_run_manifest_records_answer_profile_and_limits() -> None:
    settings = Settings(project_root=REPO_ROOT)
    manifest = build_official_site_run_manifest(
        settings,
        eval_path=Path("datasets/eval/dev/parity19_dev.jsonl"),
        answer_profile_name="baseline_vector",
        max_live_pages=5,
        max_live_chunks=20,
    )

    assert manifest["mode"] == "official_site_live_browse_v1"
    assert manifest["answer_profile_name"] == "baseline_vector"
    assert manifest["max_live_pages"] == 5
    assert manifest["max_live_chunks"] == 20
    assert manifest["eval_path"] == "datasets/eval/dev/parity19_dev.jsonl"


def test_render_official_site_baseline_markdown_lists_visited_pages() -> None:
    case = EvalCase(id="HR_001", question="What is a standing offer?")
    case_result = EvalCaseResult(
        case=case,
        answer={
            "question": case.question,
            "answer_text": "A standing offer is not a contract until a call-up is issued.",
            "strategy_name": "inline_evidence_chat",
            "model_name": "command-a-03-2025",
            "raw_response": {
                "official_site_live_browse": {
                    "visited_pages": [
                        {
                            "canonical_url": "https://example.test/standing-offers",
                            "page_selection_score": 0.98,
                            "chunk_count": 4,
                        }
                    ]
                }
            },
        },
        judgment={"required_claim_recall": 1.0},
        metrics={"required_claim_recall": 1.0, "total_case_seconds": 3.2},
    )
    run = EvalRunResult(
        run_name="official_site_live_baseline_test",
        created_at=datetime.now(timezone.utc),
        profile_name="official_site_live_baseline_vector",
        answer_model="command-a-03-2025",
        judge_model="command-a-03-2025",
        cases=[case_result],
        overall_metrics={"required_claim_recall_mean": 1.0},
    )

    markdown = render_official_site_baseline_markdown(run)

    assert "# Official-Site Live Baseline" in markdown
    assert "https://example.test/standing-offers" in markdown
    assert "score=0.98" in markdown
