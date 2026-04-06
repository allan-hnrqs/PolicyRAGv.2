from datetime import datetime, timezone

from bgrag.answer_replay_benchmark import (
    AnswerReplayBenchmarkRun,
    AnswerReplayCaseResult,
    render_answer_replay_markdown,
)


def test_render_answer_replay_markdown_includes_summary_and_cases() -> None:
    run = AnswerReplayBenchmarkRun(
        run_name="answer_replay_inline_123",
        created_at=datetime.now(timezone.utc),
        source_run_path="datasets/runs/example.json",
        source_profile_name="baseline_vector",
        strategy_name="inline_evidence_chat",
        answer_model="command-a-03-2025",
        case_filter_ids=["HR_001"],
        case_results=[
            AnswerReplayCaseResult(
                case_id="HR_001",
                question="Question?",
                strategy_name="inline_evidence_chat",
                answer_model="command-a-03-2025",
                source_profile_name="baseline_vector",
                citation_count=2,
                answer_generation_seconds=1.25,
                required_claim_recall=1.0,
                forbidden_claims_clean=True,
                answer_abstains=False,
                abstain_correct=None,
                answer_text="Answer.",
            )
        ],
        summary={
            "case_count": 1,
            "successful_case_count": 1,
            "error_case_count": 0,
            "required_claim_recall_mean": 1.0,
            "forbidden_claims_clean_rate": 1.0,
            "citation_count_mean": 2.0,
            "nonzero_citation_rate": 1.0,
            "answer_generation_seconds_mean": 1.25,
            "answer_generation_seconds_p50": 1.25,
            "answer_generation_seconds_p95": 1.25,
        },
    )

    markdown = render_answer_replay_markdown(run)

    assert "# Answer Replay Benchmark: inline_evidence_chat" in markdown
    assert "- case_filter_ids: HR_001" in markdown
    assert "- required_claim_recall_mean: 1.0000" in markdown
    assert "### HR_001" in markdown
    assert "- citation_count: 2" in markdown
