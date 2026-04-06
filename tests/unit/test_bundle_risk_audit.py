import json
from datetime import datetime, timezone
from pathlib import Path

from bgrag.benchmarks.bundle_risk import render_bundle_risk_audit_markdown, run_bundle_risk_audit
from bgrag.config import Settings
from bgrag.types import (
    AnswerCitation,
    AnswerResult,
    ChunkRecord,
    EvidenceBundle,
    EvalCase,
    EvalCaseResult,
    EvalRunResult,
    RetrievalCandidate,
    SourceFamily,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _chunk(chunk_id: str, *, doc_id: str, heading_path: list[str], family: SourceFamily = SourceFamily.BUYERS_GUIDE):
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        canonical_url=f"https://example.com/{doc_id}",
        title=doc_id,
        source_family=family,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text=f"text for {chunk_id}",
        heading_path=heading_path,
    )


def _candidate(chunk: ChunkRecord, score: float = 1.0) -> RetrievalCandidate:
    return RetrievalCandidate(chunk=chunk, blended_score=score, rerank_score=score)


def test_run_bundle_risk_audit_summarizes_flagged_low_recall_cases(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()

    low_raw = [
        _chunk("c1", doc_id="d1", heading_path=["A"]),
        _chunk("c2", doc_id="d1", heading_path=["A"]),
        _chunk("c3", doc_id="d2", heading_path=["B"]),
        _chunk("c4", doc_id="d3", heading_path=["C"], family=SourceFamily.BUY_CANADIAN_POLICY),
        _chunk("c5", doc_id="d4", heading_path=["D"], family=SourceFamily.TBS_DIRECTIVE),
        _chunk("c6", doc_id="d5", heading_path=["E"]),
    ]
    low_evidence = EvidenceBundle(
        query="q1",
        raw_shortlist=[_candidate(chunk) for chunk in low_raw],
        selected_candidates=[_candidate(chunk) for chunk in low_raw[:4]],
        packed_chunks=[low_raw[0], low_raw[1], low_raw[2], low_raw[0]],
    )
    good_chunks = [
        _chunk("g1", doc_id="g1", heading_path=["A"]),
        _chunk("g2", doc_id="g2", heading_path=["B"]),
        _chunk("g3", doc_id="g3", heading_path=["C"], family=SourceFamily.BUY_CANADIAN_POLICY),
        _chunk("g4", doc_id="g4", heading_path=["D"], family=SourceFamily.TBS_DIRECTIVE),
    ]
    good_evidence = EvidenceBundle(
        query="q2",
        raw_shortlist=[_candidate(chunk) for chunk in good_chunks],
        selected_candidates=[_candidate(chunk) for chunk in good_chunks],
        packed_chunks=good_chunks,
    )

    run = EvalRunResult(
        run_name="eval_test",
        created_at=datetime(2026, 4, 4, tzinfo=timezone.utc),
        profile_name="baseline_vector_rerank_shortlist",
        answer_model="command-a-03-2025",
        judge_model="command-a-03-2025",
        cases=[
            EvalCaseResult(
                case=EvalCase(id="HR_LOW", question="Low recall case", split="holdout"),
                answer=AnswerResult(
                    question="Low recall case",
                    answer_text="answer",
                    strategy_name="inline_evidence_chat",
                    model_name="command-a-03-2025",
                    citations=[AnswerCitation(chunk_id="c1", canonical_url="https://example.com/d1")],
                    evidence_bundle=low_evidence,
                ),
                metrics={"required_claim_recall": 0.5},
            ),
            EvalCaseResult(
                case=EvalCase(id="HR_GOOD", question="Good case", split="dev"),
                answer=AnswerResult(
                    question="Good case",
                    answer_text="answer",
                    strategy_name="inline_evidence_chat",
                    model_name="command-a-03-2025",
                    citations=[AnswerCitation(chunk_id="g1", canonical_url="https://example.com/g1")],
                    evidence_bundle=good_evidence,
                ),
                metrics={"required_claim_recall": 1.0},
            ),
        ],
    )
    source_path = tmp_path / "runs" / "eval.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")

    audit = run_bundle_risk_audit(settings, source_run_path=source_path)

    assert audit.summary["case_count"] == 2
    assert audit.summary["low_recall_case_count"] == 1
    assert audit.summary["flagged_low_recall_case_count"] == 1
    assert audit.summary["flag_recall_on_low_recall"] == 1.0


def test_render_bundle_risk_audit_markdown_includes_summary_and_cases(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()

    chunk = _chunk("c1", doc_id="d1", heading_path=["A"])
    run = EvalRunResult(
        run_name="eval_test",
        created_at=datetime(2026, 4, 4, tzinfo=timezone.utc),
        profile_name="baseline_vector_rerank_shortlist",
        answer_model="command-a-03-2025",
        judge_model="command-a-03-2025",
        cases=[
            EvalCaseResult(
                case=EvalCase(id="HR_001", question="Question", split="dev"),
                answer=AnswerResult(
                    question="Question",
                    answer_text="answer",
                    strategy_name="inline_evidence_chat",
                    model_name="command-a-03-2025",
                    evidence_bundle=EvidenceBundle(
                        query="Question",
                        raw_shortlist=[_candidate(chunk)],
                        selected_candidates=[_candidate(chunk)],
                        packed_chunks=[chunk],
                    ),
                ),
                metrics={"required_claim_recall": 1.0},
            )
        ],
    )
    source_path = tmp_path / "runs" / "eval.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")

    audit = run_bundle_risk_audit(settings, source_run_path=source_path)
    markdown = render_bundle_risk_audit_markdown(audit)

    assert "# Bundle Risk Audit" in markdown
    assert "- case_count: 1" in markdown
    assert "### HR_001" in markdown
