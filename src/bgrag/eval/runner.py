"""Evaluation runner."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

from bgrag.config import Settings
from bgrag.eval.judge import CohereJudge
from bgrag.eval.loader import load_eval_cases
from bgrag.eval.run_composition import compute_overall_metrics
from bgrag.manifests import build_run_name
from bgrag.eval.retrieval_metrics import compute_retrieval_metrics
from bgrag.profiles.models import RuntimeProfile
from bgrag.types import AnswerResult, EvalCase, EvalCaseResult, EvalRunResult


def run_eval(
    settings: Settings,
    profile: RuntimeProfile,
    eval_path: Path,
    answer_callback: Callable[[EvalCase], AnswerResult],
    run_manifest: dict[str, object] | None = None,
) -> EvalRunResult:
    cases = load_eval_cases(eval_path)
    judge = CohereJudge(settings)
    case_results: list[EvalCaseResult] = []
    for case in cases:
        case_start = perf_counter()
        answer = answer_callback(case)
        before_judge = perf_counter()
        judgment = judge.judge(case, answer)
        after_judge = perf_counter()
        packed_metrics = compute_retrieval_metrics(case, answer.evidence_bundle.packed_chunks if answer.evidence_bundle else [])
        candidate_metrics = compute_retrieval_metrics(
            case,
            [candidate.chunk for candidate in answer.evidence_bundle.candidates] if answer.evidence_bundle else [],
        )
        case_results.append(
            EvalCaseResult(
                case=case,
                answer=answer,
                judgment=judgment,
                metrics={
                    "required_claim_recall": float(judgment["required_claim_recall"]),
                    "abstained": answer.abstained,
                    "judge_answer_abstains": bool(judgment["answer_abstains"]),
                    "expect_abstain_annotated": case.expect_abstain is not None,
                    "expect_abstain": case.expect_abstain,
                    "abstain_correct": judgment["abstain_correct"],
                    "failed": bool(answer.failure_reason),
                    "forbidden_claims_clean": bool(judgment["forbidden_claims_clean"]),
                    "forbidden_claim_violation_count": int(judgment["forbidden_claim_violation_count"]),
                    "query_embedding_seconds": answer.timings.get("query_embedding_seconds", 0.0),
                    "retrieval_seconds": answer.timings.get("retrieval_seconds", 0.0),
                    "answer_generation_seconds": answer.timings.get("answer_generation_seconds", 0.0),
                    "judge_seconds": after_judge - before_judge,
                    "total_case_seconds": after_judge - case_start,
                    "packed_primary_url_hit": packed_metrics.primary_url_hit,
                    "candidate_primary_url_hit": candidate_metrics.primary_url_hit,
                    "packed_supporting_url_hit": packed_metrics.supporting_url_hit,
                    "candidate_supporting_url_hit": candidate_metrics.supporting_url_hit,
                    "packed_expected_url_recall": packed_metrics.expected_url_recall,
                    "candidate_expected_url_recall": candidate_metrics.expected_url_recall,
                    "packed_claim_evidence_recall": packed_metrics.claim_evidence_recall,
                    "candidate_claim_evidence_recall": candidate_metrics.claim_evidence_recall,
                    "claim_evidence_annotated": packed_metrics.claim_evidence_annotated,
                },
            )
        )
    return EvalRunResult(
        run_name=build_run_name(profile.name),
        created_at=datetime.now(timezone.utc),
        profile_name=profile.name,
        answer_model=settings.cohere_chat_model,
        judge_model=settings.cohere_judge_model,
        run_manifest=run_manifest or {},
        cases=case_results,
        overall_metrics=compute_overall_metrics(case_results),
    )
