"""Pairwise A/B comparison over existing eval run artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from diskcache import Cache
from openai import OpenAI

from bgrag.config import Settings
from bgrag.manifests import build_run_name
from bgrag.types import EvalCaseResult, EvalRunResult, PairwiseCaseResult, PairwiseJudgeVerdict, PairwiseRunResult


def _load_eval_run(path: Path) -> EvalRunResult:
    return EvalRunResult.model_validate_json(path.read_text(encoding="utf-8"))


def _case_result_map(run: EvalRunResult) -> dict[str, EvalCaseResult]:
    return {case_result.case.id: case_result for case_result in run.cases}


def _stable_order(case_id: str, control_run_name: str, candidate_run_name: str) -> Literal["control_first", "candidate_first"]:
    digest = hashlib.sha256(f"{case_id}|{control_run_name}|{candidate_run_name}".encode("utf-8")).hexdigest()
    return "control_first" if int(digest[:2], 16) % 2 == 0 else "candidate_first"


def _winner_to_source(raw_winner: str, answer_a_source: str, answer_b_source: str) -> str:
    normalized = raw_winner.strip().lower()
    if normalized in {"a", "answer_a"}:
        return answer_a_source
    if normalized in {"b", "answer_b"}:
        return answer_b_source
    return "tie"


def _prompt_payload(case_result_a: EvalCaseResult, case_result_b: EvalCaseResult) -> dict[str, object]:
    case = case_result_a.case
    return {
        "question": case.question,
        "reference_answer": case.reference_answer,
        "required_claims": case.required_claims,
        "forbidden_claims": case.forbidden_claims,
        "expect_abstain": case.expect_abstain,
        "answer_a": case_result_a.answer.answer_text,
        "answer_b": case_result_b.answer.answer_text,
    }


def _prompt_key(model: str, payload: dict[str, object]) -> str:
    stable = json.dumps({"model": model, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _build_instructions() -> str:
    return (
        "You are comparing two answers to the same procurement-policy evaluation case.\n"
        "Use the question, reference answer, required claims, and forbidden claims to decide which answer is better.\n"
        "Prefer answers that:\n"
        "- cover more of the required claims\n"
        "- stay faithful to the reference and do not invent unsupported details\n"
        "- avoid forbidden or unsafe content\n"
        "- abstain appropriately when the case expects abstention\n"
        "If the answers are effectively equal, return Tie.\n"
        "Judge only the answer texts you are given.\n"
        "Use only these exact values for winners: answer_a, answer_b, tie.\n"
        "Use only these exact values for confidence: low, medium, high.\n"
        "Keep the rationale concise: at most 2 sentences and under 80 words."
    )


class PairwiseOpenAIJudge:
    def __init__(self, settings: Settings) -> None:
        settings.require_openai_key("Pairwise OpenAI evaluation")
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.cache = Cache(str(settings.resolve(settings.cache_dir) / "openai_pairwise"))

    def judge(self, case_result_a: EvalCaseResult, case_result_b: EvalCaseResult) -> tuple[PairwiseJudgeVerdict, bool]:
        payload = _prompt_payload(case_result_a, case_result_b)
        cache_key = _prompt_key(self.settings.openai_eval_model, payload)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return PairwiseJudgeVerdict.model_validate(cached), True

        last_exception: Exception | None = None
        for max_output_tokens in (
            self.settings.openai_eval_max_output_tokens,
            self.settings.openai_eval_max_output_tokens * 2,
        ):
            try:
                response = self.client.responses.parse(
                    model=self.settings.openai_eval_model,
                    instructions=_build_instructions(),
                    input=json.dumps(payload, ensure_ascii=False, indent=2),
                    text_format=PairwiseJudgeVerdict,
                    max_output_tokens=max_output_tokens,
                    reasoning={"effort": self.settings.openai_eval_reasoning_effort},
                    prompt_cache_key=cache_key,
                    prompt_cache_retention="24h",
                    store=False,
                )
                verdict = response.output_parsed
                if verdict is None:
                    raise RuntimeError("OpenAI pairwise response did not contain a parsed structured verdict.")
                self.cache.set(cache_key, verdict.model_dump())
                return verdict, False
            except Exception as exc:  # pragma: no cover - exercised by live API only
                last_exception = exc

        assert last_exception is not None
        raise last_exception


def compare_pairwise_runs(
    settings: Settings,
    control_run_path: Path,
    candidate_run_path: Path,
    *,
    run_manifest: dict[str, object] | None = None,
) -> PairwiseRunResult:
    control_run = _load_eval_run(control_run_path)
    candidate_run = _load_eval_run(candidate_run_path)
    control_cases = _case_result_map(control_run)
    candidate_cases = _case_result_map(candidate_run)

    if set(control_cases) != set(candidate_cases):
        missing_from_candidate = sorted(set(control_cases) - set(candidate_cases))
        missing_from_control = sorted(set(candidate_cases) - set(control_cases))
        raise RuntimeError(
            "Pairwise comparison requires both run artifacts to contain the same case IDs. "
            f"Missing from candidate: {missing_from_candidate}. Missing from control: {missing_from_control}."
        )

    judge = PairwiseOpenAIJudge(settings)
    case_results: list[PairwiseCaseResult] = []
    for case_id in sorted(control_cases):
        control_case = control_cases[case_id]
        candidate_case = candidate_cases[case_id]
        order = _stable_order(case_id, control_run.run_name, candidate_run.run_name)
        if order == "control_first":
            answer_a_source = "control"
            answer_b_source = "candidate"
            verdict, cache_hit = judge.judge(control_case, candidate_case)
        else:
            answer_a_source = "candidate"
            answer_b_source = "control"
            verdict, cache_hit = judge.judge(candidate_case, control_case)

        case_results.append(
            PairwiseCaseResult(
                case_id=case_id,
                split=control_case.case.split,
                question=control_case.case.question,
                control_run_name=control_run.run_name,
                candidate_run_name=candidate_run.run_name,
                answer_a_source=answer_a_source,
                answer_b_source=answer_b_source,
                overall_winner=_winner_to_source(verdict.winner, answer_a_source, answer_b_source),
                confidence=verdict.confidence,
                coverage_winner=_winner_to_source(verdict.coverage_winner, answer_a_source, answer_b_source),
                faithfulness_winner=_winner_to_source(verdict.faithfulness_winner, answer_a_source, answer_b_source),
                safety_winner=_winner_to_source(verdict.safety_winner, answer_a_source, answer_b_source),
                rationale=verdict.rationale,
                control_answer_text=control_case.answer.answer_text,
                candidate_answer_text=candidate_case.answer.answer_text,
                cache_hit=cache_hit,
            )
        )

    control_wins = sum(1 for case in case_results if case.overall_winner == "control")
    candidate_wins = sum(1 for case in case_results if case.overall_winner == "candidate")
    ties = sum(1 for case in case_results if case.overall_winner == "tie")
    non_ties = control_wins + candidate_wins
    cache_hits = sum(1 for case in case_results if case.cache_hit)

    return PairwiseRunResult(
        run_name=build_run_name(f"pairwise_{control_run.run_name}_vs_{candidate_run.run_name}"),
        created_at=datetime.now(timezone.utc),
        control_run_path=str(control_run_path),
        candidate_run_path=str(candidate_run_path),
        judge_model=settings.openai_eval_model,
        run_manifest=run_manifest or {},
        cases=case_results,
        overall_metrics={
            "case_count": len(case_results),
            "control_win_count": control_wins,
            "candidate_win_count": candidate_wins,
            "tie_count": ties,
            "candidate_win_rate_non_tie": (candidate_wins / non_ties) if non_ties else 0.0,
            "cache_hit_count": cache_hits,
        },
        notes=[
            "Pairwise A/B comparison is a secondary promotion check, not a replacement for deterministic metrics or the main judge harness.",
            "Uses official OpenAI Responses API structured parsing with gpt-5.4.",
            "Order is blinded and deterministic per case to reduce position bias while keeping runs reproducible.",
        ],
    )
