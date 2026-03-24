"""Secondary Ragas-based evaluation lane.

This module is intentionally additive: it does not replace the repo's existing
deterministic metrics or the current structured Cohere judge. It provides a
second measurement surface based on Ragas' claim-oriented metrics so we can
cross-check evaluation conclusions.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"instructor\.providers\.gemini\.client",
)
warnings.filterwarnings(
    "ignore",
    message=r"\s*All support for the `google\.generativeai` package has ended\..*",
    category=FutureWarning,
)

import cohere
from instructor import from_cohere
from ragas import EvaluationDataset, evaluate
from ragas.cache import DiskCacheBackend
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

from bgrag.config import Settings
from bgrag.manifests import build_run_name
from bgrag.eval.loader import load_eval_cases
from bgrag.profiles.models import RuntimeProfile
from bgrag.types import (
    AnswerResult,
    ChunkRecord,
    EvalCase,
    RagasCaseResult,
    RagasRunResult,
)

RAGAS_CACHE_SUBDIR = "ragas"


def _format_chunk_context(chunk: ChunkRecord) -> str:
    heading = " > ".join(chunk.heading_path) if chunk.heading_path else (chunk.heading or "")
    parts = [
        f"Title: {chunk.title}",
        f"URL: {chunk.canonical_url}",
    ]
    if heading:
        parts.append(f"Heading: {heading}")
    parts.append(f"Content: {chunk.text}")
    return "\n".join(parts)


def _build_ragas_llm(settings: Settings) -> InstructorLLM:
    settings.require_cohere_key("Ragas evaluation")
    cache_dir = settings.resolve(settings.cache_dir) / RAGAS_CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = DiskCacheBackend(cache_dir=str(cache_dir))

    patched_client = from_cohere(cohere.ClientV2(settings.cohere_api_key))
    llm = InstructorLLM(
        client=patched_client,
        model=settings.cohere_judge_model,
        provider="cohere",
        model_args=InstructorModelArgs(
            max_tokens=settings.ragas_max_output_tokens,
            temperature=0.0,
        ),
        cache=cache,
    )
    # Cohere V2 chat rejects top_p, but InstructorModelArgs currently includes
    # it by default. The stock ragas llm_factory path does not handle this
    # correctly for Cohere in the installed version, so we patch it here.
    llm.model_args.pop("top_p", None)
    return llm


def _build_metrics(llm: InstructorLLM) -> list[object]:
    context_recall = ContextRecall(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    correctness = FactualCorrectness(llm=llm, mode="precision")
    coverage = FactualCorrectness(llm=llm, mode="recall")
    correctness.name = "correctness_precision"
    coverage.name = "coverage_recall"
    return [context_recall, faithfulness, correctness, coverage]


def _normalize_metric_key(key: str) -> str:
    if key.startswith("correctness_precision"):
        return "correctness_precision"
    if key.startswith("coverage_recall"):
        return "coverage_recall"
    return key


def _normalize_metric_value(value: object) -> float | int | bool | str | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            converted = value.item()
        except Exception:
            return str(value)
        if isinstance(converted, float) and math.isnan(converted):
            return None
        if isinstance(converted, (bool, int, float, str)) or converted is None:
            return converted
        return str(converted)
    return str(value)


def _build_dataset_row(case: EvalCase, answer: AnswerResult) -> dict[str, object] | None:
    if not case.reference_answer:
        return None
    if not answer.evidence_bundle or not answer.evidence_bundle.packed_chunks:
        return None
    return {
        "user_input": case.question,
        "retrieved_contexts": [_format_chunk_context(chunk) for chunk in answer.evidence_bundle.packed_chunks],
        "response": answer.answer_text,
        "reference": case.reference_answer,
    }


def _compute_overall_metrics(case_results: list[RagasCaseResult]) -> dict[str, float | int | str]:
    evaluated = [case for case in case_results if case.evaluated]
    metric_names: set[str] = set()
    for case in evaluated:
        metric_names.update(
            key
            for key, value in case.metrics.items()
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))
        )

    overall: dict[str, float | int | str] = {
        "case_count": len(case_results),
        "evaluated_case_count": len(evaluated),
        "skipped_case_count": len(case_results) - len(evaluated),
    }
    for name in sorted(metric_names):
        values = [
            float(case.metrics[name])
            for case in evaluated
            if isinstance(case.metrics.get(name), (int, float))
            and not (isinstance(case.metrics.get(name), float) and math.isnan(case.metrics.get(name)))
        ]
        overall[f"{name}_mean"] = mean(values) if values else 0.0
    return overall


def run_ragas_eval(
    settings: Settings,
    profile: RuntimeProfile,
    eval_path: Path,
    answer_callback: Callable[[EvalCase], AnswerResult],
    run_manifest: dict[str, object] | None = None,
) -> RagasRunResult:
    cases = load_eval_cases(eval_path)
    llm = _build_ragas_llm(settings)
    metrics = _build_metrics(llm)

    prepared_rows: list[dict[str, object]] = []
    prepared_cases: list[tuple[EvalCase, AnswerResult]] = []
    case_results: list[RagasCaseResult] = []

    answer_phase_start = perf_counter()
    for case in cases:
        answer = answer_callback(case)
        dataset_row = _build_dataset_row(case, answer)
        if dataset_row is None:
            skip_reason = "reference_answer_missing" if not case.reference_answer else "packed_context_missing"
            case_results.append(
                RagasCaseResult(
                    case_id=case.id,
                    split=case.split,
                    question=case.question,
                    answer_strategy=answer.strategy_name,
                    answer_text=answer.answer_text,
                    packed_chunk_count=len(answer.evidence_bundle.packed_chunks) if answer.evidence_bundle else 0,
                    candidate_chunk_count=len(answer.evidence_bundle.candidates) if answer.evidence_bundle else 0,
                    evaluated=False,
                    skip_reason=skip_reason,
                    metrics={},
                )
            )
            continue
        prepared_rows.append(dataset_row)
        prepared_cases.append((case, answer))
    answer_phase_seconds = perf_counter() - answer_phase_start

    ragas_phase_seconds = 0.0
    if prepared_rows:
        ragas_phase_start = perf_counter()
        dataset = EvaluationDataset.from_list(prepared_rows)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=RunConfig(
                timeout=settings.ragas_timeout_seconds,
                max_workers=settings.ragas_max_workers,
            ),
            raise_exceptions=False,
            show_progress=False,
        )
        ragas_phase_seconds = perf_counter() - ragas_phase_start
        score_rows = result.scores
        for (case, answer), score_row in zip(prepared_cases, score_rows, strict=True):
            normalized_metrics = {
                _normalize_metric_key(key): _normalize_metric_value(value)
                for key, value in score_row.items()
            }
            evaluated = any(value is not None for value in normalized_metrics.values())
            case_results.append(
                RagasCaseResult(
                    case_id=case.id,
                    split=case.split,
                    question=case.question,
                    answer_strategy=answer.strategy_name,
                    answer_text=answer.answer_text,
                    packed_chunk_count=len(answer.evidence_bundle.packed_chunks) if answer.evidence_bundle else 0,
                    candidate_chunk_count=len(answer.evidence_bundle.candidates) if answer.evidence_bundle else 0,
                    evaluated=evaluated,
                    skip_reason=None if evaluated else "ragas_metric_timeout_or_error",
                    metrics=normalized_metrics,
                )
            )

    overall_metrics = _compute_overall_metrics(case_results)
    overall_metrics["answer_phase_seconds"] = answer_phase_seconds
    overall_metrics["ragas_phase_seconds"] = ragas_phase_seconds

    notes = [
        "Secondary eval lane only; compare against deterministic metrics and the main judge before promotion.",
        "Uses a repo-native Cohere InstructorLLM wrapper because the installed ragas llm_factory Cohere path is currently incompatible with cohere.ClientV2.",
        "Cases without reference answers or without packed evidence are skipped by design.",
        (
            "Ragas runtime uses explicit RunConfig settings from repo config. Metric-level "
            "timeouts or errors are recorded as missing values rather than aborting the full run."
        ),
    ]

    return RagasRunResult(
        run_name=build_run_name(f"{profile.name}_ragas"),
        created_at=datetime.now(timezone.utc),
        profile_name=profile.name,
        eval_model=settings.cohere_judge_model,
        run_manifest=run_manifest or {},
        cases=case_results,
        overall_metrics=overall_metrics,
        notes=notes,
    )
