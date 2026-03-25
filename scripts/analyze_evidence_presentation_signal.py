from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import REPO_ROOT
from bgrag.answering.strategies import _build_inline_evidence_prompt
from bgrag.eval.retrieval_metrics import _normalized_urls, _prefix_hit
from bgrag.types import ChunkRecord, EvalCase


@dataclass
class CasePresentationSignal:
    case_id: str
    required_claim_recall: float
    packed_claim_evidence_recall: float
    recall_gap: float
    packed_chunk_count: int
    prompt_chars: int
    prompt_words: int
    total_chunk_chars: int
    mean_chunk_chars: float
    max_chunk_chars: int
    earliest_primary_rank: float
    earliest_supporting_rank: float
    claim_top1_recall: float
    claim_top3_recall: float
    claim_top5_recall: float
    claim_top8_recall: float
    claim_mean_earliest_rank: float


def _load_run(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rankdata(values: list[float]) -> list[float]:
    sorted_idx = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if not pairs:
        return float("nan")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return float("nan")
    return numerator / (denominator_x * denominator_y)


def _spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if not pairs:
        return float("nan")
    ranked_xs = _rankdata([x for x, _ in pairs])
    ranked_ys = _rankdata([y for _, y in pairs])
    return _pearson(ranked_xs, ranked_ys)


def _as_chunks(raw_chunks: list[dict[str, object]]) -> list[ChunkRecord]:
    return [ChunkRecord.model_validate(chunk) for chunk in raw_chunks]


def _earliest_rank_for_urls(chunks: list[ChunkRecord], urls: list[str], prefixes: list[str]) -> float:
    normalized_urls = _normalized_urls(urls)
    normalized_prefixes = [prefix.strip() for prefix in prefixes if prefix.strip()]
    for index, chunk in enumerate(chunks, start=1):
        chunk_url = chunk.canonical_url.strip().rstrip("/") if chunk.canonical_url else ""
        chunk_doc_id = chunk.doc_id.strip() if chunk.doc_id else ""
        if normalized_urls and chunk_url in normalized_urls:
            return float(index)
        if not normalized_urls and normalized_prefixes and any(chunk_doc_id.startswith(prefix) for prefix in normalized_prefixes):
            return float(index)
    return float("nan")


def _claim_rank_metrics(case: EvalCase, chunks: list[ChunkRecord]) -> tuple[float, float, float, float, float]:
    if not case.claim_evidence:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    claim_ranks: list[float] = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top8_hits = 0
    for item in case.claim_evidence:
        rank = _earliest_rank_for_urls(chunks, item.evidence_doc_urls, item.evidence_doc_prefixes)
        claim_ranks.append(rank)
        if not math.isnan(rank):
            if rank <= 1:
                top1_hits += 1
            if rank <= 3:
                top3_hits += 1
            if rank <= 5:
                top5_hits += 1
            if rank <= 8:
                top8_hits += 1
    claim_total = len(case.claim_evidence)
    valid_ranks = [rank for rank in claim_ranks if not math.isnan(rank)]
    return (
        top1_hits / claim_total,
        top3_hits / claim_total,
        top5_hits / claim_total,
        top8_hits / claim_total,
        statistics.mean(valid_ranks) if valid_ranks else float("nan"),
    )


def _case_signal(case_payload: dict[str, object]) -> CasePresentationSignal:
    case = EvalCase.model_validate(case_payload["case"])
    metrics = dict(case_payload["metrics"])
    answer = dict(case_payload["answer"])
    evidence_bundle = answer.get("evidence_bundle") or {}
    chunks = _as_chunks(list(evidence_bundle.get("packed_chunks", [])))
    prompt = _build_inline_evidence_prompt(case.question, chunks)
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    claim_top1, claim_top3, claim_top5, claim_top8, claim_mean_rank = _claim_rank_metrics(case, chunks)
    return CasePresentationSignal(
        case_id=case.id,
        required_claim_recall=float(metrics["required_claim_recall"]),
        packed_claim_evidence_recall=float(metrics["packed_claim_evidence_recall"]),
        recall_gap=float(metrics["packed_claim_evidence_recall"]) - float(metrics["required_claim_recall"]),
        packed_chunk_count=len(chunks),
        prompt_chars=len(prompt),
        prompt_words=len(prompt.split()),
        total_chunk_chars=sum(chunk_lengths),
        mean_chunk_chars=statistics.mean(chunk_lengths) if chunk_lengths else 0.0,
        max_chunk_chars=max(chunk_lengths) if chunk_lengths else 0,
        earliest_primary_rank=_earliest_rank_for_urls(chunks, case.primary_urls, case.expected_doc_prefixes),
        earliest_supporting_rank=_earliest_rank_for_urls(chunks, case.supporting_urls, case.supporting_doc_prefixes),
        claim_top1_recall=claim_top1,
        claim_top3_recall=claim_top3,
        claim_top5_recall=claim_top5,
        claim_top8_recall=claim_top8,
        claim_mean_earliest_rank=claim_mean_rank,
    )


def _format_float(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def _render_markdown(run_path: Path, signals: list[CasePresentationSignal]) -> str:
    gaps = [signal.recall_gap for signal in signals]
    correlations = [
        ("prompt_chars", [float(signal.prompt_chars) for signal in signals]),
        ("prompt_words", [float(signal.prompt_words) for signal in signals]),
        ("packed_chunk_count", [float(signal.packed_chunk_count) for signal in signals]),
        ("total_chunk_chars", [float(signal.total_chunk_chars) for signal in signals]),
        ("mean_chunk_chars", [float(signal.mean_chunk_chars) for signal in signals]),
        ("max_chunk_chars", [float(signal.max_chunk_chars) for signal in signals]),
        ("earliest_primary_rank", [float(signal.earliest_primary_rank) for signal in signals]),
        ("earliest_supporting_rank", [float(signal.earliest_supporting_rank) for signal in signals]),
        ("claim_top1_recall", [float(signal.claim_top1_recall) for signal in signals]),
        ("claim_top3_recall", [float(signal.claim_top3_recall) for signal in signals]),
        ("claim_top5_recall", [float(signal.claim_top5_recall) for signal in signals]),
        ("claim_top8_recall", [float(signal.claim_top8_recall) for signal in signals]),
        ("claim_mean_earliest_rank", [float(signal.claim_mean_earliest_rank) for signal in signals]),
    ]
    lines = [
        "# Evidence Presentation Signal Analysis",
        "",
        f"- run_file: `{run_path}`",
        f"- case_count: {len(signals)}",
        "",
        "## Correlation With Recall Gap",
        "",
        "| metric | spearman_to_gap | pearson_to_gap |",
        "|---|---:|---:|",
    ]
    for label, values in correlations:
        lines.append(f"| {label} | {_format_float(_spearman(values, gaps))} | {_format_float(_pearson(values, gaps))} |")
    lines.extend(
        [
            "",
            "## Largest Recall Gaps",
            "",
            "| case_id | recall_gap | answer_recall | prompt_chars | packed_chunks | earliest_primary | claim_top3 | claim_mean_rank |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for signal in sorted(signals, key=lambda item: (-item.recall_gap, -item.prompt_chars))[:12]:
        lines.append(
            f"| {signal.case_id} | {signal.recall_gap:.4f} | {signal.required_claim_recall:.4f} | "
            f"{signal.prompt_chars} | {signal.packed_chunk_count} | {_format_float(signal.earliest_primary_rank)} | "
            f"{_format_float(signal.claim_top3_recall)} | {_format_float(signal.claim_mean_earliest_rank)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether baseline evidence-presentation properties correlate with answer-side recall gap."
    )
    parser.add_argument("run_path", help="Path to an eval run artifact JSON")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    if not run_path.is_absolute():
        run_path = (REPO_ROOT / run_path).resolve()
    payload = _load_run(run_path)
    signals = [_case_signal(case_payload) for case_payload in payload.get("cases", [])]
    report = _render_markdown(run_path, signals)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = REPO_ROOT / "datasets" / "runs" / f"evidence_presentation_signal_{run_path.stem}_{stamp}.md"
    output_path.write_text(report, encoding="utf-8")
    print(output_path)
    print(report)


if __name__ == "__main__":
    main()
