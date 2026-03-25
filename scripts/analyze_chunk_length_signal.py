from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import REPO_ROOT


@dataclass
class CaseSignal:
    case_id: str
    packed_claim_evidence_recall: float
    required_claim_recall: float
    recall_gap: float
    packed_chunk_count: int
    max_chunk_chars: int
    mean_chunk_chars: float
    total_chunk_chars: int
    count_over_1200: int
    count_over_1600: int
    count_over_3000: int
    share_chars_over_1600: float


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
    if not xs or not ys or len(xs) != len(ys):
        return float("nan")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return float("nan")
    return numerator / (denominator_x * denominator_y)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rankdata(xs), _rankdata(ys))


def _case_signal(case_payload: dict[str, object]) -> CaseSignal:
    answer = dict(case_payload["answer"])
    metrics = dict(case_payload["metrics"])
    case = dict(case_payload["case"])
    evidence_bundle = answer.get("evidence_bundle") or {}
    packed_chunks = list(evidence_bundle.get("packed_chunks", []))
    lengths = [len(str(chunk.get("text") or "")) for chunk in packed_chunks]
    total_chars = sum(lengths)
    packed_claim_recall = float(metrics["packed_claim_evidence_recall"])
    required_claim_recall = float(metrics["required_claim_recall"])
    return CaseSignal(
        case_id=str(case["id"]),
        packed_claim_evidence_recall=packed_claim_recall,
        required_claim_recall=required_claim_recall,
        recall_gap=packed_claim_recall - required_claim_recall,
        packed_chunk_count=len(lengths),
        max_chunk_chars=max(lengths) if lengths else 0,
        mean_chunk_chars=statistics.mean(lengths) if lengths else 0.0,
        total_chunk_chars=total_chars,
        count_over_1200=sum(1 for length in lengths if length > 1200),
        count_over_1600=sum(1 for length in lengths if length > 1600),
        count_over_3000=sum(1 for length in lengths if length > 3000),
        share_chars_over_1600=(sum(length for length in lengths if length > 1600) / total_chars) if total_chars else 0.0,
    )


def _render_markdown(run_path: Path, signals: list[CaseSignal]) -> str:
    xs_gap = [signal.recall_gap for signal in signals]
    correlation_rows = [
        ("max_chunk_chars", [float(signal.max_chunk_chars) for signal in signals]),
        ("mean_chunk_chars", [float(signal.mean_chunk_chars) for signal in signals]),
        ("total_chunk_chars", [float(signal.total_chunk_chars) for signal in signals]),
        ("count_over_1200", [float(signal.count_over_1200) for signal in signals]),
        ("count_over_1600", [float(signal.count_over_1600) for signal in signals]),
        ("count_over_3000", [float(signal.count_over_3000) for signal in signals]),
        ("share_chars_over_1600", [float(signal.share_chars_over_1600) for signal in signals]),
    ]
    lines = [
        "# Chunk Length Signal Analysis",
        "",
        f"- run_file: `{run_path}`",
        f"- case_count: {len(signals)}",
        "",
        "## Correlation With Recall Gap",
        "",
        "| metric | spearman_to_gap | pearson_to_gap |",
        "|---|---:|---:|",
    ]
    for label, xs in correlation_rows:
        lines.append(f"| {label} | {_spearman(xs, xs_gap):.4f} | {_pearson(xs, xs_gap):.4f} |")
    lines.extend(
        [
            "",
            "## Largest Recall Gaps",
            "",
            "| case_id | recall_gap | packed_claim_recall | answer_recall | max_chunk_chars | mean_chunk_chars | over_1600 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for signal in sorted(signals, key=lambda item: (-item.recall_gap, -item.max_chunk_chars))[:12]:
        lines.append(
            "| {case_id} | {recall_gap:.4f} | {packed_claim_evidence_recall:.4f} | {required_claim_recall:.4f} | "
            "{max_chunk_chars} | {mean_chunk_chars:.1f} | {count_over_1600} |".format(**signal.__dict__)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether packed chunk length correlates with answer-side recall gap.")
    parser.add_argument("run_path", help="Path to an eval run artifact JSON")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    if not run_path.is_absolute():
        run_path = (REPO_ROOT / run_path).resolve()
    payload = _load_run(run_path)
    signals = [_case_signal(case_payload) for case_payload in payload.get("cases", [])]
    report = _render_markdown(run_path, signals)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_name = f"chunk_length_signal_{run_path.stem}_{stamp}.md"
    output_path = REPO_ROOT / "datasets" / "runs" / output_name
    output_path.write_text(report, encoding="utf-8")
    print(output_path)
    print(report)


if __name__ == "__main__":
    main()
