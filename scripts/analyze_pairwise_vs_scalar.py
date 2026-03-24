from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


@dataclass
class CaseAnalysis:
    case_id: str
    pairwise_winner: str
    control_required_claim_recall: float
    candidate_required_claim_recall: float
    recall_delta: float
    control_length: int
    candidate_length: int
    length_delta: int
    candidate_length_ratio: float
    control_truncation_risk: bool
    candidate_truncation_risk: bool
    rationale: str


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "analysis"


def _case_map(run_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {case["case"]["id"]: case for case in run_payload["cases"]}


def _truncation_risk(answer_text: str) -> bool:
    stripped = answer_text.strip()
    if not stripped:
        return False
    if stripped.endswith(("...", "…", "[", "(", "-", "—")):
        return True
    if stripped[-1] not in ".!?)]}`\"'":
        return True
    tail = stripped[-12:].lower()
    return tail.endswith((" the", " and", " or", " if", " of", " to", " in", " is"))


def _case_analyses(
    pairwise_payload: dict[str, object],
    control_payload: dict[str, object],
    candidate_payload: dict[str, object],
) -> list[CaseAnalysis]:
    control_map = _case_map(control_payload)
    candidate_map = _case_map(candidate_payload)
    analyses: list[CaseAnalysis] = []
    for pairwise_case in pairwise_payload["cases"]:
        case_id = pairwise_case["case_id"]
        control_case = control_map[case_id]
        candidate_case = candidate_map[case_id]
        control_answer = str(control_case["answer"]["answer_text"])
        candidate_answer = str(candidate_case["answer"]["answer_text"])
        control_recall = float(control_case["metrics"]["required_claim_recall"])
        candidate_recall = float(candidate_case["metrics"]["required_claim_recall"])
        analyses.append(
            CaseAnalysis(
                case_id=case_id,
                pairwise_winner=str(pairwise_case["overall_winner"]),
                control_required_claim_recall=control_recall,
                candidate_required_claim_recall=candidate_recall,
                recall_delta=candidate_recall - control_recall,
                control_length=len(control_answer),
                candidate_length=len(candidate_answer),
                length_delta=len(candidate_answer) - len(control_answer),
                candidate_length_ratio=(len(candidate_answer) / len(control_answer)) if len(control_answer) else 0.0,
                control_truncation_risk=_truncation_risk(control_answer),
                candidate_truncation_risk=_truncation_risk(candidate_answer),
                rationale=str(pairwise_case["rationale"]),
            )
        )
    return analyses


def _summary(analyses: list[CaseAnalysis]) -> dict[str, object]:
    control_wins = [item for item in analyses if item.pairwise_winner == "control"]
    candidate_wins = [item for item in analyses if item.pairwise_winner == "candidate"]
    ties = [item for item in analyses if item.pairwise_winner == "tie"]
    positive_delta = [item for item in analyses if item.recall_delta > 0]
    pairwise_control_despite_positive_delta = [item for item in control_wins if item.recall_delta > 0]
    pairwise_control_equal_delta = [item for item in control_wins if item.recall_delta == 0]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "case_count": len(analyses),
        "control_win_count": len(control_wins),
        "candidate_win_count": len(candidate_wins),
        "tie_count": len(ties),
        "positive_recall_delta_case_count": len(positive_delta),
        "control_wins_despite_positive_recall_delta": len(pairwise_control_despite_positive_delta),
        "control_wins_with_equal_recall": len(pairwise_control_equal_delta),
        "candidate_truncation_risk_count": sum(1 for item in analyses if item.candidate_truncation_risk),
        "control_truncation_risk_count": sum(1 for item in analyses if item.control_truncation_risk),
        "mean_candidate_length_ratio": _mean([item.candidate_length_ratio for item in analyses]),
        "mean_candidate_length_ratio_control_wins": _mean([item.candidate_length_ratio for item in control_wins]),
        "mean_candidate_length_ratio_candidate_wins": _mean([item.candidate_length_ratio for item in candidate_wins]),
    }


def _render_markdown(
    *,
    pairwise_path: Path,
    control_path: Path,
    candidate_path: Path,
    analyses: list[CaseAnalysis],
    summary: dict[str, object],
) -> str:
    lines = [
        "# Pairwise vs Scalar Analysis",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- pairwise_run: {pairwise_path}",
        f"- control_run: {control_path}",
        f"- candidate_run: {candidate_path}",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Control Wins Despite Candidate Recall Gain", ""])
    for item in sorted(
        [entry for entry in analyses if entry.pairwise_winner == "control" and entry.recall_delta > 0],
        key=lambda entry: entry.recall_delta,
        reverse=True,
    ):
        lines.extend(
            [
                f"### {item.case_id}",
                f"- recall_delta: {item.recall_delta:.4f}",
                f"- control_recall: {item.control_required_claim_recall:.4f}",
                f"- candidate_recall: {item.candidate_required_claim_recall:.4f}",
                f"- control_length: {item.control_length}",
                f"- candidate_length: {item.candidate_length}",
                f"- candidate_length_ratio: {item.candidate_length_ratio:.2f}",
                f"- control_truncation_risk: {item.control_truncation_risk}",
                f"- candidate_truncation_risk: {item.candidate_truncation_risk}",
                f"- rationale: {item.rationale}",
                "",
            ]
        )

    lines.extend(["## Control Wins With Equal Recall", ""])
    for item in sorted(
        [entry for entry in analyses if entry.pairwise_winner == "control" and entry.recall_delta == 0],
        key=lambda entry: entry.candidate_length_ratio,
        reverse=True,
    ):
        lines.extend(
            [
                f"### {item.case_id}",
                f"- control_recall: {item.control_required_claim_recall:.4f}",
                f"- candidate_recall: {item.candidate_required_claim_recall:.4f}",
                f"- control_length: {item.control_length}",
                f"- candidate_length: {item.candidate_length}",
                f"- candidate_length_ratio: {item.candidate_length_ratio:.2f}",
                f"- control_truncation_risk: {item.control_truncation_risk}",
                f"- candidate_truncation_risk: {item.candidate_truncation_risk}",
                f"- rationale: {item.rationale}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pairwise-vs-scalar disagreements.")
    parser.add_argument("pairwise_run")
    parser.add_argument("control_run")
    parser.add_argument("candidate_run")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pairwise_path = (repo_root / args.pairwise_run).resolve() if not Path(args.pairwise_run).is_absolute() else Path(args.pairwise_run)
    control_path = (repo_root / args.control_run).resolve() if not Path(args.control_run).is_absolute() else Path(args.control_run)
    candidate_path = (repo_root / args.candidate_run).resolve() if not Path(args.candidate_run).is_absolute() else Path(args.candidate_run)

    pairwise_payload = _load_json(pairwise_path)
    control_payload = _load_json(control_path)
    candidate_payload = _load_json(candidate_path)
    analyses = _case_analyses(pairwise_payload, control_payload, candidate_payload)
    summary = _summary(analyses)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pairwise_run": str(pairwise_path),
        "control_run": str(control_path),
        "candidate_run": str(candidate_path),
        "summary": summary,
        "cases": [asdict(item) for item in analyses],
    }

    runs_dir = repo_root / "datasets" / "runs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    slug = _slugify(pairwise_path.stem)
    json_path = runs_dir / f"pairwise_scalar_analysis_{slug}_{timestamp}.json"
    md_path = runs_dir / f"pairwise_scalar_analysis_{slug}_{timestamp}.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md_path.write_text(
        _render_markdown(
            pairwise_path=pairwise_path,
            control_path=control_path,
            candidate_path=candidate_path,
            analyses=analyses,
            summary=summary,
        ),
        encoding="utf-8",
    )
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
