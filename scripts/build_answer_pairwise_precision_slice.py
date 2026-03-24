from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


SLICE_NAME = "answer_pairwise_precision_slice"
CASE_IDS = [
    "HR_001",
    "HR_003",
    "HR_004",
    "HR_005",
    "HR_007",
    "HR_008",
    "HR_009",
    "HR_011",
    "HR_013",
    "HR_015",
    "HR_017",
    "HR_018",
    "HR_019",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    repo_root = _repo_root()
    source_path = repo_root / "datasets" / "eval" / "parity" / "parity19.jsonl"
    output_path = repo_root / "datasets" / "eval" / "generated" / f"{SLICE_NAME}.jsonl"
    manifest_path = (
        repo_root / "datasets" / "eval" / "manifests" / f"{SLICE_NAME}_manifest.json"
    )

    source_cases = _load_jsonl(source_path)
    case_map = {str(case["id"]): case for case in source_cases}
    missing_ids = [case_id for case_id in CASE_IDS if case_id not in case_map]
    if missing_ids:
        raise SystemExit(f"Missing case ids in parity19 source: {', '.join(missing_ids)}")

    selected_cases = [case_map[case_id] for case_id in CASE_IDS]
    _write_jsonl(output_path, selected_cases)

    manifest = {
        "name": SLICE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_eval_path": "datasets/eval/parity/parity19.jsonl",
        "output_eval_path": f"datasets/eval/generated/{SLICE_NAME}.jsonl",
        "purpose": (
            "Canonical answer-side precision gate built from pairwise disagreement cases "
            "where scalar recall was equal or misleading but pairwise judging still "
            "preferred the baseline for faithfulness/directness."
        ),
        "selection_criteria": [
            "Start from the original 19-case parity control surface.",
            (
                "Include equal-recall or otherwise precision-sensitive cases from pairwise "
                "control-win analyses of answer-side branches."
            ),
            (
                "Keep regression guards for cases where the candidate briefly looked better "
                "by scalar recall but still needed strong precision discipline."
            ),
        ],
        "source_artifacts": [
            "datasets/runs/pairwise_baseline_20260324_020025_vs_selective_mode_aware_answer_repair_20260324_021259_20260324_021545.json",
            "datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_020025_vs_selective_mode_aware_answer_repair_20260324_021259_20260324_021545_20260324_021606_633997.json",
            "datasets/runs/pairwise_baseline_20260324_020025_vs_selective_mode_aware_compact_answering_20260324_032955_20260324_033148.json",
            "datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_020025_vs_selective_mode_aware_compact_answering_20260324_032955_20260324_033148_20260324_035234_825529.md",
        ],
        "case_ids": CASE_IDS,
        "notes": [
            (
                "HR_001 and HR_005 are carried as regression guards because answer-side "
                "branches can look stronger by scalar recall while still drifting on "
                "faithfulness or directness."
            ),
            (
                "The slice is intentionally precision-heavy and should complement, not "
                "replace, the canonical parity19 dev/holdout and rebuilt parity39 surfaces."
            ),
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"output_path": str(output_path), "manifest_path": str(manifest_path), "case_count": len(selected_cases)}, indent=2))


if __name__ == "__main__":
    main()
