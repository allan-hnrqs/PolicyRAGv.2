from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import REPO_ROOT

DEV_IDS = ["HR_017", "HR_038"]
HOLDOUT_IDS = ["HR_016", "HR_037"]


def _load_cases(path: Path) -> dict[str, dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases[str(row["id"])] = row
    return cases


def _select_cases(source: dict[str, dict[str, object]], ids: list[str], split: str) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for case_id in ids:
        if case_id not in source:
            raise KeyError(f"Case `{case_id}` not found in source draft.")
        row = dict(source[case_id])
        row["split"] = split
        selected.append(row)
    return selected


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    repo_root = REPO_ROOT
    dev_source_path = repo_root / "datasets" / "eval" / "dev" / "parity39_dev_draft.jsonl"
    holdout_source_path = repo_root / "datasets" / "eval" / "holdout" / "parity39_holdout_draft.jsonl"
    dev_source = _load_cases(dev_source_path)
    holdout_source = _load_cases(holdout_source_path)

    dev_rows = _select_cases(dev_source, DEV_IDS, "dev")
    holdout_rows = _select_cases(holdout_source, HOLDOUT_IDS, "holdout")

    dev_target = repo_root / "datasets" / "eval" / "dev" / "parity39_exactness_dev.jsonl"
    holdout_target = repo_root / "datasets" / "eval" / "holdout" / "parity39_exactness_holdout.jsonl"
    manifest_target = repo_root / "datasets" / "eval" / "manifests" / "parity39_exactness_slice_manifest.json"

    _write_jsonl(dev_target, dev_rows)
    _write_jsonl(holdout_target, holdout_rows)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(
        json.dumps(
            {
                "name": "parity39_exactness_slices",
                "created_from": {
                    "dev_source": "datasets/eval/dev/parity39_dev_draft.jsonl",
                    "holdout_source": "datasets/eval/holdout/parity39_holdout_draft.jsonl",
                },
                "rationale": (
                    "Split-safe rebuilt-parity39 exactness/abstention subset for preserved-baseline "
                    "evaluation of unsupported exact-detail and contact/internal-detail failures."
                ),
                "dev_case_ids": DEV_IDS,
                "holdout_case_ids": HOLDOUT_IDS,
                "targets": {
                    "dev": "datasets/eval/dev/parity39_exactness_dev.jsonl",
                    "holdout": "datasets/eval/holdout/parity39_exactness_holdout.jsonl",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(dev_target)
    print(holdout_target)
    print(manifest_target)


if __name__ == "__main__":
    main()
