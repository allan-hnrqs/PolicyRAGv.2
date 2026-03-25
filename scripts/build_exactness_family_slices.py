from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import REPO_ROOT


AUDIT_MANIFEST_PATH = REPO_ROOT / "datasets" / "eval" / "manifests" / "exactness_family_case_audit.json"
AUTHORED_CASES_PATH = REPO_ROOT / "datasets" / "eval" / "manifests" / "exactness_family_authored_cases.json"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows[str(payload["id"])] = payload
    return rows


def _ids_for_split(audit_manifest: dict[str, object], split: str) -> list[str]:
    included = audit_manifest.get("included_cases", [])
    if not isinstance(included, list):
        raise TypeError("Audit manifest `included_cases` must be a list.")

    ids: list[str] = []
    for row in included:
        if not isinstance(row, dict):
            raise TypeError("Audit manifest entries must be objects.")
        if row.get("split") != split:
            continue
        case_id = row.get("id")
        if not isinstance(case_id, str):
            raise TypeError("Audit manifest case ids must be strings.")
        ids.append(case_id)
    return ids


def _load_authored_cases(path: Path) -> dict[str, dict[str, object]]:
    payload = _load_json(path)
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise TypeError("Authored cases manifest `cases` must be a list.")

    authored: dict[str, dict[str, object]] = {}
    for row in cases:
        if not isinstance(row, dict):
            raise TypeError("Authored case entries must be objects.")
        case_id = row.get("id")
        if not isinstance(case_id, str):
            raise TypeError("Authored case ids must be strings.")
        authored[case_id] = row
    return authored


def _select_cases(
    ids: list[str],
    split: str,
    base_source: dict[str, dict[str, object]],
    authored_source: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for case_id in ids:
        if case_id in base_source:
            row = dict(base_source[case_id])
        elif case_id in authored_source:
            row = dict(authored_source[case_id])
        else:
            raise KeyError(f"Case `{case_id}` not found in base or authored exactness sources.")
        row["split"] = split
        selected.append(row)
    return selected


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    audit_manifest = _load_json(AUDIT_MANIFEST_PATH)
    authored_cases = _load_authored_cases(AUTHORED_CASES_PATH)
    base_dev = _load_jsonl(REPO_ROOT / "datasets" / "eval" / "dev" / "parity39_exactness_dev.jsonl")
    base_holdout = _load_jsonl(REPO_ROOT / "datasets" / "eval" / "holdout" / "parity39_exactness_holdout.jsonl")

    dev_ids = _ids_for_split(audit_manifest, "dev")
    holdout_ids = _ids_for_split(audit_manifest, "holdout")

    dev_rows = _select_cases(dev_ids, "dev", base_dev, authored_cases)
    holdout_rows = _select_cases(holdout_ids, "holdout", base_holdout, authored_cases)

    dev_target = REPO_ROOT / "datasets" / "eval" / "dev" / "exactness_family_dev.jsonl"
    holdout_target = REPO_ROOT / "datasets" / "eval" / "holdout" / "exactness_family_holdout.jsonl"
    manifest_target = REPO_ROOT / "datasets" / "eval" / "manifests" / "exactness_family_slice_manifest.json"

    _write_jsonl(dev_target, dev_rows)
    _write_jsonl(holdout_target, holdout_rows)
    manifest_target.write_text(
        json.dumps(
            {
                "name": "exactness_family_slices",
                "created_from": {
                    "base_dev": "datasets/eval/dev/parity39_exactness_dev.jsonl",
                    "base_holdout": "datasets/eval/holdout/parity39_exactness_holdout.jsonl",
                    "authored_cases": "datasets/eval/manifests/exactness_family_authored_cases.json",
                },
                "selection_manifest": "datasets/eval/manifests/exactness_family_case_audit.json",
                "rationale": (
                    "Expanded exactness-family surface that preserves the split-safe rebuilt-parity39 abstention "
                    "cases and adds newly authored negative exactness cases for missing contact and internal "
                    "artifact details."
                ),
                "dev_case_ids": dev_ids,
                "holdout_case_ids": holdout_ids,
                "targets": {
                    "dev": "datasets/eval/dev/exactness_family_dev.jsonl",
                    "holdout": "datasets/eval/holdout/exactness_family_holdout.jsonl",
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
