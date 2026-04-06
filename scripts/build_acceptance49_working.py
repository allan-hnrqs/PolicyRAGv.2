from __future__ import annotations

import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_additions(manifests_dir: Path) -> list[dict[str, object]]:
    additions: list[dict[str, object]] = []
    for path in sorted(manifests_dir.glob("acceptance49_additions_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"{path} must contain a JSON list of cases")
        additions.extend(payload)
    return additions


def _write_jsonl(path: Path, cases: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifests_dir = repo_root / "datasets" / "eval" / "manifests"

    parity39_working_path = repo_root / "datasets" / "eval" / "parity" / "parity39_working.jsonl"
    parity39_dev_path = repo_root / "datasets" / "eval" / "dev" / "parity39_dev_draft.jsonl"
    parity39_holdout_path = repo_root / "datasets" / "eval" / "holdout" / "parity39_holdout_draft.jsonl"

    working_path = repo_root / "datasets" / "eval" / "parity" / "acceptance49_working.jsonl"
    dev_path = repo_root / "datasets" / "eval" / "dev" / "acceptance49_dev_draft.jsonl"
    holdout_path = repo_root / "datasets" / "eval" / "holdout" / "acceptance49_holdout_draft.jsonl"

    base_working = _load_jsonl(parity39_working_path)
    base_dev_ids = {case["id"] for case in _load_jsonl(parity39_dev_path)}
    base_holdout_ids = {case["id"] for case in _load_jsonl(parity39_holdout_path)}
    additions = _load_additions(manifests_dir)

    cases_by_id: dict[str, dict[str, object]] = {}
    for case in base_working + additions:
        case_id = str(case["id"])
        if case_id in cases_by_id:
            raise RuntimeError(f"Duplicate case ID detected: {case_id}")
        cases_by_id[case_id] = case

    working_cases = [cases_by_id[case["id"]] for case in base_working]
    working_cases.extend(sorted(additions, key=lambda case: str(case["id"])))

    dev_cases: list[dict[str, object]] = []
    holdout_cases: list[dict[str, object]] = []
    for case in working_cases:
        case_id = str(case["id"])
        split = case.get("split")
        if case_id in base_dev_ids or split == "dev":
            dev_cases.append(case)
        elif case_id in base_holdout_ids or split == "holdout":
            holdout_cases.append(case)
        else:
            raise RuntimeError(f"Case {case_id} has no draft split assignment")

    _write_jsonl(working_path, working_cases)
    _write_jsonl(dev_path, dev_cases)
    _write_jsonl(holdout_path, holdout_cases)

    print(working_path)
    print(dev_path)
    print(holdout_path)
    print(f"working_cases={len(working_cases)} dev_cases={len(dev_cases)} holdout_cases={len(holdout_cases)}")


if __name__ == "__main__":
    main()
