from __future__ import annotations

import json
from pathlib import Path


DEV_IDS = {
    "HR_001",
    "HR_003",
    "HR_005",
    "HR_007",
    "HR_009",
    "HR_010",
    "HR_013",
    "HR_015",
    "HR_017",
}

HOLDOUT_IDS = {
    "HR_002",
    "HR_004",
    "HR_006",
    "HR_008",
    "HR_011",
    "HR_012",
    "HR_014",
    "HR_016",
    "HR_018",
    "HR_019",
}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parity_path = repo_root / "datasets" / "eval" / "parity" / "parity19.jsonl"
    dev_path = repo_root / "datasets" / "eval" / "dev" / "parity19_dev.jsonl"
    holdout_path = repo_root / "datasets" / "eval" / "holdout" / "parity19_holdout.jsonl"

    cases = [json.loads(line) for line in parity_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_cases = [case for case in cases if case["id"] in DEV_IDS]
    holdout_cases = [case for case in cases if case["id"] in HOLDOUT_IDS]

    if len(dev_cases) != len(DEV_IDS):
        missing = sorted(DEV_IDS - {case["id"] for case in dev_cases})
        raise RuntimeError(f"Missing dev IDs in parity19.jsonl: {missing}")
    if len(holdout_cases) != len(HOLDOUT_IDS):
        missing = sorted(HOLDOUT_IDS - {case["id"] for case in holdout_cases})
        raise RuntimeError(f"Missing holdout IDs in parity19.jsonl: {missing}")

    dev_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in dev_cases) + "\n", encoding="utf-8")
    holdout_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in holdout_cases) + "\n",
        encoding="utf-8",
    )

    print(dev_path)
    print(holdout_path)


if __name__ == "__main__":
    main()
