from __future__ import annotations

import json
import sys
from pathlib import Path

from bgrag.types import EvalCase


def validate_jsonl(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
            continue
        try:
            EvalCase.model_validate(payload)
        except Exception as exc:  # ValidationError is fine, but keep dependency surface simple here.
            errors.append(f"{path}:{line_number}: schema error: {exc}")
            continue
        count += 1
    return count, errors


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/validate_eval_cases.py <jsonl-path> [<jsonl-path> ...]")
        return 2

    total_cases = 0
    total_errors: list[str] = []
    for raw_path in argv[1:]:
        path = Path(raw_path)
        count, errors = validate_jsonl(path)
        total_cases += count
        total_errors.extend(errors)
        print(f"{path}: {count} valid case(s)")

    if total_errors:
        print("\nErrors:")
        for error in total_errors:
            print(error)
        return 1

    print(f"\nValidation passed: {total_cases} total case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
