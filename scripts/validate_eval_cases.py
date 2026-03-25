from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import REPO_ROOT
from bgrag.eval.validation import load_and_validate_eval_cases


def validate_jsonl(path: Path) -> tuple[int, list[str], list[str]]:
    cases, issues = load_and_validate_eval_cases(path)
    errors = [issue.message for issue in issues if issue.severity == "error"]
    warnings = [issue.message for issue in issues if issue.severity == "warning"]
    return len(cases), errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/validate_eval_cases.py <jsonl-path> [<jsonl-path> ...]")
        return 2

    total_cases = 0
    total_errors: list[str] = []
    total_warnings: list[str] = []
    for raw_path in argv[1:]:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        count, errors, warnings = validate_jsonl(path)
        total_cases += count
        total_errors.extend(errors)
        total_warnings.extend(warnings)
        print(f"{path}: {count} valid case(s)")

    if total_errors:
        print("\nErrors:")
        for error in total_errors:
            print(error)
        return 1

    if total_warnings:
        print("\nWarnings:")
        for warning in total_warnings:
            print(warning)

    print(f"\nValidation passed: {total_cases} total case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
