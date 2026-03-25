"""Evaluation suite loading."""

from __future__ import annotations

from pathlib import Path

from bgrag.eval.validation import load_and_validate_eval_cases
from bgrag.types import EvalCase


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases, issues = load_and_validate_eval_cases(path)
    errors = [issue.message for issue in issues if issue.severity == "error"]
    if errors:
        joined = "\n".join(errors)
        raise ValueError(f"Eval case validation failed for {path}:\n{joined}")
    return cases
