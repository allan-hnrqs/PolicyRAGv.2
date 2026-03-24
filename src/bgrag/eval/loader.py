"""Evaluation suite loading."""

from __future__ import annotations

import json
from pathlib import Path

from bgrag.types import EvalCase


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate(json.loads(line)))
    return cases
