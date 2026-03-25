from pathlib import Path

import pytest

from bgrag.eval.loader import load_eval_cases
from bgrag.types import EvalCase


def test_eval_case_minimal_schema() -> None:
    case = EvalCase.model_validate(
        {
            "id": "T1",
            "question": "Q?",
            "primary_urls": [],
            "supporting_urls": [],
        }
    )
    assert case.id == "T1"


def test_load_eval_cases_infers_split_from_parent_folder(tmp_path: Path) -> None:
    eval_dir = tmp_path / "dev"
    eval_dir.mkdir()
    eval_path = eval_dir / "sample.jsonl"
    eval_path.write_text('{"id":"T1","question":"Q?","primary_urls":["https://example.com/a"],"required_claims":["A"],"reference_answer":"A."}\n', encoding="utf-8")

    cases = load_eval_cases(eval_path)

    assert len(cases) == 1
    assert cases[0].split == "dev"


def test_load_eval_cases_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    eval_path = tmp_path / "cases.jsonl"
    eval_path.write_text(
        '\n'.join(
            [
                '{"id":"T1","question":"Q1?","primary_urls":["https://example.com/a"],"required_claims":["A"],"reference_answer":"A."}',
                '{"id":"T1","question":"Q2?","primary_urls":["https://example.com/b"],"required_claims":["B"],"reference_answer":"B."}',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case id"):
        load_eval_cases(eval_path)
