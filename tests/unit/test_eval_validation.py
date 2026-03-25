from pathlib import Path

from bgrag.eval.validation import infer_eval_split, load_and_validate_eval_cases


def test_infer_eval_split_from_parent_folder() -> None:
    path = Path("datasets/eval/holdout/parity19_holdout.jsonl")
    assert infer_eval_split(path) == "holdout"


def test_load_and_validate_eval_cases_reports_warnings_and_backfills_split(tmp_path: Path) -> None:
    eval_dir = tmp_path / "generated"
    eval_dir.mkdir()
    eval_path = eval_dir / "sample.jsonl"
    eval_path.write_text(
        (
            '{"id":"T1","question":"Q?","primary_urls":["https://example.com/a"],'
            '"required_claims":["A","B"],"reference_answer":"Ref.",'
            '"claim_evidence":[{"claim":"A","evidence_doc_urls":["https://example.com/a"]}]}\n'
        ),
        encoding="utf-8",
    )

    cases, issues = load_and_validate_eval_cases(eval_path)

    assert len(cases) == 1
    assert cases[0].split == "generated"
    assert any(issue.severity == "warning" and "claim_evidence item(s)" in issue.message for issue in issues)
