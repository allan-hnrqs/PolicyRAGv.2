import json
from pathlib import Path

from bgrag.benchmarks.heuristic_trigger_audit import run_heuristic_trigger_audit
from bgrag.config import Settings


def test_heuristic_trigger_audit_detects_changed_signals(tmp_path: Path) -> None:
    source_eval = tmp_path / "source.jsonl"
    source_eval.write_text(
        (
            json.dumps(
                {
                    "id": "HR_999",
                    "question": "The Buy Canadian page mentions a policy framework. Who owns it?",
                    "primary_urls": ["https://example.com/a"],
                    "required_claims": ["A"],
                    "reference_answer": "Ref.",
                }
            )
            + "\n"
        ),
        encoding="utf-8",
    )
    generated_eval = tmp_path / "generated.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_eval_paths": [str(source_eval)],
                "output_eval_path": str(generated_eval),
                "pairs": [
                    {
                        "source_case_id": "HR_999",
                        "paraphrase_case_id": "HR_999P1",
                        "paraphrase_question": "That rollout page refers to a framework. Which organization maintains it?",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = Settings(repo_root=tmp_path)
    run = run_heuristic_trigger_audit(settings, manifest_path=manifest_path)

    assert run.case_count == 1
    assert run.authority_support_changed_count == 1
    assert "authority_support" in run.cases[0].changed_signals
