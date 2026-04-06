from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgrag.config import Settings
from bgrag.benchmarks.heuristic_trigger_audit import (
    run_heuristic_trigger_audit,
    write_heuristic_trigger_audit_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit wording-triggered runtime heuristics against paraphrase pairs.")
    parser.add_argument(
        "--manifest",
        default="datasets/eval/manifests/heuristic_trigger_robustness_v1.json",
        help="Path to the paraphrase-pair manifest.",
    )
    args = parser.parse_args()

    settings = Settings()
    run = run_heuristic_trigger_audit(settings, manifest_path=Path(args.manifest))
    json_path, md_path = write_heuristic_trigger_audit_artifacts(settings, run)
    print(
        json.dumps(
            {
                "run_name": run.run_name,
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "case_count": run.case_count,
                "authority_support_changed_count": run.authority_support_changed_count,
                "question_risk_level_changed_count": run.question_risk_level_changed_count,
                "exactness_changed_count": run.exactness_changed_count,
                "branch_changed_count": run.branch_changed_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
