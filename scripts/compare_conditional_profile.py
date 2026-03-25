from __future__ import annotations

import argparse
from datetime import datetime, timezone

from _bootstrap import REPO_ROOT
from bgrag.cli import _repo_env_values
from bgrag.config import Settings
from bgrag.eval.conditional_compare import (
    resolve_cli_path,
    run_conditional_compare,
)

def _emit_progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    print(f"[{stamp}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a conditional candidate profile against a control profile using an intervention-only composite."
    )
    parser.add_argument("eval_path", help="Path to eval jsonl relative to repo root or absolute")
    parser.add_argument("--control-profile", default="baseline")
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--index-namespace", default=None, help="Explicit index namespace")
    parser.add_argument(
        "--intervention-path",
        action="append",
        default=None,
        help=(
            "Raw selected_path values that count as true candidate interventions, "
            "for example 'rewrite_structured_contract'. Repeatable."
        ),
    )
    parser.add_argument("--pairwise", action="store_true", help="Run OpenAI pairwise judging on control vs composite.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    settings = Settings(project_root=repo_root, **_repo_env_values(repo_root))
    settings.ensure_directories()
    eval_path = resolve_cli_path(repo_root, args.eval_path)
    intervention_paths = set(args.intervention_path or ["rewrite_structured_contract"])
    _emit_progress(
        "starting_conditional_compare "
        f"eval_path={eval_path} control_profile={args.control_profile} "
        f"candidate_profile={args.candidate_profile} intervention_paths={sorted(intervention_paths)}"
    )
    artifacts = run_conditional_compare(
        settings=settings,
        eval_path=eval_path,
        control_profile=args.control_profile,
        candidate_profile=args.candidate_profile,
        index_namespace=args.index_namespace,
        intervention_paths=intervention_paths,
        include_pairwise=args.pairwise,
        progress=_emit_progress,
    )

    _emit_progress("conditional_compare_complete")
    print(artifacts.control.path)
    print(artifacts.candidate.path)
    print(artifacts.composite.json_path)
    print(artifacts.composite.markdown_path)
    if artifacts.pairwise is not None:
        print(artifacts.pairwise.path)
    print(artifacts.summary_json_path)
    print(artifacts.summary_markdown_path)


if __name__ == "__main__":
    main()
