from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bgrag.config import Settings, detect_project_root
from bgrag.benchmarks.retrieval_budget_sweep import (
    run_retrieval_budget_sweep,
    write_retrieval_budget_sweep_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded retrieval-budget sweep over a base profile.")
    parser.add_argument("--base-profile", default="baseline_vector_rerank_shortlist")
    parser.add_argument("--eval", type=Path, default=Path("datasets/eval/parity/parity19.jsonl"))
    parser.add_argument("--query-mode", choices=("single", "profile"), default="profile")
    parser.add_argument("--index-namespace", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = detect_project_root(Path.cwd())
    settings = Settings(project_root=project_root)
    settings.ensure_directories()
    run = run_retrieval_budget_sweep(
        settings,
        base_profile_name=args.base_profile,
        eval_path=args.eval,
        query_mode=args.query_mode,
        index_namespace=args.index_namespace,
    )
    json_path, md_path = write_retrieval_budget_sweep_artifacts(settings, run)
    print(json_path)
    print(md_path)
    print(run.summary)


if __name__ == "__main__":
    main()
