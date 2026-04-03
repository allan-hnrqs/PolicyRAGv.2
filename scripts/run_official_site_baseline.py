from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bgrag.config import Settings, detect_project_root
from bgrag.official_site_baseline import (
    run_official_site_baseline_eval,
    write_official_site_baseline_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded official-site live browsing baseline.")
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("datasets/eval/dev/parity19_dev.jsonl"),
        help="Eval JSONL path to run against.",
    )
    parser.add_argument(
        "--answer-profile",
        default="baseline_vector",
        help="Answer profile to reuse for the final answer strategy and answer model.",
    )
    parser.add_argument("--max-live-pages", type=int, default=6)
    parser.add_argument("--max-live-chunks", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit for smoke runs.")
    args = parser.parse_args()

    project_root = detect_project_root(Path.cwd())
    settings = Settings(project_root=project_root)
    settings.ensure_directories()
    run = run_official_site_baseline_eval(
        settings,
        eval_path=args.eval,
        answer_profile_name=args.answer_profile,
        max_live_pages=args.max_live_pages,
        max_live_chunks=args.max_live_chunks,
        case_limit=args.limit,
    )
    json_path, md_path = write_official_site_baseline_artifacts(settings, run)
    print(f"Wrote official-site baseline JSON to {json_path}")
    print(f"Wrote official-site baseline Markdown to {md_path}")
    print(run.overall_metrics)


if __name__ == "__main__":
    main()
