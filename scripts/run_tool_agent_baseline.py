from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bgrag.config import Settings, detect_project_root
from bgrag.tool_agent_baseline import run_tool_agent_baseline_eval, write_tool_agent_baseline_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded tool-using official-site agent baseline.")
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("datasets/eval/dev/parity19_dev.jsonl"),
        help="Eval JSONL path to run against.",
    )
    parser.add_argument(
        "--answer-profile",
        default="baseline_vector",
        help="Answer profile to reuse for final answering.",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-live-pages", type=int, default=6)
    parser.add_argument("--max-live-chunks", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit for smoke runs.")
    args = parser.parse_args()

    project_root = detect_project_root(Path.cwd())
    settings = Settings(project_root=project_root)
    settings.ensure_directories()
    run = run_tool_agent_baseline_eval(
        settings,
        eval_path=args.eval,
        answer_profile_name=args.answer_profile,
        max_steps=args.max_steps,
        max_live_pages=args.max_live_pages,
        max_live_chunks=args.max_live_chunks,
        case_limit=args.limit,
    )
    json_path, md_path = write_tool_agent_baseline_artifacts(settings, run)
    print(f"Wrote tool-agent baseline JSON to {json_path}")
    print(f"Wrote tool-agent baseline Markdown to {md_path}")
    print(run.overall_metrics)


if __name__ == "__main__":
    main()
