from __future__ import annotations

import argparse
from pathlib import Path

from bgrag.benchmarks.answer_replay import run_answer_replay_benchmark, write_answer_replay_artifacts
from bgrag.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an answer-only replay benchmark from a prior eval artifact.")
    parser.add_argument("--source-run", required=True, help="Path to an existing eval run JSON artifact.")
    parser.add_argument("--strategy", required=True, help="Answer strategy to replay, e.g. inline_evidence_chat.")
    parser.add_argument("--answer-model", default=None, help="Optional Cohere chat model override.")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional case ID filter. Repeat the flag to replay only a failure slice.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    run = run_answer_replay_benchmark(
        settings,
        source_run_path=Path(args.source_run),
        strategy_name=args.strategy,
        answer_model=args.answer_model,
        case_filter_ids=args.case_id,
    )
    json_path, md_path = write_answer_replay_artifacts(settings, run)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
