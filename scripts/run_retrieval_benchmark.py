from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgrag.config import Settings
from bgrag.retrieval_benchmark import run_retrieval_benchmark, write_retrieval_benchmark_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a retrieval-only benchmark against an eval surface.")
    parser.add_argument("--profile", default="baseline", help="Profile name to use for retrieval settings.")
    parser.add_argument("--eval", required=True, help="Path to the eval jsonl file.")
    parser.add_argument(
        "--query-mode",
        choices=("single", "profile"),
        default="single",
        help="Use only the original question or the profile's query-expansion behavior.",
    )
    args = parser.parse_args()

    settings = Settings()
    eval_path = settings.resolve(Path(args.eval))
    run = run_retrieval_benchmark(
        settings,
        eval_path=eval_path,
        profile_name=args.profile,
        query_mode=args.query_mode,
    )
    json_path, md_path = write_retrieval_benchmark_artifacts(settings, run)
    print(
        json.dumps(
            {
                "run_name": run.run_name,
                "profile_name": run.profile_name,
                "query_mode": run.query_mode,
                "eval_path": str(eval_path),
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "overall_metrics": run.overall_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
