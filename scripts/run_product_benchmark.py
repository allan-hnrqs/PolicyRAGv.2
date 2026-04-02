from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgrag.demo_server import build_demo_settings
from bgrag.product_benchmark import run_product_benchmark, write_product_benchmark_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the product-serving benchmark against a profile.")
    parser.add_argument(
        "--profile",
        default="demo",
        help="Profile name to execute with run_demo_query (default: demo).",
    )
    parser.add_argument(
        "--manifest",
        default="datasets/eval/manifests/product_serving_benchmark_v1.json",
        help="Path to the product benchmark manifest.",
    )
    args = parser.parse_args()

    settings = build_demo_settings()
    manifest_path = settings.resolve(Path(args.manifest))
    run = run_product_benchmark(settings, manifest_path=manifest_path, profile_name=args.profile)
    json_path, md_path = write_product_benchmark_artifacts(settings, run)

    print(
        json.dumps(
            {
                "run_name": run.run_name,
                "profile_name": run.profile_name,
                "manifest_name": run.manifest_name,
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "summary": run.summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
