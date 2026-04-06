from __future__ import annotations

import argparse
from pathlib import Path

from bgrag.benchmarks.bundle_risk import run_bundle_risk_audit, write_bundle_risk_audit_artifacts
from bgrag.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit deterministic bundle-risk signals against an eval run.")
    parser.add_argument("--source-run", required=True, help="Path to an existing eval run JSON artifact.")
    parser.add_argument(
        "--low-recall-threshold",
        type=float,
        default=0.75,
        help="Threshold below which a case is treated as a material miss.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    run = run_bundle_risk_audit(
        settings,
        source_run_path=Path(args.source_run),
        low_recall_threshold=args.low_recall_threshold,
    )
    json_path, md_path = write_bundle_risk_audit_artifacts(settings, run)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
