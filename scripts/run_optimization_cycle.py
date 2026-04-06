from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgrag.benchmarks.optimization_cycle import (
    CycleClassification,
    ExternalComparatorRef,
    run_optimization_cycle,
    write_optimization_cycle_artifacts,
)
from bgrag.config import Settings


def _parse_external_ref(raw: str) -> ExternalComparatorRef:
    if "=" not in raw:
        raise ValueError(f"External comparator refs must use label=path format, got: {raw}")
    label, artifact_path = raw.split("=", 1)
    label = label.strip()
    artifact_path = artifact_path.strip()
    if not label or not artifact_path:
        raise ValueError(f"External comparator refs must use non-empty label=path format, got: {raw}")
    return ExternalComparatorRef(label=label, artifact_path=artifact_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a persistent optimization-loop cycle.")
    parser.add_argument("--cycle-id", required=True, help="Stable cycle identifier, e.g. loop01_small.")
    parser.add_argument("--hypothesis-id", required=True, help="Single owner hypothesis for this cycle.")
    parser.add_argument("--profile", required=True, help="Profile to benchmark in this cycle.")
    parser.add_argument("--cycle-kind", choices=("large", "small"), required=True)
    parser.add_argument(
        "--classification",
        choices=("pending", "promising", "mixed", "rejected"),
        default="pending",
        help="Current classification for this cycle.",
    )
    parser.add_argument("--control-profile", default="baseline_vector_rerank_shortlist")
    parser.add_argument("--dev-eval", default="datasets/eval/dev/parity19_dev.jsonl")
    parser.add_argument("--holdout-eval", default="datasets/eval/holdout/parity19_holdout.jsonl")
    parser.add_argument(
        "--failure-surface-manifest",
        default="datasets/eval/manifests/persistent_failure_surface_v1.json",
        help="Optional failure-surface manifest to materialize and benchmark.",
    )
    parser.add_argument("--product-manifest", default="datasets/eval/manifests/product_serving_benchmark_v1.json")
    parser.add_argument("--multiturn-manifest", default="datasets/eval/manifests/multiturn_benchmark_v1.json")
    parser.add_argument(
        "--external-ref",
        action="append",
        default=[],
        help="Optional external comparator reference in label=path format.",
    )
    parser.add_argument("--note", action="append", default=[], help="Optional note to store on the cycle artifact.")
    parser.add_argument("--run-canonical", action="store_true", help="Force canonical dev/holdout eval and retrieval runs.")
    parser.add_argument("--skip-canonical", action="store_true", help="Skip canonical dev/holdout runs.")
    parser.add_argument("--skip-failure-surfaces", action="store_true", help="Skip failure-surface benchmarking.")
    parser.add_argument("--run-product", action="store_true", help="Run the product benchmark.")
    parser.add_argument("--skip-product", action="store_true", help="Skip the product benchmark.")
    parser.add_argument("--run-multiturn", action="store_true", help="Run the multi-turn benchmark.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()

    run_canonical: bool | None
    if args.run_canonical and args.skip_canonical:
        raise ValueError("Cannot set both --run-canonical and --skip-canonical.")
    if args.run_canonical:
        run_canonical = True
    elif args.skip_canonical:
        run_canonical = False
    else:
        run_canonical = None

    run_product: bool | None
    if args.run_product and args.skip_product:
        raise ValueError("Cannot set both --run-product and --skip-product.")
    if args.run_product:
        run_product = True
    elif args.skip_product:
        run_product = False
    else:
        run_product = None

    external_refs = [_parse_external_ref(raw) for raw in args.external_ref]
    failure_manifest = Path(args.failure_surface_manifest) if args.failure_surface_manifest else None

    run = run_optimization_cycle(
        settings,
        cycle_id=args.cycle_id,
        hypothesis_id=args.hypothesis_id,
        cycle_kind=args.cycle_kind,
        profile_name=args.profile,
        control_profile_name=args.control_profile,
        classification=args.classification,  # type: ignore[arg-type]
        dev_eval_path=Path(args.dev_eval),
        holdout_eval_path=Path(args.holdout_eval),
        failure_surface_manifest_path=failure_manifest,
        product_manifest_path=Path(args.product_manifest),
        multiturn_manifest_path=Path(args.multiturn_manifest),
        run_canonical=run_canonical,
        run_failure_surfaces=not args.skip_failure_surfaces,
        run_product=run_product,
        run_multiturn=args.run_multiturn,
        external_comparators=external_refs,
        notes=args.note,
    )
    json_path, md_path = write_optimization_cycle_artifacts(settings, run)
    print(
        json.dumps(
            {
                "run_name": run.run_name,
                "cycle_id": run.cycle_id,
                "cycle_kind": run.cycle_kind,
                "profile_name": run.profile_name,
                "classification": run.classification,
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "artifact_count": len(run.benchmark_artifacts),
                "materialized_surfaces": run.materialized_surfaces,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
