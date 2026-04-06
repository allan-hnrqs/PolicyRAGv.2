import json
from datetime import datetime, timezone
from pathlib import Path

from bgrag.benchmarks.optimization_cycle import (
    OptimizationCycleArtifactRef,
    OptimizationCycleResult,
    materialize_eval_surface_from_manifest,
    render_optimization_cycle_markdown,
)
from bgrag.config import Settings


def _write_eval_jsonl(path: Path, case_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in case_ids:
        rows.append(
            {
                "id": case_id,
                "question": f"Question for {case_id}",
                "persona": "tester",
                "primary_urls": [f"https://example.com/{case_id.lower()}"],
                "required_claims": [f"Claim for {case_id}"],
                "reference_answer": f"Reference answer for {case_id}",
                "claim_evidence": [],
                "tags": ["test"],
                "restricted_source_valid": True,
                "open_browse_valid": True,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_materialize_eval_surface_from_manifest_writes_split_outputs(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()

    dev_source = tmp_path / "datasets" / "eval" / "dev" / "source_dev.jsonl"
    holdout_source = tmp_path / "datasets" / "eval" / "holdout" / "source_holdout.jsonl"
    _write_eval_jsonl(dev_source, ["DEV_001", "DEV_002"])
    _write_eval_jsonl(holdout_source, ["HO_001", "HO_002"])

    manifest_path = tmp_path / "datasets" / "eval" / "manifests" / "surface.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "name": "surface",
                "purpose": "test",
                "source_eval_paths": {
                    "dev": "datasets/eval/dev/source_dev.jsonl",
                    "holdout": "datasets/eval/holdout/source_holdout.jsonl",
                },
                "target_eval_paths": {
                    "dev": "datasets/eval/generated/dev_slice.jsonl",
                    "holdout": "datasets/eval/generated/holdout_slice.jsonl",
                },
                "dev_case_ids": ["DEV_002"],
                "holdout_case_ids": ["HO_001"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    written = materialize_eval_surface_from_manifest(settings, manifest_path)

    assert set(written) == {"dev", "holdout"}
    dev_rows = [json.loads(line) for line in written["dev"].read_text(encoding="utf-8").splitlines() if line.strip()]
    holdout_rows = [json.loads(line) for line in written["holdout"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["id"] for row in dev_rows] == ["DEV_002"]
    assert [row["id"] for row in holdout_rows] == ["HO_001"]


def test_render_optimization_cycle_markdown_includes_core_fields() -> None:
    run = OptimizationCycleResult(
        run_name="optimization_cycle_loop01",
        created_at=datetime(2026, 4, 4, tzinfo=timezone.utc),
        cycle_id="loop01_small",
        cycle_kind="small",
        hypothesis_id="packing_audit",
        profile_name="baseline_vector_rerank_shortlist",
        control_profile_name="baseline_vector_rerank_shortlist",
        classification="mixed",
        failure_surface_manifest_path="datasets/eval/manifests/persistent_failure_surface_v1.json",
        materialized_surfaces={"dev": "datasets/eval/generated/persistent_failure_surface_dev_v1.jsonl"},
        benchmark_artifacts=[
            OptimizationCycleArtifactRef(
                label="failure_surface_eval_dev",
                artifact_path="datasets/runs/failure_eval.json",
                details_path="datasets/runs/manifests/failure_eval.manifest.json",
                summary={"required_claim_recall_mean": 0.5},
            )
        ],
        notes=["diagnostic cycle"],
    )

    markdown = render_optimization_cycle_markdown(run)

    assert "# Optimization Cycle: loop01_small" in markdown
    assert "- classification: mixed" in markdown
    assert "failure_surface_eval_dev" in markdown
    assert "required_claim_recall_mean: 0.5" in markdown
