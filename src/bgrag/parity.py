"""Parity helpers for freezing feat baselines and seeding new eval datasets."""

from __future__ import annotations

import json
from pathlib import Path

from bgrag.config import Settings


def copy_jsonl(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def freeze_feat_parity_inputs(settings: Settings, workspace_root: Path) -> dict[str, int]:
    cross_eval = workspace_root / "cross_eval"
    copied: dict[str, int] = {}
    mapping = {
        "parity19.jsonl": cross_eval / "human_realistic_buyers_guide_cases.jsonl",
        "holdout_feat.jsonl": cross_eval / "core_suite" / "feat_retrieval_human_realistic_frozen_holdout_cases.jsonl",
        "dev_feat.jsonl": cross_eval / "core_suite" / "feat_retrieval_human_realistic_dev_cases.jsonl",
    }
    output_dir = settings.resolve(Path("datasets/eval/parity"))
    for target_name, source_path in mapping.items():
        copied[target_name] = copy_jsonl(source_path, output_dir / target_name)
    manifest_path = settings.resolve(Path("datasets/eval/manifests/frozen_feat_parity_manifest.json"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(copied, indent=2), encoding="utf-8")
    return copied
