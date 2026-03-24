from pathlib import Path

from bgrag.config import Settings
from bgrag.parity import freeze_feat_parity_inputs


def _write_jsonl(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f'{{"id":"case_{index:03d}"}}' for index in range(count)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_freeze_feat_parity_inputs(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    workspace_root = tmp_path / "workspace"
    cross_eval = workspace_root / "cross_eval"
    _write_jsonl(cross_eval / "human_realistic_buyers_guide_cases.jsonl", 19)
    _write_jsonl(
        cross_eval / "core_suite" / "feat_retrieval_human_realistic_frozen_holdout_cases.jsonl",
        10,
    )
    _write_jsonl(
        cross_eval / "core_suite" / "feat_retrieval_human_realistic_dev_cases.jsonl",
        9,
    )
    copied = freeze_feat_parity_inputs(settings, workspace_root)
    assert copied["parity19.jsonl"] == 19
    assert copied["holdout_feat.jsonl"] == 10
    assert copied["dev_feat.jsonl"] == 9
    assert (tmp_path / "datasets/eval/manifests/frozen_feat_parity_manifest.json").exists()
