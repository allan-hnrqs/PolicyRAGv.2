import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_exactness_slice_manifest_matches_audit() -> None:
    audit_path = REPO_ROOT / "datasets" / "eval" / "manifests" / "parity39_exactness_case_audit.json"
    slice_manifest_path = REPO_ROOT / "datasets" / "eval" / "manifests" / "parity39_exactness_slice_manifest.json"
    dev_slice_path = REPO_ROOT / "datasets" / "eval" / "dev" / "parity39_exactness_dev.jsonl"
    holdout_slice_path = REPO_ROOT / "datasets" / "eval" / "holdout" / "parity39_exactness_holdout.jsonl"
    source_path = REPO_ROOT / "datasets" / "eval" / "parity" / "parity39_working.jsonl"

    audit = _load_json(audit_path)
    slice_manifest = _load_json(slice_manifest_path)
    dev_rows = _load_jsonl(dev_slice_path)
    holdout_rows = _load_jsonl(holdout_slice_path)
    source_rows = _load_jsonl(source_path)
    source_by_id = {str(row["id"]): row for row in source_rows}

    included = audit["included_cases"]
    assert isinstance(included, list)
    excluded = audit["excluded_adjacent_cases"]
    assert isinstance(excluded, list)

    dev_ids = [str(row["id"]) for row in included if row["split"] == "dev"]
    holdout_ids = [str(row["id"]) for row in included if row["split"] == "holdout"]
    excluded_ids = [str(row["id"]) for row in excluded]

    assert slice_manifest["selection_manifest"] == "datasets/eval/manifests/parity39_exactness_case_audit.json"
    assert slice_manifest["dev_case_ids"] == dev_ids
    assert slice_manifest["holdout_case_ids"] == holdout_ids
    assert slice_manifest["excluded_adjacent_case_ids"] == excluded_ids

    assert [str(row["id"]) for row in dev_rows] == dev_ids
    assert [str(row["id"]) for row in holdout_rows] == holdout_ids
    assert set(dev_ids).isdisjoint(excluded_ids)
    assert set(holdout_ids).isdisjoint(excluded_ids)

    for case_id in dev_ids + holdout_ids:
        row = source_by_id[case_id]
        assert row.get("expect_abstain") is True
        assert row.get("forbidden_claims")

