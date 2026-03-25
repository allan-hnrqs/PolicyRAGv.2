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


def test_exactness_family_manifest_matches_assets() -> None:
    audit_path = REPO_ROOT / "datasets" / "eval" / "manifests" / "exactness_family_case_audit.json"
    authored_path = REPO_ROOT / "datasets" / "eval" / "manifests" / "exactness_family_authored_cases.json"
    slice_manifest_path = REPO_ROOT / "datasets" / "eval" / "manifests" / "exactness_family_slice_manifest.json"
    dev_slice_path = REPO_ROOT / "datasets" / "eval" / "dev" / "exactness_family_dev.jsonl"
    holdout_slice_path = REPO_ROOT / "datasets" / "eval" / "holdout" / "exactness_family_holdout.jsonl"

    audit = _load_json(audit_path)
    authored = _load_json(authored_path)
    slice_manifest = _load_json(slice_manifest_path)
    dev_rows = _load_jsonl(dev_slice_path)
    holdout_rows = _load_jsonl(holdout_slice_path)

    included = audit["included_cases"]
    assert isinstance(included, list)

    dev_ids = [str(row["id"]) for row in included if row["split"] == "dev"]
    holdout_ids = [str(row["id"]) for row in included if row["split"] == "holdout"]

    authored_cases = authored["cases"]
    assert isinstance(authored_cases, list)
    authored_ids = {str(row["id"]) for row in authored_cases}
    assert {"EX_001", "EX_002"}.issubset(authored_ids)

    assert slice_manifest["selection_manifest"] == "datasets/eval/manifests/exactness_family_case_audit.json"
    assert slice_manifest["dev_case_ids"] == dev_ids
    assert slice_manifest["holdout_case_ids"] == holdout_ids

    assert [str(row["id"]) for row in dev_rows] == dev_ids
    assert [str(row["id"]) for row in holdout_rows] == holdout_ids

    for row in dev_rows + holdout_rows:
        assert row.get("expect_abstain") is True
        assert row.get("forbidden_claims")


def test_hr_016_adjacent_form_guard_is_present_on_canonical_surfaces() -> None:
    target_fragment = (
        "The provided documents identify any PSPC or PWGSC-TPSGC form number as the required form for ADM "
        "approval of a reciprocal procurement exception."
    )
    surface_paths = [
        REPO_ROOT / "datasets" / "eval" / "parity" / "parity19.jsonl",
        REPO_ROOT / "datasets" / "eval" / "holdout" / "parity19_holdout.jsonl",
        REPO_ROOT / "datasets" / "eval" / "parity" / "parity39_working.jsonl",
        REPO_ROOT / "datasets" / "eval" / "holdout" / "parity39_holdout_draft.jsonl",
        REPO_ROOT / "datasets" / "eval" / "holdout" / "parity39_exactness_holdout.jsonl",
    ]

    for path in surface_paths:
        rows = _load_jsonl(path)
        hr_016 = next(row for row in rows if row["id"] == "HR_016")
        assert target_fragment in hr_016["forbidden_claims"]
