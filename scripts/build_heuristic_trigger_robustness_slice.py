from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import REPO_ROOT


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    manifest_path = REPO_ROOT / "datasets" / "eval" / "manifests" / "heuristic_trigger_robustness_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    indexed_cases: dict[str, dict[str, object]] = {}
    for rel_path in manifest["source_eval_paths"]:
        path = REPO_ROOT / rel_path
        for case in _load_jsonl(path):
            indexed_cases[str(case["id"])] = case

    output_rows: list[dict[str, object]] = []
    for pair in manifest["pairs"]:
        source_case_id = str(pair["source_case_id"])
        source_case = indexed_cases.get(source_case_id)
        if source_case is None:
            raise RuntimeError(f"Missing source case: {source_case_id}")
        paraphrase_case = dict(source_case)
        paraphrase_case["id"] = str(pair["paraphrase_case_id"])
        paraphrase_case["question"] = str(pair["paraphrase_question"])
        paraphrase_case["split"] = "generated"
        existing_notes = paraphrase_case.get("notes")
        if isinstance(existing_notes, str) and existing_notes.strip():
            paraphrase_case["notes"] = f"{existing_notes} | paraphrase_of:{source_case_id}"
        else:
            paraphrase_case["notes"] = f"paraphrase_of:{source_case_id}"
        output_rows.append(paraphrase_case)

    output_path = REPO_ROOT / manifest["output_eval_path"]
    _write_jsonl(output_path, output_rows)
    print(output_path)
    print(f"cases={len(output_rows)}")


if __name__ == "__main__":
    main()
