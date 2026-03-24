import json
from pathlib import Path

from bgrag.config import Settings
from bgrag.indexing.elastic import chunk_index_name
from bgrag.manifests import (
    build_pairwise_run_manifest,
    build_run_name,
    derive_index_namespace,
    get_active_index_namespace,
    index_embeddings_path,
    run_manifest_path,
    set_active_index_namespace,
    workspace_fingerprint,
    write_run_artifact_manifest,
    write_index_manifest,
)


def test_derive_index_namespace_is_stable_for_same_inputs(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "datasets" / "corpus").mkdir(parents=True)
    (project_root / "profiles").mkdir(parents=True)
    (project_root / "src" / "bgrag").mkdir(parents=True)
    (project_root / "datasets" / "corpus" / "chunks.jsonl").write_text("one\n", encoding="utf-8")
    (project_root / "profiles" / "baseline.yaml").write_text("name: baseline\n", encoding="utf-8")
    (project_root / "src" / "bgrag" / "__init__.py").write_text("", encoding="utf-8")
    settings = Settings(project_root=project_root, cohere_embed_model="embed-english-v3.0")

    first = derive_index_namespace(settings, "baseline")
    second = derive_index_namespace(settings, "baseline")

    assert first == second
    assert first.startswith("baseline_embed_english_v3_0_")


def test_index_pointer_round_trip(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)

    set_active_index_namespace(settings, "baseline_ns")

    assert get_active_index_namespace(settings) == "baseline_ns"


def test_index_manifest_written_under_namespace_dir(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)

    write_index_manifest(settings, "ns1", {"namespace": "ns1", "chunk_count": 3})

    manifest_path = tmp_path / "datasets" / "index" / "ns1" / "index_manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["chunk_count"] == 3
    assert index_embeddings_path(settings, "ns1") == tmp_path / "datasets" / "index" / "ns1" / "chunk_embeddings.json"


def test_chunk_index_name_includes_namespace() -> None:
    assert chunk_index_name("buyers_guide", "baseline_ns") == "bgrag_chunks_baseline_ns_buyers_guide"


def test_workspace_fingerprint_is_stable_per_repo(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)

    first = workspace_fingerprint(settings)
    second = workspace_fingerprint(settings)

    assert first == second
    assert len(first) == 8


def test_build_run_name_is_collision_resistant() -> None:
    first = build_run_name("baseline")
    second = build_run_name("baseline")

    assert first != second
    assert first.startswith("baseline_")
    assert second.startswith("baseline_")


def test_pairwise_run_manifest_and_sidecar_are_written(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, openai_api_key="test-key")
    settings.ensure_directories()
    control_run = tmp_path / "datasets" / "runs" / "control.json"
    candidate_run = tmp_path / "datasets" / "runs" / "candidate.json"
    control_run.write_text("{}", encoding="utf-8")
    candidate_run.write_text("{}", encoding="utf-8")

    manifest = build_pairwise_run_manifest(settings, control_run, candidate_run)

    assert manifest["control_run_path"] == "datasets/runs/control.json"
    assert manifest["candidate_run_path"] == "datasets/runs/candidate.json"

    sidecar = write_run_artifact_manifest(
        settings,
        run_name="pairwise_demo",
        run_kind="pairwise",
        run_artifact_path=control_run,
        run_manifest=manifest,
    )

    assert sidecar == run_manifest_path(settings, "pairwise_demo")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["run_kind"] == "pairwise"
    assert payload["run_artifact_path"] == "datasets/runs/control.json"
