"""Run and index manifest helpers."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bgrag.config import Settings
from bgrag.profiles.models import RuntimeProfile

RUN_MANIFEST_SCHEMA_VERSION = 2


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    stable = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(stable)


def tree_sha256(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def code_fingerprint(settings: Settings) -> str:
    src_root = settings.resolve(Path("src/bgrag"))
    source_files = src_root.rglob("*.py")
    return tree_sha256(source_files, settings.project_root)


def profile_sha256(settings: Settings, profile_name: str) -> str:
    return file_sha256(settings.resolved_profiles_dir / f"{profile_name}.yaml")


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "default"


def repo_relative_path(settings: Settings, path: Path) -> str:
    resolved = settings.resolve(path)
    try:
        return resolved.relative_to(settings.project_root).as_posix()
    except ValueError:
        return str(resolved)


def workspace_fingerprint(settings: Settings) -> str:
    normalized_root = str(settings.project_root.resolve()).replace("\\", "/").lower()
    return _sha256_bytes(normalized_root.encode("utf-8"))[:8]


def build_run_token() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{secrets.token_hex(2)}"


def build_run_name(prefix: str, *, run_token: str | None = None) -> str:
    return f"{prefix}_{run_token or build_run_token()}"


def derive_index_namespace(settings: Settings, profile_name: str) -> str:
    chunks_hash = file_sha256(settings.resolve(Path("datasets/corpus/chunks.jsonl")))[:10]
    model_slug = _sanitize_slug(settings.cohere_embed_model)
    profile_slug = _sanitize_slug(profile_name)
    return f"{profile_slug}_{model_slug}_{chunks_hash}"


def index_namespace_dir(settings: Settings, namespace: str) -> Path:
    return settings.resolve(Path("datasets/index")) / namespace


def index_embeddings_path(settings: Settings, namespace: str) -> Path:
    return index_namespace_dir(settings, namespace) / "chunk_embeddings.json"


def index_manifest_path(settings: Settings, namespace: str) -> Path:
    return index_namespace_dir(settings, namespace) / "index_manifest.json"


def active_index_pointer_path(settings: Settings) -> Path:
    return settings.resolve(Path("datasets/index/active_index.json"))


def write_index_manifest(settings: Settings, namespace: str, manifest: dict[str, object]) -> Path:
    target = index_manifest_path(settings, namespace)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def set_active_index_namespace(settings: Settings, namespace: str) -> Path:
    pointer = active_index_pointer_path(settings)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"namespace": namespace}, indent=2), encoding="utf-8")
    return pointer


def get_active_index_namespace(settings: Settings) -> str:
    pointer = active_index_pointer_path(settings)
    if not pointer.exists():
        raise FileNotFoundError(
            "No active index pointer found. Run `bgrag build-index` or provide an explicit index namespace."
        )
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    namespace = str(payload.get("namespace", "")).strip()
    if not namespace:
        raise RuntimeError("Active index pointer is missing its namespace value.")
    return namespace


def load_index_manifest(settings: Settings, namespace: str | None = None) -> dict[str, object]:
    resolved_namespace = namespace or get_active_index_namespace(settings)
    path = index_manifest_path(settings, resolved_namespace)
    if not path.exists():
        raise FileNotFoundError(f"Index manifest not found for namespace `{resolved_namespace}`: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("namespace", resolved_namespace)
    return payload


def build_index_manifest(settings: Settings, profile_name: str, namespace: str, chunk_count: int) -> dict[str, object]:
    collection_manifest = settings.resolve(Path("datasets/corpus/collection_manifest.json"))
    chunks_path = settings.resolve(Path("datasets/corpus/chunks.jsonl"))
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "namespace": namespace,
        "workspace_fingerprint": workspace_fingerprint(settings),
        "project_root": str(settings.project_root.resolve()),
        "profile_name": profile_name,
        "profile_path": repo_relative_path(settings, settings.resolved_profiles_dir / f"{profile_name}.yaml"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embed_model": settings.cohere_embed_model,
        "elastic_url": settings.elastic_url,
        "chunk_count": chunk_count,
        "chunks_path": repo_relative_path(settings, chunks_path),
        "chunks_sha256": file_sha256(chunks_path),
        "collection_manifest_path": repo_relative_path(settings, collection_manifest),
        "collection_manifest_sha256": json_file_sha256(collection_manifest),
        "profile_sha256": profile_sha256(settings, profile_name),
        "code_sha256": code_fingerprint(settings),
    }


def build_eval_run_manifest(
    settings: Settings,
    profile: RuntimeProfile,
    eval_path: Path,
    index_manifest: dict[str, object],
) -> dict[str, object]:
    resolved_eval_path = settings.resolve(eval_path)
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_fingerprint": workspace_fingerprint(settings),
        "project_root": str(settings.project_root.resolve()),
        "profile_name": profile.name,
        "profile_path": repo_relative_path(settings, settings.resolved_profiles_dir / f"{profile.name}.yaml"),
        "profile_sha256": profile_sha256(settings, profile.name),
        "eval_path": repo_relative_path(settings, resolved_eval_path),
        "eval_sha256": file_sha256(resolved_eval_path),
        "code_sha256": code_fingerprint(settings),
        "index_namespace": index_manifest.get("namespace"),
        "index_manifest_path": repo_relative_path(
            settings,
            index_manifest_path(settings, str(index_manifest["namespace"])),
        ),
        "index_manifest_sha256": json_file_sha256(index_manifest_path(settings, str(index_manifest["namespace"]))),
        "chunks_path": str(index_manifest.get("chunks_path", "")),
        "chunks_sha256": str(index_manifest.get("chunks_sha256", "")),
        "collection_manifest_path": str(index_manifest.get("collection_manifest_path", "")),
        "collection_manifest_sha256": str(index_manifest.get("collection_manifest_sha256", "")),
        "answer_model": profile.answering.model_name,
        "planner_model": profile.answering.planner_model_name or settings.cohere_query_planner_model,
        "judge_model": settings.cohere_judge_model,
        "embed_model": settings.cohere_embed_model,
        "rerank_model": settings.cohere_rerank_model,
    }


def build_pairwise_run_manifest(
    settings: Settings,
    control_run_path: Path,
    candidate_run_path: Path,
) -> dict[str, object]:
    resolved_control = settings.resolve(control_run_path)
    resolved_candidate = settings.resolve(candidate_run_path)
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_fingerprint": workspace_fingerprint(settings),
        "project_root": str(settings.project_root.resolve()),
        "code_sha256": code_fingerprint(settings),
        "control_run_path": repo_relative_path(settings, resolved_control),
        "control_run_sha256": file_sha256(resolved_control),
        "candidate_run_path": repo_relative_path(settings, resolved_candidate),
        "candidate_run_sha256": file_sha256(resolved_candidate),
        "judge_model": settings.openai_eval_model,
    }


def run_manifest_dir(settings: Settings) -> Path:
    return settings.resolved_runs_dir / "manifests"


def run_manifest_path(settings: Settings, run_name: str) -> Path:
    return run_manifest_dir(settings) / f"{run_name}.manifest.json"


def write_run_artifact_manifest(
    settings: Settings,
    *,
    run_name: str,
    run_kind: str,
    run_artifact_path: Path,
    run_manifest: dict[str, object],
) -> Path:
    target = run_manifest_path(settings, run_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "run_name": run_name,
                "run_kind": run_kind,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_fingerprint": workspace_fingerprint(settings),
                "run_artifact_path": repo_relative_path(settings, run_artifact_path),
                "run_artifact_sha256": file_sha256(run_artifact_path),
                "run_manifest": run_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target
