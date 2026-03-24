"""CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import dotenv_values
from rich.console import Console
from rich.table import Table

from bgrag.config import Settings, detect_project_root
from bgrag.eval.pairwise import compare_pairwise_runs
from bgrag.eval.ragas_runner import run_ragas_eval
from bgrag.eval.runner import run_eval
from bgrag.manifests import (
    build_eval_run_manifest,
    build_pairwise_run_manifest,
    get_active_index_namespace,
    load_index_manifest,
    write_run_artifact_manifest,
)
from bgrag.parity import freeze_feat_parity_inputs
from bgrag.pipeline import build_answer_callback, run_build_corpus, run_build_index, run_collect
from bgrag.profiles.loader import list_profiles, load_profile

app = typer.Typer(help="Buyer’s Guide-first RAG backend.")
console = Console()


def _repo_env_values(project_root: Path) -> dict[str, str]:
    env_values = dotenv_values(project_root / ".env")
    return {key.lower(): value for key, value in env_values.items() if value is not None}


def _settings() -> Settings:
    project_root = detect_project_root(Path.cwd())
    explicit_values = _repo_env_values(project_root)
    settings = Settings(project_root=project_root, **explicit_values)
    settings.ensure_directories()
    return settings


@app.command("collect")
def collect_command(max_pages: int = 300) -> None:
    settings = _settings()
    documents = run_collect(settings, max_pages=max_pages)
    console.print(f"Collected and normalized {len(documents)} documents.")


@app.command("build-corpus")
def build_corpus_command(profile: str = "baseline") -> None:
    settings = _settings()
    chunks = run_build_corpus(settings, profile)
    console.print(f"Built {len(chunks)} chunks with profile {profile}.")


@app.command("build-index")
def build_index_command(
    profile: str = "baseline",
    limit_chunks: int = 0,
    index_namespace: str | None = typer.Option(None, "--index-namespace"),
) -> None:
    settings = _settings()
    stats = run_build_index(settings, profile, limit_chunks=limit_chunks, index_namespace=index_namespace)
    console.print(json.dumps(stats, indent=2))


@app.command("query")
def query_command(
    question: str,
    profile: str = "baseline",
    index_namespace: str | None = typer.Option(None, "--index-namespace"),
) -> None:
    settings = _settings()
    answer_callback = build_answer_callback(settings, profile, index_namespace=index_namespace)

    class AdHocCase:
        def __init__(self, question: str) -> None:
            self.question = question

    result = answer_callback(AdHocCase(question))
    console.print(result.answer_text)


@app.command("eval")
def eval_command(
    eval_path: Path,
    profile: str = "baseline",
    index_namespace: str | None = typer.Option(None, "--index-namespace"),
) -> None:
    settings = _settings()
    runtime_profile = load_profile(profile, settings)
    index_manifest = load_index_manifest(settings, index_namespace)
    answer_callback = build_answer_callback(settings, profile, index_namespace=str(index_manifest["namespace"]))
    result = run_eval(
        settings,
        runtime_profile,
        eval_path,
        answer_callback,
        run_manifest=build_eval_run_manifest(settings, runtime_profile, eval_path, index_manifest),
    )
    output_path = settings.resolved_runs_dir / f"{result.run_name}.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    manifest_path = write_run_artifact_manifest(
        settings,
        run_name=result.run_name,
        run_kind="eval",
        run_artifact_path=output_path,
        run_manifest=result.run_manifest,
    )
    console.print(f"Wrote eval run to {output_path}")
    console.print(f"Wrote run manifest to {manifest_path}")
    console.print(json.dumps(result.overall_metrics, indent=2))


@app.command("eval-ragas")
def eval_ragas_command(
    eval_path: Path,
    profile: str = "baseline",
    index_namespace: str | None = typer.Option(None, "--index-namespace"),
) -> None:
    settings = _settings()
    runtime_profile = load_profile(profile, settings)
    index_manifest = load_index_manifest(settings, index_namespace)
    answer_callback = build_answer_callback(settings, profile, index_namespace=str(index_manifest["namespace"]))
    result = run_ragas_eval(
        settings,
        runtime_profile,
        eval_path,
        answer_callback,
        run_manifest=build_eval_run_manifest(settings, runtime_profile, eval_path, index_manifest),
    )
    output_path = settings.resolved_runs_dir / f"{result.run_name}.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    manifest_path = write_run_artifact_manifest(
        settings,
        run_name=result.run_name,
        run_kind="eval_ragas",
        run_artifact_path=output_path,
        run_manifest=result.run_manifest,
    )
    console.print(f"Wrote Ragas eval run to {output_path}")
    console.print(f"Wrote run manifest to {manifest_path}")
    console.print(json.dumps(result.overall_metrics, indent=2))


@app.command("eval-pairwise")
def eval_pairwise_command(
    control_run: Path,
    candidate_run: Path,
) -> None:
    settings = _settings()
    resolved_control_run = settings.resolve(control_run)
    resolved_candidate_run = settings.resolve(candidate_run)
    result = compare_pairwise_runs(
        settings,
        resolved_control_run,
        resolved_candidate_run,
        run_manifest=build_pairwise_run_manifest(settings, resolved_control_run, resolved_candidate_run),
    )
    output_path = settings.resolved_runs_dir / f"{result.run_name}.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    manifest_path = write_run_artifact_manifest(
        settings,
        run_name=result.run_name,
        run_kind="pairwise",
        run_artifact_path=output_path,
        run_manifest=result.run_manifest,
    )
    console.print(f"Wrote pairwise eval run to {output_path}")
    console.print(f"Wrote run manifest to {manifest_path}")
    console.print(json.dumps(result.overall_metrics, indent=2))


@app.command("freeze-baseline")
def freeze_baseline_command(workspace_root: Path = Path("..")) -> None:
    settings = _settings()
    copied = freeze_feat_parity_inputs(settings, settings.resolve(workspace_root))
    console.print(json.dumps(copied, indent=2))


@app.command("inspect")
def inspect_command(profile: str = "baseline") -> None:
    settings = _settings()
    runtime_profile = load_profile(profile, settings)
    active_namespace = None
    try:
        active_namespace = get_active_index_namespace(settings)
    except Exception:
        active_namespace = None
    table = Table(title=f"Profile: {runtime_profile.name}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("description", runtime_profile.description)
    table.add_row("source_topology", runtime_profile.retrieval.source_topology)
    table.add_row("retrieval_mode", runtime_profile.retrieval.retrieval_mode)
    table.add_row("chunker", runtime_profile.chunking.chunker)
    table.add_row("answer_strategy", runtime_profile.answering.strategy)
    table.add_row("eval_suite", runtime_profile.evaluation.suite_name)
    table.add_row("active_index_namespace", active_namespace or "<none>")
    console.print(table)
    console.print(f"Available profiles: {', '.join(list_profiles(settings))}")


if __name__ == "__main__":
    app()
