"""Typed settings and path helpers."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def detect_project_root(start: Path | None = None) -> Path:
    override = os.environ.get("BGRAG_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "bgrag").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=detect_project_root)
    data_dir: Path = Field(default_factory=lambda: Path("datasets"))
    docs_dir: Path = Field(default_factory=lambda: Path("docs"))
    profiles_dir: Path = Field(default_factory=lambda: Path("profiles"))
    cache_dir: Path = Field(default_factory=lambda: Path(".cache"))
    raw_dir: Path = Field(default_factory=lambda: Path("datasets/raw"))
    corpus_dir: Path = Field(default_factory=lambda: Path("datasets/corpus"))
    index_dir: Path = Field(default_factory=lambda: Path("datasets/index"))
    runs_dir: Path = Field(default_factory=lambda: Path("datasets/runs"))
    cohere_api_key: str = ""
    openai_api_key: str = ""
    cohere_chat_model: str = "command-a-03-2025"
    cohere_query_planner_model: str = "command-a-03-2025"
    cohere_judge_model: str = "command-a-03-2025"
    openai_eval_model: str = "gpt-5.4"
    openai_eval_reasoning_effort: str = "low"
    openai_eval_max_output_tokens: int = 512
    cohere_embed_model: str = "embed-english-v3.0"
    cohere_rerank_model: str = "rerank-v4.0-fast"
    cohere_embed_batch_size: int = 96
    ragas_max_output_tokens: int = 3000
    ragas_timeout_seconds: int = 480
    ragas_max_workers: int = 4
    elastic_url: str = "http://127.0.0.1:9200"
    opensearch_url: str = "http://127.0.0.1:9200"
    elastic_request_timeout: int = 60
    chat_temperature: float = 0.0
    top_k: int = 16
    retrieval_candidate_k: int = 48
    max_packed_docs: int = 24
    max_doc_chars: int = 1600
    chat_max_output_tokens: int = 500
    answer_timeout_seconds: int = 90

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def resolved_data_dir(self) -> Path:
        return self.resolve(self.data_dir)

    @property
    def resolved_profiles_dir(self) -> Path:
        return self.resolve(self.profiles_dir)

    @property
    def resolved_runs_dir(self) -> Path:
        return self.resolve(self.runs_dir)

    def ensure_directories(self) -> None:
        for path in (
            self.resolved_data_dir,
            self.resolve(self.raw_dir),
            self.resolve(self.corpus_dir),
            self.resolve(self.index_dir),
            self.resolved_runs_dir,
            self.resolve(self.cache_dir),
        ):
            path.mkdir(parents=True, exist_ok=True)

    def has_cohere_key(self) -> bool:
        return bool(self.cohere_api_key.strip())

    def require_cohere_key(self, purpose: str) -> None:
        if not self.has_cohere_key():
            raise RuntimeError(
                f"{purpose} requires COHERE_API_KEY in this repo environment. "
                "Add it to this repo's .env or export it in the shell."
            )

    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.strip())

    def require_openai_key(self, purpose: str) -> None:
        if not self.has_openai_key():
            raise RuntimeError(
                f"{purpose} requires OPENAI_API_KEY in this repo environment. "
                "Add it to this repo's .env or export it in the shell."
            )
