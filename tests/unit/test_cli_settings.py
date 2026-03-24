from pathlib import Path

from bgrag.cli import _repo_env_values, _settings


def test_repo_env_values_lowercase_and_preserve_repo_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=repo-openai\nCOHERE_API_KEY=repo-cohere\nOPENAI_EVAL_MODEL=gpt-5.4\n",
        encoding="utf-8",
    )

    values = _repo_env_values(tmp_path)

    assert values["openai_api_key"] == "repo-openai"
    assert values["cohere_api_key"] == "repo-cohere"
    assert values["openai_eval_model"] == "gpt-5.4"


def test_settings_prefers_repo_root_from_cwd(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='bgrag'\n", encoding="utf-8")
    (tmp_path / "src" / "bgrag").mkdir(parents=True)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=repo-openai\nCOHERE_API_KEY=repo-cohere\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = _settings()

    assert settings.project_root == tmp_path
    assert settings.openai_api_key == "repo-openai"
    assert settings.cohere_api_key == "repo-cohere"
