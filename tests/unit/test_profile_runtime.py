from pathlib import Path

from bgrag.config import Settings
from bgrag.profiles.loader import load_profile
from bgrag.profiles.runtime import build_runtime_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_runtime_settings_applies_answer_and_planner_model_overrides() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("demo_vector_documents", settings)

    runtime = build_runtime_settings(settings, profile)

    assert runtime.cohere_chat_model == "command-a-03-2025"
    assert runtime.cohere_query_planner_model == "command-r7b-12-2024"
    assert runtime.max_packed_docs == 12
    assert runtime.max_doc_chars == 1200
