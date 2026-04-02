"""Profile-scoped runtime settings helpers."""

from __future__ import annotations

from bgrag.config import Settings
from bgrag.profiles.models import RuntimeProfile


def build_runtime_settings(settings: Settings, profile: RuntimeProfile) -> Settings:
    answering = profile.answering
    return settings.model_copy(
        update={
            "cohere_chat_model": getattr(answering, "model_name", settings.cohere_chat_model),
            "cohere_query_planner_model": (
                getattr(answering, "planner_model_name", None) or settings.cohere_query_planner_model
            ),
            "max_packed_docs": getattr(answering, "max_packed_docs", settings.max_packed_docs),
            "max_doc_chars": getattr(answering, "max_doc_chars", settings.max_doc_chars),
        }
    )
