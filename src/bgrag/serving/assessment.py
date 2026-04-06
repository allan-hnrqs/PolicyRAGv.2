"""Structured retrieval sufficiency assessment."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local install state
    Agent = Any  # type: ignore[assignment]
    CohereModel = Any  # type: ignore[assignment]
    CohereProvider = Any  # type: ignore[assignment]
    _PYDANTIC_AI_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _PYDANTIC_AI_IMPORT_ERROR = None

from bgrag.config import Settings
from bgrag.types import ConversationState, EvidenceBundle, RetrievalAssessment


def _require_pydantic_ai() -> None:
    if _PYDANTIC_AI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PydanticAI is required for retrieval assessment. "
            "Install the repo dependencies before using the strict agentic path."
        ) from _PYDANTIC_AI_IMPORT_ERROR


def _compact_text(text: str, *, max_chars: int = 600) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _render_evidence(evidence: EvidenceBundle, *, max_chunks: int) -> str:
    sections: list[str] = []
    for index, chunk in enumerate(evidence.packed_chunks[:max_chunks], start=1):
        sections.append(
            "\n".join(
                [
                    f"Chunk {index}",
                    f"- chunk_id: {chunk.chunk_id}",
                    f"- title: {chunk.title}",
                    f"- heading: {chunk.heading or ''}",
                    f"- url: {chunk.canonical_url}",
                    f"- text: {_compact_text(chunk.text)}",
                ]
            )
        )
    return "\n\n".join(sections) or "<no packed evidence>"


def _render_conversation_state(state: ConversationState | None) -> str:
    if state is None:
        return "<none>"
    turn_lines = [
        f"- {turn.role}: {_compact_text(turn.content, max_chars=180)}"
        for turn in state.recent_turns[-6:]
    ]
    return "\n".join(
        [
            f"conversation_id: {state.conversation_id}",
            f"resolved_query: {state.resolved_query or ''}",
            f"active_entities: {', '.join(state.active_entities) or '<none>'}",
            f"comparison_axes: {', '.join(state.comparison_axes) or '<none>'}",
            "recent_turns:",
            *(turn_lines or ["- <none>"]),
        ]
    )


@lru_cache(maxsize=8)
def _assessment_agent(model_name: str, api_key: str) -> Agent[None, RetrievalAssessment]:
    _require_pydantic_ai()
    return Agent(
        CohereModel(model_name, provider=CohereProvider(api_key=api_key)),
        output_type=RetrievalAssessment,
        system_prompt=(
            "You assess whether retrieved procurement-policy evidence is sufficient for a safe answer.\n"
            "Return a structured decision only.\n"
            "Use browse_official only when the current evidence is likely missing decisive support.\n"
            "Use retry_retrieve when the evidence is partially relevant but looks incomplete or poorly prioritized.\n"
            "Use answer only when the evidence appears sufficient for a complete and exact answer.\n"
            "Be conservative on exact identifiers, contact details, form numbers, and approval authorities."
        ),
        defer_model_check=True,
        retries=1,
        output_retries=1,
    )


def assess_retrieval(
    settings: Settings,
    *,
    model_name: str,
    question: str,
    evidence: EvidenceBundle,
    conversation_state: ConversationState | None = None,
    max_chunks: int = 8,
) -> RetrievalAssessment:
    settings.require_cohere_key("Retrieval assessment")
    prompt = "\n\n".join(
        [
            f"Question:\n{question}",
            f"Conversation state:\n{_render_conversation_state(conversation_state)}",
            f"Retrieval notes:\n{', '.join(evidence.notes) or '<none>'}",
            f"Evidence:\n{_render_evidence(evidence, max_chunks=max_chunks)}",
            (
                "Decision rules:\n"
                "- sufficient_for_answer should be true only if the current evidence plausibly supports a complete answer.\n"
                "- coverage_risk is about likely missing branches, steps, or conditions.\n"
                "- exactness_risk is about risky specifics like form numbers, named templates, contacts, deadlines, or approval authorities.\n"
                "- support_conflict is true only when the evidence itself seems internally conflicting.\n"
                "- recommended_next_step must be one of answer, retry_retrieve, browse_official."
            ),
        ]
    )
    result = _assessment_agent(model_name, settings.cohere_api_key).run_sync(prompt)
    assessment = result.output
    return assessment.model_copy(
        update={
            "raw_response": {
                "provider": "pydantic_ai",
                "model_name": model_name,
            }
        }
    )
