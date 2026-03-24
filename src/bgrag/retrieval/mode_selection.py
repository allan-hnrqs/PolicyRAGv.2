"""LLM-assisted retrieval-mode selection."""

from __future__ import annotations

import json
from dataclasses import dataclass

import cohere
import cohere.types as ct

from bgrag.config import Settings
from bgrag.types import EvidenceBundle, SourceFamily


@dataclass(frozen=True)
class RetrievalModeDecision:
    mode: str
    rationale: str


def _extract_text_from_chat_response(response: object) -> str:
    message = getattr(response, "message", None)
    contents = getattr(message, "content", None)
    if not contents:
        return ""
    parts: list[str] = []
    for item in contents:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
    return "".join(parts).strip()


def normalize_retrieval_mode_decision(raw_text: str) -> RetrievalModeDecision:
    parsed = json.loads(raw_text)
    mode = str(parsed.get("mode", "baseline")).strip().lower()
    if mode not in {"baseline", "page_family_expansion"}:
        mode = "baseline"
    rationale = " ".join(str(parsed.get("rationale", "")).split()).strip()
    return RetrievalModeDecision(mode=mode, rationale=rationale)


def _summarize_evidence(evidence: EvidenceBundle, *, max_chunks: int) -> str:
    lines: list[str] = []
    for chunk in evidence.packed_chunks[:max_chunks]:
        source_label = "buyers_guide" if chunk.source_family == SourceFamily.BUYERS_GUIDE else chunk.source_family.value
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else (chunk.heading or chunk.title)
        lines.append(
            f"- chunk_id={chunk.chunk_id} | source={source_label} | order={chunk.order} | "
            f"title={chunk.title} | heading={heading} | url={chunk.canonical_url}"
        )
    return "\n".join(lines)


def _build_retrieval_mode_prompt(question: str, evidence: EvidenceBundle, *, max_chunks: int) -> str:
    retrieval_aspects = ""
    if len(evidence.retrieval_queries) > 1:
        aspect_lines = "\n".join(f"- {query}" for query in evidence.retrieval_queries[1:])
        retrieval_aspects = f"Expanded retrieval aspects:\n{aspect_lines}\n\n"
    notes_block = ""
    if evidence.notes:
        notes_block = "Current retrieval notes:\n" + "\n".join(f"- {note}" for note in evidence.notes) + "\n\n"
    evidence_summary = _summarize_evidence(evidence, max_chunks=max_chunks)
    return (
        "You are selecting a retrieval mode for a procurement-policy RAG system.\n"
        "Return JSON only in this exact shape:\n"
        '{"mode":"baseline","rationale":"short explanation"}\n\n'
        "Allowed values for mode are:\n"
        '- "baseline"\n'
        '- "page_family_expansion"\n\n'
        "Choose page_family_expansion only when the current retrieved evidence suggests a page-coverage problem that would likely be improved by expanding within a closely related Buyer\\'s Guide page family.\n"
        "Strong signals for page_family_expansion:\n"
        "1. The question is mainly about workflow branches, where to start, or multi-part operational guidance.\n"
        "2. The question asks about both the rule and what happens next, or compares the ordinary rule with an exception, cure period, consequence, or follow-up action.\n"
        "2. The current evidence is mostly Buyer\\'s Guide content.\n"
        "3. The current evidence appears to rely on deep sections from one or more related pages without enough overview or sibling-page context.\n"
        "4. Expanded retrieval aspects or the question itself suggest multiple operational subparts, and the likely fix is better page-family coverage rather than more policy/support-source detail.\n\n"
        "Choose baseline when any of these is true:\n"
        "1. The current evidence already looks sufficient.\n"
        "2. The question is mainly asking for an exact identifier, exact form number, or other missing-detail question.\n"
        "3. Supporting policy/directive detail or source-boundary reasoning appears central.\n"
        "4. Expanding page-family context would likely drift away from the relevant evidence.\n\n"
        f"Question:\n{question}\n\n"
        f"{retrieval_aspects}"
        f"{notes_block}"
        f"Current retrieved evidence summary:\n{evidence_summary}"
    )


class CohereRetrievalModeSelector:
    def __init__(self, settings: Settings) -> None:
        settings.require_cohere_key("Retrieval mode selection")
        self.settings = settings
        self.client = cohere.ClientV2(settings.cohere_api_key)

    def select(self, question: str, evidence: EvidenceBundle, *, max_chunks: int) -> RetrievalModeDecision:
        response = self.client.chat(
            model=self.settings.cohere_query_planner_model,
            messages=[
                ct.UserChatMessageV2(
                    content=_build_retrieval_mode_prompt(question, evidence, max_chunks=max_chunks)
                )
            ],
            response_format=ct.JsonObjectResponseFormatV2(),
            temperature=0,
            max_tokens=200,
        )
        text = _extract_text_from_chat_response(response)
        return normalize_retrieval_mode_decision(text)
