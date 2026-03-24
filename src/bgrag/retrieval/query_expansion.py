"""LLM-assisted retrieval query decomposition."""

from __future__ import annotations

import json

import cohere
import cohere.types as ct

from bgrag.config import Settings


def normalize_expanded_queries(
    question: str,
    raw_queries: list[object],
    *,
    max_expanded_queries: int,
) -> list[str]:
    normalized_question = " ".join(question.split()).strip()
    seen = {normalized_question.lower()}
    clean_queries: list[str] = []
    for item in raw_queries:
        if not isinstance(item, str):
            continue
        query = " ".join(item.split()).strip()
        if not query:
            continue
        if query.lower() in seen:
            continue
        clean_queries.append(query)
        seen.add(query.lower())
        if len(clean_queries) >= max_expanded_queries:
            break
    return clean_queries


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


def _planner_prompt(question: str, max_expanded_queries: int) -> str:
    return (
        "You are planning retrieval queries for a procurement-policy RAG system.\n"
        "Break a complex user question into a small set of focused retrieval queries.\n"
        "Rules:\n"
        "1. Return JSON only.\n"
        f"2. Return at most {max_expanded_queries} queries.\n"
        "3. Each query should target one distinct aspect needed to answer the original question.\n"
        "4. Keep procurement terms and source-specific language intact.\n"
        "5. Do not include the original question verbatim unless it is needed as one focused sub-query.\n"
        'Return this exact shape: {"queries":[string,...]}\n\n'
        f"Original question:\n{question}"
    )


class CohereQueryExpander:
    def __init__(self, settings: Settings) -> None:
        settings.require_cohere_key("Query decomposition")
        self.settings = settings
        self.client = cohere.ClientV2(settings.cohere_api_key)

    def expand(self, question: str, max_expanded_queries: int) -> list[str]:
        response = self.client.chat(
            model=self.settings.cohere_query_planner_model,
            messages=[ct.UserChatMessageV2(content=_planner_prompt(question, max_expanded_queries))],
            response_format=ct.JsonObjectResponseFormatV2(),
            temperature=0,
            max_tokens=400,
        )
        text = _extract_text_from_chat_response(response)
        parsed = json.loads(text)
        queries = parsed.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("Query planner response must include a list under 'queries'")
        return normalize_expanded_queries(
            question,
            queries,
            max_expanded_queries=max_expanded_queries,
        )
