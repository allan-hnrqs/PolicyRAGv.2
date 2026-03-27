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
        "Decide whether the user's question actually needs query decomposition for retrieval.\n"
        "Rules:\n"
        "1. Return JSON only.\n"
        "2. Prefer not decomposing unless the question clearly needs multiple distinct retrieval aspects.\n"
        f"3. If decomposition is needed, return at most {max_expanded_queries} focused queries.\n"
        "4. Each query should target one distinct aspect needed to answer the original question.\n"
        "5. Keep procurement terms and source-specific language intact.\n"
        "6. If the original question is already focused and self-contained, set should_decompose to false and return an empty queries list.\n"
        "7. Do not answer the question.\n"
        "8. Do not include the original question verbatim unless it is needed as one focused sub-query.\n"
        'Return this exact shape: {"should_decompose":true|false,"queries":[string,...]}\n\n'
        f"Original question:\n{question}"
    )


def parse_query_plan(
    question: str,
    parsed: dict[str, object],
    *,
    max_expanded_queries: int,
) -> list[str]:
    raw_queries = parsed.get("queries", [])
    if not isinstance(raw_queries, list):
        raise ValueError("Query planner response must include a list under 'queries'")

    should_decompose = parsed.get("should_decompose")
    if isinstance(should_decompose, bool) and not should_decompose:
        return []

    return normalize_expanded_queries(
        question,
        raw_queries,
        max_expanded_queries=max_expanded_queries,
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
        if not isinstance(parsed, dict):
            raise ValueError("Query planner response must be a JSON object")
        return parse_query_plan(
            question,
            parsed,
            max_expanded_queries=max_expanded_queries,
        )
