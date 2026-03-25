"""Structured LLM judge for parity evaluation."""

from __future__ import annotations

import json

import cohere
import cohere.types as ct

from bgrag.config import Settings
from bgrag.types import AnswerResult, EvalCase


def _judge_prompt(case: EvalCase, answer: AnswerResult) -> str:
    payload = {
        "question": case.question,
        "answer": answer.answer_text,
        "required_claims": case.required_claims,
        "forbidden_claims": case.forbidden_claims,
        "expect_abstain": case.expect_abstain,
    }
    return (
        "You are evaluating whether an answer satisfies a procurement-policy evaluation case.\n"
        "Judge only the answer text you are given. Do not infer missing facts.\n"
        "Return JSON with this exact shape:\n"
        "{"
        '"required_claims":[{"claim":string,"supported":boolean,"reason":string}],'
        '"forbidden_claims":[{"claim":string,"violated":boolean,"reason":string}],'
        '"answer_abstains":boolean,'
        '"abstain_correct":boolean|null,'
        '"overall_notes":string'
        "}\n\n"
        f"Case:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


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


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Judge response {field_name} must be a boolean")
    return value


def _normalize_claim_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip().casefold()


def _normalize_judgment(parsed: dict[str, object], case: EvalCase) -> dict[str, object]:
    required_items = parsed.get("required_claims", [])
    forbidden_items = parsed.get("forbidden_claims", [])
    if not isinstance(required_items, list):
        raise ValueError("Judge response required_claims must be a list")
    if not isinstance(forbidden_items, list):
        raise ValueError("Judge response forbidden_claims must be a list")
    if len(required_items) != len(case.required_claims):
        raise ValueError(
            "Judge response required_claims length mismatch: "
            f"expected {len(case.required_claims)}, got {len(required_items)}"
        )
    if len(forbidden_items) != len(case.forbidden_claims):
        raise ValueError(
            "Judge response forbidden_claims length mismatch: "
            f"expected {len(case.forbidden_claims)}, got {len(forbidden_items)}"
        )
    if any(not isinstance(item, dict) for item in required_items):
        raise ValueError("Judge response required_claims items must be objects")
    if any(not isinstance(item, dict) for item in forbidden_items):
        raise ValueError("Judge response forbidden_claims items must be objects")
    for index, (item, expected_claim) in enumerate(zip(required_items, case.required_claims, strict=True)):
        returned_claim = item.get("claim")
        if _normalize_claim_text(returned_claim) != _normalize_claim_text(expected_claim):
            raise ValueError(
                "Judge response required_claims claim mismatch at index "
                f"{index}: expected {expected_claim!r}, got {returned_claim!r}"
            )
    for index, (item, expected_claim) in enumerate(zip(forbidden_items, case.forbidden_claims, strict=True)):
        returned_claim = item.get("claim")
        if _normalize_claim_text(returned_claim) != _normalize_claim_text(expected_claim):
            raise ValueError(
                "Judge response forbidden_claims claim mismatch at index "
                f"{index}: expected {expected_claim!r}, got {returned_claim!r}"
            )

    supported_count = sum(
        1
        for item in required_items
        if isinstance(item, dict) and _require_bool(item.get("supported"), "required_claims[].supported")
    )
    violated_count = sum(
        1
        for item in forbidden_items
        if isinstance(item, dict) and _require_bool(item.get("violated"), "forbidden_claims[].violated")
    )
    required_total = len(case.required_claims)
    forbidden_total = len(case.forbidden_claims)
    abstain_correct = parsed.get("abstain_correct")
    if abstain_correct is not None and not isinstance(abstain_correct, bool):
        raise ValueError("Judge response abstain_correct must be boolean or null")

    return {
        "required_claims": required_items,
        "forbidden_claims": forbidden_items,
        "answer_abstains": _require_bool(parsed.get("answer_abstains"), "answer_abstains"),
        "abstain_correct": abstain_correct,
        "overall_notes": str(parsed.get("overall_notes", "")),
        "required_claim_recall": supported_count / required_total if required_total else 0.0,
        "required_claim_supported_count": supported_count,
        "required_claim_total": required_total,
        "forbidden_claim_violation_count": violated_count,
        "forbidden_claim_total": forbidden_total,
        "forbidden_claims_clean": violated_count == 0,
    }


class CohereJudge:
    def __init__(self, settings: Settings) -> None:
        settings.require_cohere_key("LLM judging")
        self.settings = settings
        self.client = cohere.ClientV2(settings.cohere_api_key)

    def judge(self, case: EvalCase, answer: AnswerResult) -> dict[str, object]:
        response = self.client.chat(
            model=self.settings.cohere_judge_model,
            messages=[ct.UserChatMessageV2(content=_judge_prompt(case, answer))],
            response_format=ct.JsonObjectResponseFormatV2(),
            temperature=0,
            max_tokens=1200,
        )
        text = _extract_text_from_chat_response(response)
        parsed = json.loads(text)
        normalized = _normalize_judgment(parsed, case)
        normalized["raw_judge_text"] = text
        return normalized
