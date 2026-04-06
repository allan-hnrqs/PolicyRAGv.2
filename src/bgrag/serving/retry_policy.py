"""Hybrid retry policy that combines LLM assessment with deterministic signals."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from bgrag.serving.bundle_risk import BundleRiskAssessment, assess_bundle_risk
from bgrag.types import ConversationState, EvidenceBundle, RetrievalAssessment

_EXACTNESS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bemail\b", re.IGNORECASE),
    re.compile(r"\bphone\b|\btelephone\b", re.IGNORECASE),
    re.compile(r"\bcontact\b", re.IGNORECASE),
    re.compile(r"\bform(?: number)?\b", re.IGNORECASE),
    re.compile(r"\btemplate\b", re.IGNORECASE),
    re.compile(r"\bfile name\b|\bfilename\b|\bdocument name\b", re.IGNORECASE),
    re.compile(r"\bapproval authority\b|\bwho approves\b", re.IGNORECASE),
    re.compile(r"\bthreshold\b", re.IGNORECASE),
)

_BRANCH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhat happens if\b", re.IGNORECASE),
    re.compile(r"\bif\b", re.IGNORECASE),
    re.compile(r"\bunless\b", re.IGNORECASE),
    re.compile(r"\bcompare\b|\bdifference\b", re.IGNORECASE),
    re.compile(r"\bhow are we supposed to\b", re.IGNORECASE),
)


class QuestionRiskAssessment(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    exactness_sensitive: bool = False
    branch_sensitive: bool = False
    reasons: list[str] = Field(default_factory=list)


class HybridRetryDecision(BaseModel):
    recommended_next_step: Literal["answer", "retry_retrieve", "browse_official"]
    reasons: list[str] = Field(default_factory=list)
    question_risk: QuestionRiskAssessment
    bundle_risk: BundleRiskAssessment


def assess_question_risk(question: str, conversation_state: ConversationState | None = None) -> QuestionRiskAssessment:
    text_parts = [question]
    if conversation_state is not None:
        text_parts.extend(turn.content for turn in conversation_state.recent_turns[-2:])
    text = "\n".join(part for part in text_parts if part).strip()

    exactness_matches = [pattern.pattern for pattern in _EXACTNESS_PATTERNS if pattern.search(text)]
    branch_matches = [pattern.pattern for pattern in _BRANCH_PATTERNS if pattern.search(text)]

    reasons: list[str] = []
    if exactness_matches:
        reasons.append("question asks for an exact or identifier-like detail")
    if len(branch_matches) >= 2:
        reasons.append("question likely requires multi-branch workflow coverage")

    if exactness_matches and len(branch_matches) >= 2:
        risk_level: Literal["low", "medium", "high"] = "high"
    elif exactness_matches or len(branch_matches) >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    return QuestionRiskAssessment(
        risk_level=risk_level,
        exactness_sensitive=bool(exactness_matches),
        branch_sensitive=len(branch_matches) >= 2,
        reasons=reasons,
    )


def decide_hybrid_retry(
    *,
    question: str,
    evidence: EvidenceBundle,
    retrieval_assessment: RetrievalAssessment,
    conversation_state: ConversationState | None = None,
    enable_official_site_escalation: bool = False,
) -> HybridRetryDecision:
    bundle_risk = assess_bundle_risk(evidence)
    question_risk = assess_question_risk(question, conversation_state)
    reasons: list[str] = []

    if retrieval_assessment.recommended_next_step == "retry_retrieve":
        reasons.append("structured retrieval assessment explicitly requested a retry")
        return HybridRetryDecision(
            recommended_next_step="retry_retrieve",
            reasons=reasons,
            question_risk=question_risk,
            bundle_risk=bundle_risk,
        )

    if retrieval_assessment.recommended_next_step == "browse_official":
        if enable_official_site_escalation:
            reasons.append("structured retrieval assessment explicitly requested official-site browsing")
            return HybridRetryDecision(
                recommended_next_step="browse_official",
                reasons=reasons,
                question_risk=question_risk,
                bundle_risk=bundle_risk,
            )
        reasons.append("official-site browse is unavailable, so downgrade the browse request to an indexed retry")
        return HybridRetryDecision(
            recommended_next_step="retry_retrieve",
            reasons=reasons,
            question_risk=question_risk,
            bundle_risk=bundle_risk,
        )

    if (
        bundle_risk.retry_signal
        and retrieval_assessment.coverage_risk == "high"
        and not retrieval_assessment.sufficient_for_answer
    ):
        reasons.extend(bundle_risk.reasons)
        reasons.append("bundle weakness aligns with a high coverage-risk assessment")
        return HybridRetryDecision(
            recommended_next_step="retry_retrieve",
            reasons=reasons,
            question_risk=question_risk,
            bundle_risk=bundle_risk,
        )

    if (
        question_risk.exactness_sensitive
        and retrieval_assessment.exactness_risk == "high"
        and not retrieval_assessment.sufficient_for_answer
    ):
        reasons.extend(question_risk.reasons)
        reasons.append("exactness-sensitive question with high exactness risk should retry before answering")
        return HybridRetryDecision(
            recommended_next_step="retry_retrieve",
            reasons=reasons,
            question_risk=question_risk,
            bundle_risk=bundle_risk,
        )

    if (
        question_risk.branch_sensitive
        and bundle_risk.risk_level == "high"
        and retrieval_assessment.coverage_risk in {"medium", "high"}
    ):
        reasons.extend(question_risk.reasons)
        reasons.extend(bundle_risk.reasons)
        reasons.append("branch-sensitive question plus weak bundle justifies one indexed retry")
        return HybridRetryDecision(
            recommended_next_step="retry_retrieve",
            reasons=reasons,
            question_risk=question_risk,
            bundle_risk=bundle_risk,
        )

    if retrieval_assessment.sufficient_for_answer:
        reasons.append("structured retrieval assessment judged the bundle sufficient")
    else:
        reasons.append("no retry trigger fired despite a non-sufficient assessment")

    return HybridRetryDecision(
        recommended_next_step="answer",
        reasons=reasons,
        question_risk=question_risk,
        bundle_risk=bundle_risk,
    )
