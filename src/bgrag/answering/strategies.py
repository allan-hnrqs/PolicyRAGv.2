"""Pluggable answer strategies."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from time import perf_counter

import cohere
import cohere.types as ct
from instructor import from_cohere
from pydantic import BaseModel, Field

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"instructor\.providers\.gemini\.client",
)

from bgrag.config import Settings
from bgrag.registry import answer_strategy_registry
from bgrag.types import AnswerCitation, AnswerResult, ChunkRecord, EvidenceBundle, SourceFamily


@dataclass(frozen=True)
class ModeAwareAnswerPlan:
    answer_mode: str
    should_abstain: bool
    abstain_reason: str
    coverage_points: list[str]


@dataclass(frozen=True)
class SelectiveAnswerPlanRoute:
    prompt: str
    selected_path: str
    abstained: bool


@dataclass(frozen=True)
class StructuredAnswerContract:
    answer_mode: str
    should_abstain: bool
    abstain_reason: str
    slots: dict[str, str]


@dataclass(frozen=True)
class StructuredAnswerSlotValue:
    text: str
    citation_chunk_ids: list[str]


@dataclass(frozen=True)
class CitedStructuredAnswerContract:
    answer_mode: str
    should_abstain: bool
    abstain_reason: str
    slots: dict[str, StructuredAnswerSlotValue]


class StructuredAnswerSlotPayload(BaseModel):
    text: str = Field(
        default="",
        description="One short evidence-backed sentence or phrase, ideally under 30 words.",
    )
    citation_chunk_ids: list[str] = Field(
        default_factory=list,
        description="One to three chunk IDs that directly support the slot text.",
    )


class CitedStructuredAnswerContractPayload(BaseModel):
    answer_mode: str = Field(default="direct_rule", description="workflow, navigation, missing_detail, or direct_rule")
    should_abstain: bool = Field(default=False, description="True only when the exact requested detail is not established")
    abstain_reason: str = Field(default="", description="Short reason for abstention when should_abstain is true")
    slots: dict[str, StructuredAnswerSlotPayload] = Field(
        default_factory=dict,
        description="Only populate slot keys allowed for the chosen answer mode.",
    )


@dataclass(frozen=True)
class AnswerRewriteVerdict:
    action: str
    confidence: str
    rationale: str
    omission_risk: bool
    exact_detail_abstain_risk: bool
    unsupported_detail_risk: bool


class AnswerRewriteVerdictPayload(BaseModel):
    action: str = Field(default="keep", description="keep or rewrite_structured_contract")
    confidence: str = Field(default="low", description="low, medium, or high")
    rationale: str = Field(default="", description="Short explanation of the verdict")
    omission_risk: bool = False
    exact_detail_abstain_risk: bool = False
    unsupported_detail_risk: bool = False


@dataclass(frozen=True)
class MissingDetailExactnessVerdict:
    confidence: str
    rationale: str
    exact_detail_overstatement_risk: bool
    offending_details: list[str]


class MissingDetailExactnessVerdictPayload(BaseModel):
    confidence: str = Field(default="low", description="low, medium, or high")
    rationale: str = Field(default="", description="Short explanation of the verdict")
    exact_detail_overstatement_risk: bool = False
    offending_details: list[str] = Field(
        default_factory=list,
        description=(
            "Specific exact identifiers, template names, forms, contacts, or nearby details that the "
            "draft presents too strongly."
        ),
    )


@dataclass(frozen=True)
class ContractSlotCoverageVerdict:
    confidence: str
    rationale: str
    missing_or_weakened_slots: list[str]
    unsupported_detail_risk: bool


class ContractSlotCoverageVerdictPayload(BaseModel):
    confidence: str = Field(default="low", description="low, medium, or high")
    rationale: str = Field(default="", description="Short explanation of the verdict")
    missing_or_weakened_slots: list[str] = Field(
        default_factory=list,
        description="List of populated contract slot keys that are absent or materially weakened in the draft answer.",
    )
    unsupported_detail_risk: bool = False


@dataclass(frozen=True)
class AnswerRepairPlan:
    needs_revision: bool
    missing_supported_points: list[str]
    unsupported_or_overstated_points: list[str]


def _render_evidence_sections(chunks: list[ChunkRecord]) -> str:
    sections: list[str] = []
    for chunk in chunks:
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.title
        sections.append(f"[{chunk.chunk_id}] {heading}\nURL: {chunk.canonical_url}\n{chunk.text}")
    return "\n\n---\n\n".join(sections)


def _build_inline_evidence_prompt(question: str, evidence: EvidenceBundle | list[ChunkRecord]) -> str:
    if isinstance(evidence, EvidenceBundle):
        chunks = evidence.packed_chunks
        retrieved_aspects = [query.strip() for query in evidence.retrieval_queries[1:] if query.strip()]
    else:
        chunks = evidence
        retrieved_aspects = []
    joined = _render_evidence_sections(chunks)
    aspect_block = ""
    if retrieved_aspects:
        aspect_block = "Retrieved aspects to cover when supported:\n" + "\n".join(
            f"- {query}" for query in retrieved_aspects
        ) + "\n\n"
    supporting_sources_present = any(chunk.source_family != SourceFamily.BUYERS_GUIDE for chunk in chunks)
    supporting_source_rule = (
        "If you rely on supporting policy or directive evidence, mention that briefly only where needed.\n"
        if supporting_sources_present
        else "Do not mention supporting sources unless they are needed for the answer.\n"
    )
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Requirements:\n"
        "1. Start with the direct answer to the user's actual question.\n"
        "2. Resolve every distinct part of the question explicitly.\n"
        "3. If retrieved aspects are listed below, cover each supported aspect explicitly.\n"
        "4. Preserve prerequisites, branch conditions, deadlines, exceptions, follow-on requirements, and what happens if a condition is or is not met.\n"
        "5. If the question compares two or more mechanisms, options, or scenarios, answer each one explicitly under a short heading or bullet.\n"
        "6. If the user asks for an exact identifier, contact detail, form number, template name, or file name and the evidence does not establish it, say that directly in the first sentence and then give only the closest supported context.\n"
        "7. Do not present nearby identifiers, related forms, or adjacent artifacts as the exact requested answer unless the evidence explicitly ties them to the user's request.\n"
        "8. Preserve force: if the evidence says must, only, cannot, required, or mandatory, keep that force.\n"
        "9. Use short bullets only when they help cover multiple branches or steps; otherwise keep the answer compact.\n"
        "10. Cite chunk IDs in square brackets after the statements they support.\n"
        "11. Do not say 'the provided evidence' or refer to internal retrieval mechanics in the answer.\n"
        "12. Do not end with an unfinished heading, bullet, or partial sentence.\n"
        f"13. {supporting_source_rule.strip()}\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Evidence:\n{joined}"
    )


def _build_structured_inline_evidence_prompt(question: str, chunks: list[ChunkRecord]) -> str:
    joined = _render_evidence_sections(chunks)
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Requirements:\n"
        "1. Resolve each distinct part of the user's question explicitly.\n"
        "2. Use short section headings or bullets when the question has multiple parts.\n"
        "3. State the operational rule, threshold, or decision point directly instead of hedging.\n"
        "4. If the evidence does not establish part of the answer, say that clearly.\n"
        "5. Cite chunk IDs in square brackets after the statements they support.\n"
        "6. If you use supporting policy or directive evidence, say so explicitly.\n\n"
        f"Question:\n{question}\n\nEvidence:\n{joined}"
    )


def _build_query_guided_inline_evidence_prompt(question: str, evidence: EvidenceBundle) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    aspect_lines = []
    for query in evidence.retrieval_queries[1:]:
        aspect_lines.append(f"- {query}")
    aspect_block = ""
    if aspect_lines:
        aspect_block = (
            "Retrieved aspects to cover explicitly:\n"
            + "\n".join(aspect_lines)
            + "\n\n"
        )
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Requirements:\n"
        "1. Resolve the original question completely.\n"
        "2. Explicitly address each retrieved aspect listed below if the evidence supports it.\n"
        "3. For workflow or policy questions, do not omit prerequisites, branch conditions, or what happens if a deadline or condition is not met.\n"
        "4. Use short headings or bullets when there are multiple branches or steps.\n"
        "5. If the evidence does not establish part of the answer, say that clearly.\n"
        "6. Cite chunk IDs in square brackets after the statements they support.\n"
        "7. If you use supporting policy or directive evidence, say so explicitly.\n\n"
        "8. Preserve the force of the rule: if the evidence says something is mandatory or the only permitted option, say must/only if rather than may.\n"
        "9. For exceptions or branch conditions, list every explicit condition stated in the evidence instead of summarizing loosely.\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Evidence:\n{joined}"
    )


def _build_answer_plan_prompt(question: str, evidence: EvidenceBundle) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    aspect_lines = []
    for query in evidence.retrieval_queries[1:]:
        aspect_lines.append(f"- {query}")
    aspect_block = ""
    if aspect_lines:
        aspect_block = "Retrieved aspects:\n" + "\n".join(aspect_lines) + "\n\n"
    return (
        "You are preparing an evidence-grounded answer plan for a procurement-policy RAG system.\n"
        "Use only the evidence below.\n"
        "Return JSON only in this exact shape:\n"
        '{"coverage_points":["concrete evidence-backed point"]}\n\n'
        "Rules:\n"
        "1. Include 4 to 8 coverage points unless the evidence supports fewer.\n"
        "2. Each point must capture a concrete rule, condition, exception, deadline, or outcome needed to answer the question.\n"
        "3. Preserve force: if the evidence says must, only, or cannot, keep that force.\n"
        "4. Include branch outcomes when the evidence states what happens if a condition is or is not met.\n"
        "5. Keep each point short and concrete.\n"
        "6. Do not add background points that are not needed to answer the question.\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Evidence:\n{joined}"
    )


def _build_mode_aware_answer_plan_prompt(question: str, evidence: EvidenceBundle) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    aspect_lines = []
    for query in evidence.retrieval_queries[1:]:
        aspect_lines.append(f"- {query}")
    aspect_block = ""
    if aspect_lines:
        aspect_block = "Retrieved aspects:\n" + "\n".join(aspect_lines) + "\n\n"
    return (
        "You are preparing an evidence-grounded answer plan for a procurement-policy RAG system.\n"
        "Use only the evidence below.\n"
        "Return JSON only in this exact shape:\n"
        '{"answer_mode":"workflow","should_abstain":false,"abstain_reason":"","coverage_points":["concrete evidence-backed point"]}\n\n'
        "Allowed values for answer_mode are:\n"
        '- "workflow"\n'
        '- "navigation"\n'
        '- "missing_detail"\n'
        '- "direct_rule"\n\n'
        "Rules:\n"
        "1. Choose navigation when the user is mainly asking where to go, where to start, or which page/path contains the rule.\n"
        "2. Choose missing_detail when the user asks for an exact contact detail, exact file name, exact form number, or similarly precise identifier that the evidence does not actually provide.\n"
        "3. Choose workflow when the user asks about options, branches, exceptions, deadlines, consequences, or what happens if some but not all conditions hold.\n"
        "4. Choose direct_rule for straightforward rule, threshold, or authority questions that do not need a more specialized frame.\n"
        "5. Set should_abstain to true only when the exact requested detail or determination is not established by the evidence.\n"
        "6. If the evidence gives a category, template family, page URL, or surrounding context but not the exact requested identifier, treat the exact identifier as unavailable and explain that in abstain_reason.\n"
        "7. coverage_points must contain 2 to 8 concrete evidence-backed points unless the evidence supports fewer.\n"
        "8. Each coverage point must capture a rule, condition, exception, branch outcome, navigation destination, or abstention-supporting fact needed for the answer.\n"
        "9. Preserve force: if the evidence says must, only, cannot, or required, keep that force.\n"
        "10. For workflow questions, include all explicit branches or outcomes needed to answer the question completely.\n"
        "11. For navigation questions, include the exact page or path the user should go to and, when relevant, the child page that contains the actual rule.\n"
        "12. Do not invent background points that are not needed to answer the user's question.\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Evidence:\n{joined}"
    )


def _build_compact_mode_aware_answer_plan_prompt(question: str, evidence: EvidenceBundle) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    aspect_lines = []
    for query in evidence.retrieval_queries[1:]:
        aspect_lines.append(f"- {query}")
    aspect_block = ""
    if aspect_lines:
        aspect_block = "Retrieved aspects:\n" + "\n".join(aspect_lines) + "\n\n"
    return (
        "You are preparing an evidence-grounded concise answer plan for a procurement-policy RAG system.\n"
        "Use only the evidence below.\n"
        "Return JSON only in this exact shape:\n"
        '{"answer_mode":"workflow","should_abstain":false,"abstain_reason":"","coverage_points":["indispensable evidence-backed point"]}\n\n'
        "Allowed values for answer_mode are:\n"
        '- "workflow"\n'
        '- "navigation"\n'
        '- "missing_detail"\n'
        '- "direct_rule"\n\n'
        "Rules:\n"
        "1. Choose navigation when the user is mainly asking where to go, where to start, or which page/path contains the rule.\n"
        "2. Choose missing_detail when the user asks for an exact contact detail, exact file name, exact form number, or similarly precise identifier that the evidence does not actually provide.\n"
        "3. Choose workflow when the user asks about options, branches, exceptions, deadlines, consequences, or what happens if some but not all conditions hold.\n"
        "4. Choose direct_rule for straightforward rule, threshold, or authority questions that do not need a more specialized frame.\n"
        "5. Set should_abstain to true only when the exact requested detail or determination is not established by the evidence.\n"
        "6. coverage_points must contain 2 to 6 indispensable evidence-backed points unless the evidence supports fewer.\n"
        "7. Include only points that are necessary to answer the user's actual question completely.\n"
        "8. Do not include supported but optional background, implementation detail, escalation detail, or policy context unless omitting it would make the answer materially incomplete or misleading.\n"
        "9. Preserve force: if the evidence says must, only, cannot, or required, keep that force.\n"
        "10. For workflow questions, include the essential branches, deadlines, exceptions, and consequences needed to answer the question correctly.\n"
        "11. If the evidence states a prerequisite timing condition, expiry consequence, or condition that changes what action is permitted, include it for workflow mode.\n"
        "12. If the evidence includes a required file, documentation, notification, publication, or other follow-on step that the buyer still must do, include it for workflow mode.\n"
        "13. Include consultation or approval steps only when they materially change what the buyer may do, not as substitute background.\n"
        "14. Do not include side-process examples, escalation paths, or broad exception lists unless the question asks for them or omitting them would change the correct action.\n"
        "15. For navigation questions, include only the exact stage, page, child page, or URL needed to get the user to the rule.\n"
        "16. For missing-detail questions, include the abstention-supporting fact and the closest supported context, but prefer the general method, page, directory, list, or locator over a long list of examples unless the examples are necessary.\n"
        "17. Do not invent background points that are not needed to answer the user's question.\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Evidence:\n{joined}"
    )


def _allowed_contract_slots(answer_mode: str) -> tuple[str, ...]:
    slot_map = {
        "workflow": (
            "bottom_line",
            "prerequisite_or_scope",
            "general_rule",
            "branch_if_all",
            "branch_if_some",
            "branch_if_none",
            "required_document_or_input",
            "deadline_or_timing",
            "consequence",
            "follow_on_requirement",
            "exception",
        ),
        "navigation": (
            "start_page",
            "parent_stage",
            "child_page_with_rule",
            "what_that_page_covers",
            "direct_url",
        ),
        "missing_detail": (
            "exact_detail_status",
            "closest_supported_context",
            "page_or_location",
            "supporting_rule",
        ),
        "direct_rule": (
            "bottom_line",
            "general_rule",
            "conditions",
            "exception",
            "consequence",
        ),
    }
    return slot_map.get(answer_mode, slot_map["direct_rule"])


def _humanize_contract_slot(slot_name: str) -> str:
    return slot_name.replace("_", " ").capitalize()


def _build_structured_answer_contract_prompt(question: str, evidence: EvidenceBundle) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    aspect_lines = []
    for query in evidence.retrieval_queries[1:]:
        aspect_lines.append(f"- {query}")
    aspect_block = ""
    if aspect_lines:
        aspect_block = "Retrieved aspects:\n" + "\n".join(aspect_lines) + "\n\n"
    return (
        "You are preparing an evidence-grounded structured answer contract for a procurement-policy RAG system.\n"
        "Use only the evidence below.\n"
        "Return JSON only in this exact top-level shape:\n"
        '{"answer_mode":"workflow","should_abstain":false,"abstain_reason":"","slots":{"bottom_line":"","prerequisite_or_scope":"","general_rule":"","branch_if_all":"","branch_if_some":"","branch_if_none":"","required_document_or_input":"","deadline_or_timing":"","consequence":"","follow_on_requirement":"","exception":""}}\n\n'
        "Allowed values for answer_mode are:\n"
        '- "workflow"\n'
        '- "navigation"\n'
        '- "missing_detail"\n'
        '- "direct_rule"\n\n'
        "Allowed slot keys by mode:\n"
        "- workflow: bottom_line, prerequisite_or_scope, general_rule, branch_if_all, branch_if_some, branch_if_none, required_document_or_input, deadline_or_timing, consequence, follow_on_requirement, exception\n"
        "- navigation: start_page, parent_stage, child_page_with_rule, what_that_page_covers, direct_url\n"
        "- missing_detail: exact_detail_status, closest_supported_context, page_or_location, supporting_rule\n"
        "- direct_rule: bottom_line, general_rule, conditions, exception, consequence\n\n"
        "Rules:\n"
        "1. Choose navigation when the user mainly wants to know where in the Buyer's Guide to go.\n"
        "2. Choose missing_detail when the user asks for an exact identifier or contact detail that the evidence may not provide.\n"
        "3. Choose workflow when the user asks about options, branches, exceptions, deadlines, or consequences.\n"
        "4. Choose direct_rule for straightforward rule or authority questions that do not need a more specialized frame.\n"
        "5. Populate only slot keys allowed for the chosen answer_mode.\n"
        "6. Leave unsupported or unstated slots as empty strings.\n"
        "7. Every non-empty slot must be directly supported by the evidence.\n"
        "8. Preserve force: if the evidence says must, only, cannot, or required, keep that force.\n"
        "9. For workflow mode, use prerequisite_or_scope for threshold tests, applicability determinations, required forms, required certifications, or required authority preconditions.\n"
        "10. For workflow mode, fill separate branch slots when the evidence distinguishes all, some, none, exceptions, deadlines, or consequences.\n"
        "11. For workflow mode, use follow_on_requirement for obligations that still must be carried out after the main decision, such as notifications, file documentation, or publication steps.\n"
        "12. For workflow mode, use required_document_or_input for named forms, certifications, declarations, proof, or supporting evidence when they are part of the answer.\n"
        "13. For navigation mode, identify the exact page or path the user should open and the child page with the actual rule when relevant.\n"
        "14. For missing_detail mode, use exact_detail_status to say whether the exact requested detail is available, and use closest_supported_context for the nearest supported substitute context.\n"
        "15. Set should_abstain to true only when the exact requested detail or determination is not established by the evidence.\n"
        "16. Do not invent slot content just to fill the contract.\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Evidence:\n{joined}"
    )


def _build_cited_structured_answer_contract_prompt(question: str, evidence: EvidenceBundle) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    aspect_lines = []
    for query in evidence.retrieval_queries[1:]:
        aspect_lines.append(f"- {query}")
    aspect_block = ""
    if aspect_lines:
        aspect_block = "Retrieved aspects:\n" + "\n".join(aspect_lines) + "\n\n"
    return (
        "You are preparing an evidence-grounded structured answer contract for a procurement-policy RAG system.\n"
        "Use only the evidence below.\n"
        "Return JSON only in this exact top-level shape:\n"
        '{"answer_mode":"workflow","should_abstain":false,"abstain_reason":"","slots":{"bottom_line":{"text":"","citation_chunk_ids":["chunk_id"]},"prerequisite_or_scope":{"text":"","citation_chunk_ids":["chunk_id"]},"general_rule":{"text":"","citation_chunk_ids":["chunk_id"]},"branch_if_all":{"text":"","citation_chunk_ids":["chunk_id"]},"branch_if_some":{"text":"","citation_chunk_ids":["chunk_id"]},"branch_if_none":{"text":"","citation_chunk_ids":["chunk_id"]},"required_document_or_input":{"text":"","citation_chunk_ids":["chunk_id"]},"deadline_or_timing":{"text":"","citation_chunk_ids":["chunk_id"]},"consequence":{"text":"","citation_chunk_ids":["chunk_id"]},"follow_on_requirement":{"text":"","citation_chunk_ids":["chunk_id"]},"exception":{"text":"","citation_chunk_ids":["chunk_id"]}}}\n\n'
        "Allowed values for answer_mode are:\n"
        '- "workflow"\n'
        '- "navigation"\n'
        '- "missing_detail"\n'
        '- "direct_rule"\n\n'
        "Allowed slot keys by mode:\n"
        "- workflow: bottom_line, prerequisite_or_scope, general_rule, branch_if_all, branch_if_some, branch_if_none, required_document_or_input, deadline_or_timing, consequence, follow_on_requirement, exception\n"
        "- navigation: start_page, parent_stage, child_page_with_rule, what_that_page_covers, direct_url\n"
        "- missing_detail: exact_detail_status, closest_supported_context, page_or_location, supporting_rule\n"
        "- direct_rule: bottom_line, general_rule, conditions, exception, consequence\n\n"
        "Rules:\n"
        "1. Choose navigation when the user mainly wants to know where in the Buyer's Guide to go.\n"
        "2. Choose missing_detail when the user asks for an exact identifier or contact detail that the evidence may not provide.\n"
        "3. Choose workflow when the user asks about options, branches, exceptions, deadlines, or consequences.\n"
        "4. Choose direct_rule for straightforward rule or authority questions that do not need a more specialized frame.\n"
        "5. Populate only slot keys allowed for the chosen answer_mode.\n"
        "6. Every non-empty slot must include a short text field and 1 to 3 citation_chunk_ids from the evidence that directly support that slot.\n"
        "7. Leave unsupported or unstated slots as empty strings with empty citation_chunk_ids lists.\n"
        "8. Every non-empty slot must be directly supported by the cited evidence.\n"
        "9. Preserve force: if the evidence says must, only, cannot, or required, keep that force.\n"
        "10. For workflow mode, use prerequisite_or_scope for threshold tests, applicability determinations, required forms, required certifications, or required authority preconditions.\n"
        "11. For workflow mode, fill separate branch slots when the evidence distinguishes all, some, none, exceptions, deadlines, or consequences.\n"
        "12. For workflow mode, use follow_on_requirement for obligations that still must be carried out after the main decision, such as notifications, file documentation, or publication steps.\n"
        "13. For workflow mode, use required_document_or_input for named forms, certifications, declarations, proof, or supporting evidence when they are part of the answer.\n"
        "14. For navigation mode, identify the exact page or path the user should open and the child page with the actual rule when relevant.\n"
        "15. For missing_detail mode, use exact_detail_status to say whether the exact requested detail is available, and use closest_supported_context for the nearest supported substitute context.\n"
        "16. Set should_abstain to true only when the exact requested detail or determination is not established by the evidence.\n"
        "17. Each non-empty slot must be concise: prefer one short sentence or phrase, ideally under 30 words.\n"
        "18. Populate only the slots needed to answer the user's actual question completely; leave marginal or background slots empty.\n"
        "19. For workflow mode, prefer 4 to 6 non-empty slots unless more are strictly necessary for correctness.\n"
        "20. If the question asks about a specific branch or scenario, fill the directly relevant branch slots and leave unrelated branches empty unless they are required for the answer.\n"
        "21. Do not invent slot content or citation IDs just to fill the contract.\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Evidence:\n{joined}"
    )


def _build_answer_repair_plan_prompt(
    question: str,
    evidence: EvidenceBundle,
    answer_plan: ModeAwareAnswerPlan,
    draft_answer: str,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(answer_plan.coverage_points)
    mode_block = f"Answer mode: {answer_plan.answer_mode}\nShould abstain: {str(answer_plan.should_abstain).lower()}\n"
    return (
        "You are auditing an evidence-grounded procurement-policy answer for completeness and support.\n"
        "Use only the evidence below.\n"
        "Return JSON only in this exact shape:\n"
        '{"needs_revision":false,"missing_supported_points":["point"],"unsupported_or_overstated_points":["point"]}\n\n'
        "Rules:\n"
        "1. Compare the draft answer against the original question, the coverage checklist, and the evidence.\n"
        "2. Treat a checklist point as missing if it is absent, only partially covered, or implied too vaguely to count as a clear answer to that point.\n"
        "3. Add a point to missing_supported_points only if it is directly supported by the evidence and materially missing from the draft answer.\n"
        "4. Add a point to unsupported_or_overstated_points only if the draft answer states something not supported by the evidence or stronger than the evidence supports.\n"
        "5. Do not list points that are already covered adequately and explicitly.\n"
        "6. For workflow mode, treat missed branches, deadlines, exceptions, or consequences as missing even if the draft answer gives the main rule.\n"
        "7. For navigation mode, treat the page/path, parent stage, child page, and direct URL as separate requirements when they appear in the checklist.\n"
        "8. For missing_detail mode, if should_abstain is true, the draft answer must explicitly say the exact detail is not established and then provide the closest supported context.\n"
        "9. Keep each point short and concrete.\n"
        "10. If the answer is already complete and supported, set needs_revision to false and return empty lists.\n\n"
        f"Original question:\n{question}\n\n"
        f"{mode_block}\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_answer_rewrite_verdict_prompt(
    question: str,
    evidence: EvidenceBundle,
    draft_answer: str,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    aspect_lines = []
    for query in evidence.retrieval_queries[1:]:
        aspect_lines.append(f"- {query}")
    aspect_block = ""
    if aspect_lines:
        aspect_block = "Retrieved aspects:\n" + "\n".join(aspect_lines) + "\n\n"
    return (
        "You are verifying whether a baseline procurement-policy answer draft should be kept as-is or rewritten "
        "through a stricter structured-contract answer path.\n"
        "Use only the evidence below.\n"
        "Return JSON only in this exact shape:\n"
        '{"action":"keep","confidence":"low","rationale":"short explanation","omission_risk":false,'
        '"exact_detail_abstain_risk":false,"unsupported_detail_risk":false}\n\n'
        "Allowed values for action are:\n"
        '- "keep"\n'
        '- "rewrite_structured_contract"\n\n'
        "Allowed values for confidence are:\n"
        '- "low"\n'
        '- "medium"\n'
        '- "high"\n\n'
        "Rules:\n"
        "1. Compare the draft answer against the original question and the evidence only.\n"
        "2. Choose rewrite_structured_contract only when the draft likely failed in a way that a structured "
        "evidence-bound rewrite would clearly help.\n"
        "3. Strong reasons to rewrite are:\n"
        "   - omission of indispensable workflow branches, deadlines, consequences, or follow-on requirements\n"
        "   - failure to clearly abstain when the user asks for an exact detail the evidence does not establish\n"
        "   - materially unsupported or overstated answer content that a stricter evidence-bound rewrite would likely remove\n"
        "4. If the answer is already materially adequate, or if the issue is only extra but supported background detail, choose keep.\n"
        "5. Be conservative: if you are unsure, choose keep with low confidence.\n"
        "6. Do not propose missing points or rewrite text yourself. Only return the verdict.\n"
        "7. exact_detail_abstain_risk should be true only when the question asks for a precise identifier, contact detail, file name, form number, or similarly exact detail that the evidence does not actually provide and the draft fails to handle that correctly.\n"
        "8. omission_risk should be true only when the draft likely omits indispensable supported content needed to answer the question correctly.\n"
        "9. unsupported_detail_risk should be true only when the draft states something materially stronger or broader than the evidence supports.\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_contract_aware_answer_rewrite_verdict_prompt(
    question: str,
    evidence: EvidenceBundle,
    contract: CitedStructuredAnswerContract,
    draft_answer: str,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    contract_checklist = _render_cited_contract_checklist(contract)
    aspect_lines = []
    for query in evidence.retrieval_queries[1:]:
        aspect_lines.append(f"- {query}")
    aspect_block = ""
    if aspect_lines:
        aspect_block = "Retrieved aspects:\n" + "\n".join(aspect_lines) + "\n\n"
    return (
        "You are verifying whether a baseline procurement-policy answer draft should be kept as-is or rewritten "
        "through a stricter structured-contract answer path.\n"
        "Use only the evidence and independently extracted contract below.\n"
        "Return JSON only in this exact shape:\n"
        '{"action":"keep","confidence":"low","rationale":"short explanation","omission_risk":false,'
        '"exact_detail_abstain_risk":false,"unsupported_detail_risk":false}\n\n'
        "Allowed values for action are:\n"
        '- "keep"\n'
        '- "rewrite_structured_contract"\n\n'
        "Allowed values for confidence are:\n"
        '- "low"\n'
        '- "medium"\n'
        '- "high"\n\n'
        "Rules:\n"
        "1. Compare the draft answer against the original question, the structured contract, and the evidence.\n"
        "2. Treat each populated contract slot as a supported answer obligation for this question, but do not require the draft to match slot wording verbatim.\n"
        "3. Choose rewrite_structured_contract when the draft omits, weakens, or obscures a materially important populated contract slot in a way that changes the answer.\n"
        "4. For workflow contracts, missed branches, deadlines, consequences, follow-on requirements, or exceptions are strong reasons to rewrite.\n"
        "5. For missing-detail contracts with should_abstain=true, rewrite when the draft fails to clearly state that the exact requested detail is not established and then give the closest supported context.\n"
        "6. If the draft already covers the contract materially and any extra content is merely supported background, choose keep.\n"
        "7. unsupported_detail_risk should be true only when the draft states something materially stronger or broader than the evidence or contract support.\n"
        "8. Be conservative about style differences, but not conservative about substantive omissions of populated contract slots.\n"
        "9. Do not propose missing points or rewrite text yourself. Only return the verdict.\n\n"
        f"Original question:\n{question}\n\n"
        f"{aspect_block}"
        f"Structured answer contract:\n{contract_checklist}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_missing_detail_exactness_verdict_prompt(
    question: str,
    evidence: EvidenceBundle,
    contract: CitedStructuredAnswerContract,
    draft_answer: str,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    contract_checklist = _render_cited_contract_checklist(contract)
    return (
        "You are checking a missing-detail procurement answer for misleading exact-detail claims.\n"
        "Use only the question, the structured contract, the draft answer, and the evidence below.\n"
        "Return JSON only in this exact shape:\n"
        '{"confidence":"low","rationale":"short explanation","exact_detail_overstatement_risk":false,'
        '"offending_details":["detail"]}\n\n'
        "Rules:\n"
        "1. This check is only about whether the draft presents a nearby identifier, form, file name, template name, "
        "contact method, URL, or other detail too strongly as the exact requested answer.\n"
        "2. Set exact_detail_overstatement_risk to true when the draft says the exact detail is unavailable but then "
        "suggests a nearby detail as if it is likely the exact answer, or when it presents a contextual identifier or "
        "template name as the requested exact detail.\n"
        "3. Do not flag a draft merely for giving neutral closest-supported context after a clear abstention.\n"
        "4. For contact-detail questions, general procedures, directories, lists, or locator URLs are not by themselves "
        "exact-detail overstatements.\n"
        "5. For form-number or file-name questions, naming a related form, template, or document from a broader "
        "workflow can still be an overstatement if the evidence does not establish that it is the exact requested artifact.\n"
        "6. offending_details should list only the specific over-strong details, not general prose.\n"
        "7. If the draft stays within the contract's exact_detail_status and closest_supported_context without implying "
        "more precision than the evidence supports, set exact_detail_overstatement_risk to false.\n\n"
        f"Original question:\n{question}\n\n"
        f"Structured answer contract:\n{contract_checklist}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_contract_slot_coverage_verdict_prompt(
    question: str,
    evidence: EvidenceBundle,
    contract: CitedStructuredAnswerContract,
    draft_answer: str,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    contract_checklist = _render_cited_contract_checklist(contract)
    return (
        "You are checking how well a procurement-policy draft answer covers an independently extracted structured contract.\n"
        "Use only the contract and evidence below.\n"
        "Return JSON only in this exact shape:\n"
        '{"confidence":"low","rationale":"short explanation","missing_or_weakened_slots":["slot_key"],"unsupported_detail_risk":false}\n\n'
        "Allowed values for confidence are:\n"
        '- "low"\n'
        '- "medium"\n'
        '- "high"\n\n'
        "Rules:\n"
        "1. Only use slot keys that are populated in the structured contract.\n"
        "2. Add a slot to missing_or_weakened_slots when the draft answer omits it, only partially covers it, or weakens it enough to change the answer.\n"
        "3. Do not mark slots just because the draft uses different wording or a more concise phrasing.\n"
        "4. For workflow contracts, missed branches, deadlines, consequences, follow-on requirements, and exceptions are especially important.\n"
        "5. For missing-detail contracts with should_abstain=true, mark the relevant populated slots if the draft fails to clearly say the exact detail is not established and then give the closest supported context.\n"
        "6. unsupported_detail_risk should be true only when the draft states something materially stronger or broader than the evidence or contract support.\n"
        "7. Be willing to mark substantive omissions even if the draft has the general theme correct.\n\n"
        f"Original question:\n{question}\n\n"
        f"Structured answer contract:\n{contract_checklist}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Evidence:\n{joined}"
    )


def _normalize_string_list(raw_items: object, *, max_items: int = 8) -> list[str]:
    if not isinstance(raw_items, list):
        return []
    items: list[str] = []
    seen_items: set[str] = set()
    for item in raw_items:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.split()).strip()
        if not normalized:
            continue
        dedupe_key = normalized.lower()
        if dedupe_key in seen_items:
            continue
        items.append(normalized)
        seen_items.add(dedupe_key)
        if len(items) >= max_items:
            break
    return items


def _normalize_answer_plan(raw_text: str, *, max_points: int = 8) -> list[str]:
    parsed = json.loads(raw_text)
    points = _normalize_string_list(parsed.get("coverage_points", []), max_items=max_points)
    if not points:
        raise ValueError("Answer plan did not contain any usable coverage points")
    return points


def _normalize_mode_aware_answer_plan(raw_text: str, *, max_points: int = 8) -> ModeAwareAnswerPlan:
    parsed = json.loads(raw_text)
    answer_mode = str(parsed.get("answer_mode", "direct_rule")).strip().lower()
    if answer_mode not in {"workflow", "navigation", "missing_detail", "direct_rule"}:
        answer_mode = "direct_rule"
    should_abstain = bool(parsed.get("should_abstain", False))
    abstain_reason = " ".join(str(parsed.get("abstain_reason", "")).split()).strip()
    coverage_points = _normalize_answer_plan(
        json.dumps({"coverage_points": parsed.get("coverage_points", [])}),
        max_points=max_points,
    )
    return ModeAwareAnswerPlan(
        answer_mode=answer_mode,
        should_abstain=should_abstain,
        abstain_reason=abstain_reason,
        coverage_points=coverage_points,
    )


def _normalize_structured_answer_contract(raw_text: str) -> StructuredAnswerContract:
    parsed = json.loads(raw_text)
    answer_mode = str(parsed.get("answer_mode", "direct_rule")).strip().lower()
    if answer_mode not in {"workflow", "navigation", "missing_detail", "direct_rule"}:
        answer_mode = "direct_rule"
    should_abstain = bool(parsed.get("should_abstain", False))
    abstain_reason = " ".join(str(parsed.get("abstain_reason", "")).split()).strip()
    raw_slots = parsed.get("slots", {})
    if not isinstance(raw_slots, dict):
        raise ValueError("Structured answer contract must include a slots object")
    slots: dict[str, str] = {}
    for key in _allowed_contract_slots(answer_mode):
        value = raw_slots.get(key, "")
        if not isinstance(value, str):
            continue
        normalized = " ".join(value.split()).strip()
        if normalized:
            slots[key] = normalized
    if not slots:
        raise ValueError("Structured answer contract did not contain any usable slots")
    return StructuredAnswerContract(
        answer_mode=answer_mode,
        should_abstain=should_abstain,
        abstain_reason=abstain_reason,
        slots=slots,
    )


def _normalize_citation_chunk_ids(raw_ids: object, *, max_items: int = 3) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for item in raw_ids:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized or normalized in seen_ids:
            continue
        normalized_ids.append(normalized)
        seen_ids.add(normalized)
        if len(normalized_ids) >= max_items:
            break
    return normalized_ids


def _normalize_cited_structured_answer_contract(raw_text: str) -> CitedStructuredAnswerContract:
    parsed = json.loads(raw_text)
    answer_mode = str(parsed.get("answer_mode", "direct_rule")).strip().lower()
    if answer_mode not in {"workflow", "navigation", "missing_detail", "direct_rule"}:
        answer_mode = "direct_rule"
    should_abstain = bool(parsed.get("should_abstain", False))
    abstain_reason = " ".join(str(parsed.get("abstain_reason", "")).split()).strip()
    raw_slots = parsed.get("slots", {})
    if not isinstance(raw_slots, dict):
        raise ValueError("Cited structured answer contract must include a slots object")
    slots: dict[str, StructuredAnswerSlotValue] = {}
    for key in _allowed_contract_slots(answer_mode):
        raw_value = raw_slots.get(key, "")
        if isinstance(raw_value, str):
            normalized_text = " ".join(raw_value.split()).strip()
            normalized_citations: list[str] = []
        elif isinstance(raw_value, dict):
            normalized_text = " ".join(str(raw_value.get("text", "")).split()).strip()
            normalized_citations = _normalize_citation_chunk_ids(raw_value.get("citation_chunk_ids", []))
        else:
            continue
        if normalized_text:
            slots[key] = StructuredAnswerSlotValue(
                text=normalized_text,
                citation_chunk_ids=normalized_citations,
            )
    if not slots:
        raise ValueError("Cited structured answer contract did not contain any usable slots")
    return CitedStructuredAnswerContract(
        answer_mode=answer_mode,
        should_abstain=should_abstain,
        abstain_reason=abstain_reason,
        slots=slots,
    )


def _normalize_answer_rewrite_verdict_payload(
    payload: AnswerRewriteVerdictPayload,
) -> AnswerRewriteVerdict:
    action = payload.action.strip().lower()
    if action not in {"keep", "rewrite_structured_contract"}:
        action = "keep"
    confidence = payload.confidence.strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    rationale = " ".join(payload.rationale.split()).strip()
    return AnswerRewriteVerdict(
        action=action,
        confidence=confidence,
        rationale=rationale,
        omission_risk=payload.omission_risk,
        exact_detail_abstain_risk=payload.exact_detail_abstain_risk,
        unsupported_detail_risk=payload.unsupported_detail_risk,
    )


def _normalize_missing_detail_exactness_verdict_payload(
    payload: MissingDetailExactnessVerdictPayload,
) -> MissingDetailExactnessVerdict:
    confidence = payload.confidence.strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    rationale = " ".join(payload.rationale.split()).strip()
    return MissingDetailExactnessVerdict(
        confidence=confidence,
        rationale=rationale,
        exact_detail_overstatement_risk=payload.exact_detail_overstatement_risk,
        offending_details=_normalize_string_list(payload.offending_details, max_items=6),
    )


def _normalize_contract_slot_coverage_verdict_payload(
    payload: ContractSlotCoverageVerdictPayload,
    *,
    answer_mode: str,
    populated_slot_keys: set[str],
) -> ContractSlotCoverageVerdict:
    confidence = payload.confidence.strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    rationale = " ".join(payload.rationale.split()).strip()
    allowed_slots = set(_allowed_contract_slots(answer_mode)) & populated_slot_keys
    missing_or_weakened_slots: list[str] = []
    seen_slots: set[str] = set()
    for slot in _normalize_string_list(payload.missing_or_weakened_slots, max_items=16):
        if slot not in allowed_slots or slot in seen_slots:
            continue
        missing_or_weakened_slots.append(slot)
        seen_slots.add(slot)
        if len(missing_or_weakened_slots) >= max(len(allowed_slots), 1):
            break
    return ContractSlotCoverageVerdict(
        confidence=confidence,
        rationale=rationale,
        missing_or_weakened_slots=missing_or_weakened_slots,
        unsupported_detail_risk=payload.unsupported_detail_risk,
    )


def _normalize_cited_structured_answer_contract_payload(
    payload: CitedStructuredAnswerContractPayload,
) -> CitedStructuredAnswerContract:
    answer_mode = payload.answer_mode.strip().lower()
    if answer_mode not in {"workflow", "navigation", "missing_detail", "direct_rule"}:
        answer_mode = "direct_rule"
    abstain_reason = " ".join(payload.abstain_reason.split()).strip()
    slots: dict[str, StructuredAnswerSlotValue] = {}
    for key in _allowed_contract_slots(answer_mode):
        raw_value = payload.slots.get(key)
        if raw_value is None:
            continue
        normalized_text = " ".join(raw_value.text.split()).strip()
        normalized_citations = _normalize_citation_chunk_ids(raw_value.citation_chunk_ids)
        if normalized_text:
            slots[key] = StructuredAnswerSlotValue(
                text=normalized_text,
                citation_chunk_ids=normalized_citations,
            )
    if not slots:
        raise ValueError("Cited structured answer contract did not contain any usable slots")
    return CitedStructuredAnswerContract(
        answer_mode=answer_mode,
        should_abstain=payload.should_abstain,
        abstain_reason=abstain_reason,
        slots=slots,
    )


def _extract_cited_structured_answer_contract(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> tuple[CitedStructuredAnswerContract, CitedStructuredAnswerContractPayload]:
    settings.require_cohere_key("Structured contract extraction")
    client = from_cohere(cohere.ClientV2(settings.cohere_api_key))
    contract_payload = client.create(
        response_model=CitedStructuredAnswerContractPayload,
        messages=[{"role": "user", "content": _build_cited_structured_answer_contract_prompt(question, evidence)}],
        model=settings.cohere_query_planner_model,
        temperature=0,
        max_tokens=1400,
        max_retries=2,
    )
    contract = _normalize_cited_structured_answer_contract_payload(contract_payload)
    return contract, contract_payload


def _extract_answer_rewrite_verdict(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
    draft_answer: str,
) -> tuple[AnswerRewriteVerdict, AnswerRewriteVerdictPayload]:
    settings.require_cohere_key("Answer rewrite verification")
    client = from_cohere(cohere.ClientV2(settings.cohere_api_key))
    payload = client.create(
        response_model=AnswerRewriteVerdictPayload,
        messages=[
            {
                "role": "user",
                "content": _build_answer_rewrite_verdict_prompt(question, evidence, draft_answer),
            }
        ],
        model=settings.cohere_query_planner_model,
        temperature=0,
        max_tokens=260,
        max_retries=2,
    )
    return _normalize_answer_rewrite_verdict_payload(payload), payload


def _extract_contract_aware_answer_rewrite_verdict(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
    contract: CitedStructuredAnswerContract,
    draft_answer: str,
) -> tuple[AnswerRewriteVerdict, AnswerRewriteVerdictPayload]:
    settings.require_cohere_key("Contract-aware answer rewrite verification")
    client = from_cohere(cohere.ClientV2(settings.cohere_api_key))
    payload = client.create(
        response_model=AnswerRewriteVerdictPayload,
        messages=[
            {
                "role": "user",
                "content": _build_contract_aware_answer_rewrite_verdict_prompt(
                    question,
                    evidence,
                    contract,
                    draft_answer,
                ),
            }
        ],
        model=settings.cohere_query_planner_model,
        temperature=0,
        max_tokens=260,
        max_retries=2,
    )
    return _normalize_answer_rewrite_verdict_payload(payload), payload


def _extract_contract_slot_coverage_verdict(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
    contract: CitedStructuredAnswerContract,
    draft_answer: str,
) -> tuple[ContractSlotCoverageVerdict, ContractSlotCoverageVerdictPayload]:
    settings.require_cohere_key("Contract slot coverage verification")
    client = from_cohere(cohere.ClientV2(settings.cohere_api_key))
    payload = client.create(
        response_model=ContractSlotCoverageVerdictPayload,
        messages=[
            {
                "role": "user",
                "content": _build_contract_slot_coverage_verdict_prompt(
                    question,
                    evidence,
                    contract,
                    draft_answer,
                ),
            }
        ],
        model=settings.cohere_query_planner_model,
        temperature=0,
        max_tokens=260,
        max_retries=2,
    )
    verdict = _normalize_contract_slot_coverage_verdict_payload(
        payload,
        answer_mode=contract.answer_mode,
        populated_slot_keys=set(contract.slots),
    )
    return verdict, payload


def _extract_missing_detail_exactness_verdict(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
    contract: CitedStructuredAnswerContract,
    draft_answer: str,
) -> tuple[MissingDetailExactnessVerdict, MissingDetailExactnessVerdictPayload]:
    settings.require_cohere_key("Missing-detail exactness verification")
    client = from_cohere(cohere.ClientV2(settings.cohere_api_key))
    payload = client.create(
        response_model=MissingDetailExactnessVerdictPayload,
        messages=[
            {
                "role": "user",
                "content": _build_missing_detail_exactness_verdict_prompt(
                    question,
                    evidence,
                    contract,
                    draft_answer,
                ),
            }
        ],
        model=settings.cohere_query_planner_model,
        temperature=0,
        max_tokens=220,
        max_retries=2,
    )
    return _normalize_missing_detail_exactness_verdict_payload(payload), payload


def _normalize_answer_repair_plan(raw_text: str, *, max_points: int = 6) -> AnswerRepairPlan:
    parsed = json.loads(raw_text)
    needs_revision = bool(parsed.get("needs_revision", False))
    missing_supported_points = _normalize_string_list(
        parsed.get("missing_supported_points", []),
        max_items=max_points,
    )
    unsupported_or_overstated_points = _normalize_string_list(
        parsed.get("unsupported_or_overstated_points", []),
        max_items=max_points,
    )
    if missing_supported_points or unsupported_or_overstated_points:
        needs_revision = True
    return AnswerRepairPlan(
        needs_revision=needs_revision,
        missing_supported_points=missing_supported_points,
        unsupported_or_overstated_points=unsupported_or_overstated_points,
    )


def _render_plan_checklist(plan_points: list[str]) -> str:
    return "\n".join(f"- {point}" for point in plan_points)


def _render_cited_contract_checklist(contract: CitedStructuredAnswerContract) -> str:
    lines = [
        f"answer_mode: {contract.answer_mode}",
        f"should_abstain: {str(contract.should_abstain).lower()}",
    ]
    if contract.abstain_reason:
        lines.append(f"abstain_reason: {contract.abstain_reason}")
    lines.append("slots:")
    for key, value in contract.slots.items():
        citation_suffix = ""
        if value.citation_chunk_ids:
            citation_suffix = f" [{' '.join(value.citation_chunk_ids)}]"
        lines.append(f"- {key}: {value.text}{citation_suffix}")
    return "\n".join(lines)


def _render_structured_contract_slots(contract: StructuredAnswerContract) -> str:
    lines: list[str] = []
    for key in _allowed_contract_slots(contract.answer_mode):
        value = contract.slots.get(key)
        if value:
            lines.append(f"- {_humanize_contract_slot(key)}: {value}")
    return "\n".join(lines)


def _render_text_with_chunk_ids(text: str, chunk_ids: list[str]) -> str:
    del chunk_ids
    return text


def _core_contract_slot_keys(contract: CitedStructuredAnswerContract) -> set[str]:
    if contract.answer_mode == "workflow":
        return {"bottom_line"} & set(contract.slots)
    if contract.answer_mode == "navigation":
        return {"start_page"} & set(contract.slots)
    if contract.answer_mode == "missing_detail":
        return {"exact_detail_status"} & set(contract.slots)
    return {"bottom_line"} & set(contract.slots)


def _minimal_missing_detail_exactness_keep_set(
    contract: CitedStructuredAnswerContract,
    *,
    selector_keep_slot_keys: set[str] | None,
    missing_slots: set[str],
) -> set[str]:
    selector_keep_slot_keys = selector_keep_slot_keys or set()
    keep = _core_contract_slot_keys(contract)
    if "closest_supported_context" in contract.slots:
        keep.add("closest_supported_context")
    if "page_or_location" in contract.slots and (
        "page_or_location" in selector_keep_slot_keys or "page_or_location" in missing_slots
    ):
        keep.add("page_or_location")
    return keep


def _prune_cited_structured_answer_contract(
    contract: CitedStructuredAnswerContract,
    *,
    keep_slot_keys: set[str],
) -> CitedStructuredAnswerContract:
    retained_keys = [
        key
        for key in _allowed_contract_slots(contract.answer_mode)
        if key in contract.slots and key in keep_slot_keys
    ]
    if not retained_keys:
        return contract
    return CitedStructuredAnswerContract(
        answer_mode=contract.answer_mode,
        should_abstain=contract.should_abstain,
        abstain_reason=contract.abstain_reason,
        slots={key: contract.slots[key] for key in retained_keys},
    )
def _looks_like_missing_detail_abstention(answer_text: str) -> bool:
    normalized = " ".join(answer_text.lower().split())
    abstain_markers = (
        "not provided",
        "not available",
        "not specified",
        "not established",
        "does not establish",
        "does not explicitly",
        "does not provide",
        "not listed",
        "not in the evidence",
        "not in the provided evidence",
        "the evidence does not",
        "exact email address is not",
        "exact form number",
    )
    return any(marker in normalized for marker in abstain_markers)


def _missing_detail_exactness_rewrite_decision(
    *,
    missing_slots: set[str],
    baseline_answer_text: str,
    exactness_verdict: MissingDetailExactnessVerdict | None,
) -> tuple[bool, str]:
    baseline_corrupted = _looks_corrupted(baseline_answer_text)
    baseline_missing_detail_abstains = _looks_like_missing_detail_abstention(baseline_answer_text)
    missing_context_slots = {"closest_supported_context", "page_or_location", "supporting_rule"}
    missing_missing_detail_context_slots = sorted(missing_slots & missing_context_slots)

    if baseline_corrupted and missing_slots:
        return True, "missing_detail_corrupted_answer"
    if (
        baseline_missing_detail_abstains
        and exactness_verdict is not None
        and not exactness_verdict.exact_detail_overstatement_risk
    ):
        return False, "baseline_keep"
    if "exact_detail_status" in missing_slots:
        return True, "missing_detail_failed_abstention"
    if not baseline_missing_detail_abstains and len(missing_missing_detail_context_slots) >= 2:
        return True, "missing_detail_missing_context"
    if exactness_verdict is not None and exactness_verdict.exact_detail_overstatement_risk:
        return True, "missing_detail_exactness_overstatement"
    return False, "baseline_keep"


def _looks_corrupted(answer_text: str) -> bool:
    tokens = [token.strip(".,:;!?()[]{}\"'").lower() for token in answer_text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 40:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    short_ratio = sum(1 for token in tokens if len(token) <= 2) / len(tokens)
    return unique_ratio < 0.3 or short_ratio > 0.45


def _render_cited_workflow_contract_answer(contract: CitedStructuredAnswerContract) -> str:
    ordered_keys = (
        "bottom_line",
        "prerequisite_or_scope",
        "general_rule",
        "branch_if_all",
        "branch_if_some",
        "branch_if_none",
        "required_document_or_input",
        "deadline_or_timing",
        "follow_on_requirement",
        "exception",
        "consequence",
    )
    lines: list[str] = []
    opening = contract.slots.get("bottom_line")
    if opening:
        lines.append(_render_text_with_chunk_ids(opening.text, opening.citation_chunk_ids))
    bullet_lines: list[str] = []
    for key in ordered_keys[1:]:
        slot = contract.slots.get(key)
        if not slot:
            continue
        bullet_lines.append(f"- {_render_text_with_chunk_ids(slot.text, slot.citation_chunk_ids)}")
    if bullet_lines:
        if lines:
            lines.append("\n".join(bullet_lines))
        else:
            lines.extend(bullet_lines)
    return "\n\n".join(lines).strip()


def _render_cited_navigation_contract_answer(contract: CitedStructuredAnswerContract) -> str:
    ordered_keys = (
        "start_page",
        "parent_stage",
        "child_page_with_rule",
        "what_that_page_covers",
        "direct_url",
    )
    lines: list[str] = []
    opening = contract.slots.get("start_page")
    if opening:
        lines.append(_render_text_with_chunk_ids(opening.text, opening.citation_chunk_ids))
    bullet_lines: list[str] = []
    for key in ordered_keys[1:]:
        slot = contract.slots.get(key)
        if not slot:
            continue
        bullet_lines.append(f"- {_render_text_with_chunk_ids(slot.text, slot.citation_chunk_ids)}")
    if bullet_lines:
        if lines:
            lines.append("\n".join(bullet_lines))
        else:
            lines.extend(bullet_lines)
    return "\n\n".join(lines).strip()


def _render_cited_missing_detail_contract_answer(contract: CitedStructuredAnswerContract) -> str:
    lines: list[str] = []
    exact_status = contract.slots.get("exact_detail_status")
    if exact_status:
        lines.append(_render_text_with_chunk_ids(exact_status.text, exact_status.citation_chunk_ids))
    elif contract.should_abstain:
        lines.append(contract.abstain_reason or "The evidence does not establish the exact requested detail.")
    bullet_lines: list[str] = []
    for key in ("closest_supported_context", "page_or_location", "supporting_rule"):
        slot = contract.slots.get(key)
        if not slot:
            continue
        bullet_lines.append(f"- {_render_text_with_chunk_ids(slot.text, slot.citation_chunk_ids)}")
    if bullet_lines:
        if lines:
            lines.append("\n".join(bullet_lines))
        else:
            lines.extend(bullet_lines)
    return "\n\n".join(lines).strip()


def _render_cited_direct_rule_contract_answer(contract: CitedStructuredAnswerContract) -> str:
    ordered_keys = ("bottom_line", "general_rule", "conditions", "exception", "consequence")
    lines: list[str] = []
    opening = contract.slots.get("bottom_line")
    if opening:
        lines.append(_render_text_with_chunk_ids(opening.text, opening.citation_chunk_ids))
    bullet_lines: list[str] = []
    for key in ordered_keys[1:]:
        slot = contract.slots.get(key)
        if not slot:
            continue
        bullet_lines.append(f"- {_render_text_with_chunk_ids(slot.text, slot.citation_chunk_ids)}")
    if bullet_lines:
        if lines:
            lines.append("\n".join(bullet_lines))
        else:
            lines.extend(bullet_lines)
    return "\n\n".join(lines).strip()


def _render_cited_structured_contract_answer(contract: CitedStructuredAnswerContract) -> str:
    if contract.answer_mode == "workflow":
        return _render_cited_workflow_contract_answer(contract)
    if contract.answer_mode == "navigation":
        return _render_cited_navigation_contract_answer(contract)
    if contract.answer_mode == "missing_detail":
        return _render_cited_missing_detail_contract_answer(contract)
    return _render_cited_direct_rule_contract_answer(contract)


def _collect_contract_citations(
    contract: CitedStructuredAnswerContract,
    chunks: list[ChunkRecord],
) -> list[AnswerCitation]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for key in _allowed_contract_slots(contract.answer_mode):
        slot = contract.slots.get(key)
        if not slot:
            continue
        for chunk_id in slot.citation_chunk_ids:
            if chunk_id in seen_ids or chunk_id not in chunk_by_id:
                continue
            ordered_ids.append(chunk_id)
            seen_ids.add(chunk_id)
    return [
        AnswerCitation(chunk_id=chunk_id, canonical_url=chunk_by_id[chunk_id].canonical_url)
        for chunk_id in ordered_ids
    ]


def _build_planned_inline_evidence_prompt(
    question: str,
    evidence: EvidenceBundle,
    plan_points: list[str],
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(plan_points)
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Use the evidence-grounded coverage checklist as mandatory answer coverage.\n"
        "Requirements:\n"
        "1. Start with the direct answer to the user's actual question.\n"
        "2. Address every checklist point that is supported by the evidence.\n"
        "3. Preserve branch conditions, deadlines, exceptions, and what happens if a condition is not met.\n"
        "4. If the question compares two or more mechanisms, options, or scenarios, answer each one explicitly under a separate short heading or bullet.\n"
        "5. Use short headings or bullets when the answer has multiple branches or steps.\n"
        "6. Cite chunk IDs in square brackets after the statements they support.\n"
        "7. If the evidence does not establish part of the answer, say that clearly.\n"
        "8. Keep the answer concise and coverage-complete; do not add unsupported policy background.\n"
        "9. Do not end with an unfinished heading, unfinished branch, or partial sentence.\n\n"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_navigation_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(plan.coverage_points)
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "The user is asking where in the Buyer's Guide to go.\n"
        "Requirements:\n"
        "1. Start with the exact Buyer’s Guide section or path the user should open.\n"
        "2. Include the direct page URL if the evidence provides it.\n"
        "3. After naming the path, add only a brief explanation of what rule or topic that page covers.\n"
        "4. Prefer a short answer with 2 to 4 bullets or short lines.\n"
        "5. Do not drift into a long policy summary when the user mainly needs navigation.\n"
        "6. Cite chunk IDs in square brackets after the statements they support.\n"
        "7. If the evidence points to a child page rather than a high-level section, name the child page explicitly.\n"
        "8. Do not invent navigation labels or URLs that are not shown in the evidence.\n\n"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_compact_workflow_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(plan.coverage_points)
    supporting_sources_present = any(
        chunk.source_family != SourceFamily.BUYERS_GUIDE for chunk in evidence.packed_chunks
    )
    supporting_block = ""
    if supporting_sources_present:
        supporting_block = (
            "If you rely on supporting policy or directive evidence, mention that briefly only where needed.\n"
        )
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Use the evidence-grounded coverage checklist as mandatory minimum coverage, but keep the answer tightly focused on the user's question.\n"
        "Requirements:\n"
        "1. Start with one direct bottom-line sentence.\n"
        "2. Then use at most 4 short bullets only if they are needed to cover distinct branches, deadlines, exceptions, or consequences.\n"
        "3. Cover every indispensable checklist point that is supported by the evidence, and make each bullet map to a checklist point instead of using bullets for extra examples or commentary.\n"
        "4. If the checklist includes a required file, documentation, notification, publication, or other follow-on step, state it explicitly.\n"
        "5. Do not add extra supported background, escalation detail, implementation detail, or policy commentary unless it appears in the checklist or is needed to answer the question correctly.\n"
        "6. Do not use section labels such as Direct Answer, Checklist Coverage, Additional Notes, or similar meta headings.\n"
        "7. Do not repeat the same point in multiple bullets or in both prose and bullets.\n"
        "8. Preserve branch conditions, deadlines, exceptions, and what happens if a condition is not met.\n"
        "9. Do not add penalties, disqualification outcomes, consultation steps, escalation paths, or example lists unless the checklist requires them or the user's question explicitly asks for them.\n"
        "10. Cite chunk IDs in square brackets after the statements they support.\n"
        "11. Do not end with an unfinished bullet, heading, or partial sentence.\n"
        f"12. {supporting_block if supporting_block else 'Do not mention supporting sources unless they are needed for the answer.'}\n\n"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_compact_navigation_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(plan.coverage_points)
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "The user is asking where in the Buyer's Guide to go.\n"
        "Requirements:\n"
        "1. Use 2 to 4 short bullets or short lines only.\n"
        "2. Start with the exact Buyer’s Guide stage or page the user should open.\n"
        "3. Name the child page with the actual rule if there is one.\n"
        "4. Include the direct URL only if the evidence provides it.\n"
        "5. Add only the minimum rule context needed to explain why that page is the destination.\n"
        "6. Do not add a policy summary.\n"
        "7. Do not use meta headings such as Direct Answer or Checklist Coverage.\n"
        "8. Cite chunk IDs in square brackets after the statements they support.\n\n"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_compact_missing_detail_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(plan.coverage_points)
    abstain_block = ""
    if plan.should_abstain:
        abstain_block = (
            "The exact requested detail is not established by the evidence.\n"
            f"Reason: {plan.abstain_reason or 'The evidence does not provide the exact requested detail.'}\n"
        )
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Requirements:\n"
        "1. Use 2 to 3 short sentences or short bullets only.\n"
        "2. Start by stating whether the exact requested detail is established by the evidence.\n"
        "3. If the exact detail is not established, provide only the closest supported context from the checklist.\n"
        "4. Prefer the general method, page path, directory, list, or locator named in the evidence over suggesting a different contact path unless the checklist or evidence makes that contact path the closest supported context.\n"
        "5. Do not invent or imply exact contact details, identifiers, file names, or form numbers.\n"
        "6. Do not add contact-assistance or escalation guidance unless the evidence establishes it as the closest supported next step.\n"
        "7. Do not add background policy explanation beyond the closest supported context.\n"
        "8. Do not use meta headings such as Direct Answer or Checklist Coverage.\n"
        "9. Cite chunk IDs in square brackets after the statements they support.\n\n"
        f"{abstain_block}\n"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_compact_direct_rule_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(plan.coverage_points)
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Requirements:\n"
        "1. Start with one direct bottom-line sentence.\n"
        "2. Use up to 3 short bullets only if needed for conditions, exceptions, or consequences.\n"
        "3. Cover every indispensable checklist point that is supported by the evidence.\n"
        "4. Do not add extra background that is not needed to answer the user's question.\n"
        "5. Do not use meta headings such as Direct Answer or Checklist Coverage.\n"
        "6. Cite chunk IDs in square brackets after the statements they support.\n"
        "7. Do not end with an unfinished bullet or partial sentence.\n\n"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_workflow_contract_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    contract: StructuredAnswerContract,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    contract_block = _render_structured_contract_slots(contract)
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Use the structured answer contract as mandatory coverage.\n"
        "Requirements:\n"
        "1. Start with the Bottom line slot.\n"
        "2. Then cover every other non-empty slot exactly once.\n"
        "3. If prerequisite, required input, or follow-on requirement slots are populated, state them explicitly.\n"
        "4. If branch slots are populated, answer them under separate short bullets or headings.\n"
        "5. Preserve deadlines, exceptions, and consequences exactly as stated in the contract and evidence.\n"
        "6. Keep the answer concise and complete; do not stop after the first dispositive point.\n"
        "7. Cite chunk IDs in square brackets after the statements they support.\n"
        "8. If a slot is empty or absent, do not invent it.\n"
        "9. Do not end with an unfinished branch, unfinished heading, or partial sentence.\n\n"
        f"Original question:\n{question}\n\n"
        f"Structured answer contract:\n{contract_block}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_navigation_contract_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    contract: StructuredAnswerContract,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    contract_block = _render_structured_contract_slots(contract)
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "The user is asking where in the Buyer's Guide to go.\n"
        "Use the structured answer contract as mandatory coverage.\n"
        "Requirements:\n"
        "1. Start with the exact page or path the user should open.\n"
        "2. If a direct URL slot is populated, include it explicitly.\n"
        "3. Briefly mention the parent stage and what the destination page covers if those slots are populated.\n"
        "4. Prefer 2 to 4 short bullets or lines.\n"
        "5. Do not drift into a long policy summary.\n"
        "6. Cite chunk IDs in square brackets after the statements they support.\n"
        "7. If the child page contains the actual rule, name that child page explicitly.\n\n"
        f"Original question:\n{question}\n\n"
        f"Structured answer contract:\n{contract_block}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_missing_detail_contract_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    contract: StructuredAnswerContract,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    contract_block = _render_structured_contract_slots(contract)
    abstain_block = ""
    if contract.should_abstain:
        abstain_block = (
            "The structured contract determined that the exact requested detail is not established by the evidence.\n"
            f"Abstain reason: {contract.abstain_reason or 'The exact requested detail is not established by the evidence.'}\n\n"
        )
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Use the structured answer contract as mandatory coverage.\n"
        "Requirements:\n"
        "1. If the exact requested detail is not established, say that directly in the first sentence.\n"
        "2. Then provide only the closest supported context from the non-empty slots.\n"
        "3. Do not invent exact emails, direct contacts, file names, form numbers, or similar identifiers.\n"
        "4. Cite chunk IDs in square brackets after the statements they support.\n"
        "5. Keep the answer short and explicit about what is and is not known.\n\n"
        f"{abstain_block}"
        f"Original question:\n{question}\n\n"
        f"Structured answer contract:\n{contract_block}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_direct_rule_contract_answer_prompt(
    question: str,
    evidence: EvidenceBundle,
    contract: StructuredAnswerContract,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    contract_block = _render_structured_contract_slots(contract)
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Use the structured answer contract as mandatory coverage.\n"
        "Requirements:\n"
        "1. Start with the Bottom line slot.\n"
        "2. Then cover any populated rule, condition, exception, or consequence slots.\n"
        "3. Keep the answer concise and direct.\n"
        "4. Cite chunk IDs in square brackets after the statements they support.\n"
        "5. Do not add unsupported policy background.\n\n"
        f"Original question:\n{question}\n\n"
        f"Structured answer contract:\n{contract_block}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_answer_revision_prompt(
    question: str,
    evidence: EvidenceBundle,
    answer_plan: ModeAwareAnswerPlan,
    draft_answer: str,
    repair_plan: AnswerRepairPlan,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(answer_plan.coverage_points)
    missing_block = _render_plan_checklist(repair_plan.missing_supported_points) or "- None"
    unsupported_block = _render_plan_checklist(repair_plan.unsupported_or_overstated_points) or "- None"
    return (
        "You are revising an evidence-grounded procurement-policy answer.\n"
        "Use only the evidence below.\n"
        "Keep correct supported content from the draft answer, but repair completeness and support problems.\n"
        "Requirements:\n"
        "1. Preserve draft content that is already correct and supported.\n"
        "2. Add every missing supported point listed below.\n"
        "3. Remove or soften any unsupported or overstated point listed below.\n"
        "4. Keep the final answer concise and directly responsive to the user's question.\n"
        "5. Preserve branch conditions, deadlines, exceptions, and consequences when the evidence supports them.\n"
        "6. Cite chunk IDs in square brackets after the statements they support.\n"
        "7. Do not invent any new policy detail.\n\n"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Missing supported points to add:\n{missing_block}\n\n"
        f"Unsupported or overstated points to correct:\n{unsupported_block}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_mode_aware_planned_inline_evidence_prompt(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(plan.coverage_points)
    supporting_sources_present = any(
        chunk.source_family != SourceFamily.BUYERS_GUIDE for chunk in evidence.packed_chunks
    )

    mode_instructions = {
        "workflow": (
        "Mode-specific requirements:\n"
        "1. Start with a direct bottom-line answer.\n"
        "2. Then address every branch, option, or consequence in the checklist.\n"
        "3. If the question compares what happens when some, all, or none of a condition hold, answer each branch explicitly.\n"
        ),
        "navigation": (
            "Mode-specific requirements:\n"
            "1. Start by telling the user exactly where to go in the Buyer's Guide.\n"
            "2. Name the page or path explicitly before summarizing any rule.\n"
            "3. If the rule lives on a child page, say that directly.\n"
        ),
        "missing_detail": (
            "Mode-specific requirements:\n"
            "1. Start by stating that the provided documents do not establish the exact requested detail.\n"
            "2. Do not infer or invent exact emails, direct contacts, form numbers, file names, or similar identifiers.\n"
            "3. After abstaining, provide only the closest supported context from the checklist.\n"
        ),
        "direct_rule": (
            "Mode-specific requirements:\n"
            "1. Start with the rule or answer directly.\n"
            "2. Then list exact conditions, thresholds, or consequences from the checklist.\n"
        ),
    }[plan.answer_mode]

    abstain_block = ""
    if plan.should_abstain:
        abstain_block = (
            "The plan determined that the answer should abstain from the exact requested detail.\n"
            f"Abstain reason: {plan.abstain_reason or 'The exact requested detail is not established by the evidence.'}\n\n"
        )

    supporting_block = ""
    if supporting_sources_present:
        supporting_block = (
            "Supporting-source evidence is present. If you rely on any supporting policy or directive evidence, say so explicitly.\n\n"
        )

    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Use the evidence-grounded coverage checklist as mandatory answer coverage.\n"
        "General requirements:\n"
        "1. Resolve the original question completely.\n"
        "2. Address every checklist point that is supported by the evidence.\n"
        "3. Preserve branch conditions, deadlines, exceptions, and what happens if a condition is not met.\n"
        "4. Cite chunk IDs in square brackets after the statements they support.\n"
        "5. If the evidence does not establish part of the answer, say that clearly.\n"
        "6. Do not add unsupported policy background.\n"
        "7. Do not stop after the first dispositive point if the checklist contains additional supported points.\n\n"
        f"{mode_instructions}\n"
        f"{supporting_block}"
        f"{abstain_block}"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Evidence:\n{joined}"
    )


def _build_contextual_missing_detail_prompt(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> str:
    joined = _render_evidence_sections(evidence.packed_chunks)
    checklist = _render_plan_checklist(plan.coverage_points)
    supporting_sources_present = any(
        chunk.source_family != SourceFamily.BUYERS_GUIDE for chunk in evidence.packed_chunks
    )
    supporting_block = ""
    if supporting_sources_present:
        supporting_block = (
            "Supporting-source evidence is present. If you rely on any supporting policy or directive evidence, say so explicitly.\n\n"
        )
    opening_rule = (
        "1. Start by stating that the evidence does not establish the exact requested detail.\n"
        if plan.should_abstain
        else "1. Start with the exact supported detail if the evidence establishes it.\n"
    )
    abstain_block = ""
    if plan.should_abstain:
        abstain_block = (
            "The plan determined that the exact requested detail is not established by the evidence.\n"
            f"Abstain reason: {plan.abstain_reason or 'The exact requested detail is not established by the evidence.'}\n\n"
        )
    return (
        "You are a procurement policy assistant. Answer using only the evidence below.\n"
        "Use the evidence-grounded coverage checklist as mandatory answer coverage.\n"
        "General requirements:\n"
        f"{opening_rule}"
        "2. Then address every checklist point that is supported by the evidence.\n"
        "3. If the evidence provides nearby identifiers, page paths, template names, URLs, or related forms, present them only as the closest supported context.\n"
        "4. Do not present nearby identifiers or related forms as the exact requested detail unless the evidence explicitly ties them to the user's request.\n"
        "5. Cite chunk IDs in square brackets after the statements they support.\n"
        "6. Do not add unsupported policy background.\n\n"
        f"{supporting_block}"
        f"{abstain_block}"
        f"Original question:\n{question}\n\n"
        f"Coverage checklist:\n{checklist}\n\n"
        f"Evidence:\n{joined}"
    )


def _select_mode_aware_answer_route(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> SelectiveAnswerPlanRoute:
    if plan.answer_mode == "navigation":
        return SelectiveAnswerPlanRoute(
            prompt=_build_navigation_answer_prompt(question, evidence, plan),
            selected_path="mode_aware_navigation",
            abstained=False,
        )
    if plan.answer_mode == "missing_detail":
        return SelectiveAnswerPlanRoute(
            prompt=_build_contextual_missing_detail_prompt(question, evidence, plan),
            selected_path="contextual_missing_detail",
            abstained=plan.should_abstain,
        )
    if plan.answer_mode == "workflow":
        return SelectiveAnswerPlanRoute(
            prompt=_build_planned_inline_evidence_prompt(question, evidence, plan.coverage_points),
            selected_path="planned_workflow",
            abstained=False,
        )
    return SelectiveAnswerPlanRoute(
        prompt=_build_inline_evidence_prompt(question, evidence),
        selected_path="inline_direct_rule",
        abstained=False,
    )


def _select_compact_mode_aware_answer_route(
    question: str,
    evidence: EvidenceBundle,
    plan: ModeAwareAnswerPlan,
) -> SelectiveAnswerPlanRoute:
    if plan.answer_mode == "navigation":
        return SelectiveAnswerPlanRoute(
            prompt=_build_compact_navigation_answer_prompt(question, evidence, plan),
            selected_path="compact_navigation",
            abstained=False,
        )
    if plan.answer_mode == "missing_detail":
        return SelectiveAnswerPlanRoute(
            prompt=_build_compact_missing_detail_answer_prompt(question, evidence, plan),
            selected_path="compact_missing_detail",
            abstained=plan.should_abstain,
        )
    if plan.answer_mode == "workflow":
        return SelectiveAnswerPlanRoute(
            prompt=_build_compact_workflow_answer_prompt(question, evidence, plan),
            selected_path="compact_workflow",
            abstained=False,
        )
    return SelectiveAnswerPlanRoute(
        prompt=_build_compact_direct_rule_answer_prompt(question, evidence, plan),
        selected_path="compact_direct_rule",
        abstained=False,
    )


def _select_structured_contract_answer_route(
    question: str,
    evidence: EvidenceBundle,
    contract: StructuredAnswerContract,
) -> SelectiveAnswerPlanRoute:
    if contract.answer_mode == "navigation":
        return SelectiveAnswerPlanRoute(
            prompt=_build_navigation_contract_answer_prompt(question, evidence, contract),
            selected_path="contract_navigation",
            abstained=False,
        )
    if contract.answer_mode == "missing_detail":
        return SelectiveAnswerPlanRoute(
            prompt=_build_missing_detail_contract_answer_prompt(question, evidence, contract),
            selected_path="contract_missing_detail",
            abstained=contract.should_abstain,
        )
    if contract.answer_mode == "workflow":
        return SelectiveAnswerPlanRoute(
            prompt=_build_workflow_contract_answer_prompt(question, evidence, contract),
            selected_path="contract_workflow",
            abstained=False,
        )
    return SelectiveAnswerPlanRoute(
        prompt=_build_direct_rule_contract_answer_prompt(question, evidence, contract),
        selected_path="contract_direct_rule",
        abstained=False,
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


def _build_citations(chunks: list[ChunkRecord]) -> list[AnswerCitation]:
    return [AnswerCitation(chunk_id=chunk.chunk_id, canonical_url=chunk.canonical_url) for chunk in chunks]


def _truncate_document_snippet(text: str, *, max_words: int = 300, max_chars: int | None = None) -> str:
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if max_chars is not None and len(text) > max_chars:
        clipped = text[:max_chars].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        text = clipped
    return text.strip()


def _build_cohere_documents(
    settings: Settings,
    chunks: list[ChunkRecord],
) -> tuple[list[ct.Document], dict[str, AnswerCitation]]:
    documents: list[ct.Document] = []
    citation_lookup: dict[str, AnswerCitation] = {}
    for chunk in chunks:
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.title
        snippet = _truncate_document_snippet(
            chunk.text,
            max_words=300,
            max_chars=min(settings.max_doc_chars, 1800),
        )
        document = ct.Document(
            id=chunk.chunk_id,
            data={
                "title": heading,
                "text": snippet,
                "snippet": snippet,
                "url": chunk.canonical_url,
                "source_family": chunk.source_family.value,
            },
        )
        documents.append(document)
        citation_lookup[chunk.chunk_id] = AnswerCitation(
            chunk_id=chunk.chunk_id,
            canonical_url=chunk.canonical_url,
            snippet=snippet,
        )
    return documents, citation_lookup


def _build_response_citations(
    response: object,
    citation_lookup: dict[str, AnswerCitation],
) -> list[AnswerCitation]:
    message = getattr(response, "message", None)
    raw_citations = getattr(message, "citations", None) or []
    resolved: list[AnswerCitation] = []
    seen: set[str] = set()
    for citation in raw_citations:
        for document_id in getattr(citation, "document_ids", []) or []:
            if document_id in seen:
                continue
            source = citation_lookup.get(str(document_id))
            if source is None:
                continue
            resolved.append(source)
            seen.add(str(document_id))
    return resolved


def _documents_system_prompt() -> str:
    return (
        "You are a procurement policy assistant.\n"
        "Answer using only the supplied grounded documents.\n"
        "Start with the direct answer to the user's actual question.\n"
        "Resolve every distinct part of the question explicitly.\n"
        "Preserve prerequisites, branch conditions, deadlines, exceptions, and what happens if a condition is or is not met.\n"
        "If the question compares two or more mechanisms, options, or scenarios, answer each one explicitly under a short heading or bullet.\n"
        "If the documents do not establish an exact detail, say so clearly in the first sentence and then give only the closest supported context.\n"
        "Do not invent identifiers, forms, contact details, or workflow steps.\n"
        "Do not present nearby identifiers, related forms, or adjacent artifacts as the exact requested answer unless the documents explicitly tie them to the user's request.\n"
        "Give a direct answer first, then short bullets only when they genuinely help.\n"
        "Do not mention internal chunk IDs or say 'the provided evidence' in the answer.\n"
    )


def _build_documents_user_prompt(question: str, evidence: EvidenceBundle) -> str:
    retrieved_aspects = [query.strip() for query in evidence.retrieval_queries[1:] if query.strip()]
    if not retrieved_aspects:
        return question
    aspect_lines = "\n".join(f"- {query}" for query in retrieved_aspects)
    return f"Original question:\n{question}\n\nRetrieved aspects to cover when supported:\n{aspect_lines}"


def inline_evidence_chat(settings: Settings, question: str, evidence: EvidenceBundle) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    response = client.chat(
        model=settings.cohere_chat_model,
        messages=[ct.UserChatMessageV2(content=_build_inline_evidence_prompt(question, evidence))],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(response),
        strategy_name="inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
    )


def structured_inline_evidence_chat(settings: Settings, question: str, evidence: EvidenceBundle) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    response = client.chat(
        model=settings.cohere_chat_model,
        messages=[ct.UserChatMessageV2(content=_build_structured_inline_evidence_prompt(question, evidence.packed_chunks))],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(response),
        strategy_name="structured_inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
    )


def documents_chat(settings: Settings, question: str, evidence: EvidenceBundle) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    documents, citation_lookup = _build_cohere_documents(settings, evidence.packed_chunks)
    response = client.chat(
        model=settings.cohere_chat_model,
        messages=[
            ct.SystemChatMessageV2(content=_documents_system_prompt()),
            ct.UserChatMessageV2(content=_build_documents_user_prompt(question, evidence)),
        ],
        documents=documents,
        citation_options=ct.CitationOptions(mode="ENABLED"),
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(response),
        strategy_name="documents_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_response_citations(response, citation_lookup),
        evidence_bundle=evidence,
        raw_response={"citations": [getattr(item, "document_ids", []) for item in getattr(getattr(response, "message", None), "citations", []) or []]},
    )


def query_guided_inline_evidence_chat(settings: Settings, question: str, evidence: EvidenceBundle) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    response = client.chat(
        model=settings.cohere_chat_model,
        messages=[ct.UserChatMessageV2(content=_build_query_guided_inline_evidence_prompt(question, evidence))],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(response),
        strategy_name="query_guided_inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
    )


def planned_inline_evidence_chat(settings: Settings, question: str, evidence: EvidenceBundle) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    plan_start = perf_counter()
    plan_response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[ct.UserChatMessageV2(content=_build_answer_plan_prompt(question, evidence))],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=400,
    )
    plan_text = _extract_text_from_chat_response(plan_response)
    plan_points = _normalize_answer_plan(plan_text)
    plan_end = perf_counter()
    answer_response = client.chat(
        model=settings.cohere_chat_model,
        messages=[
            ct.UserChatMessageV2(
                content=_build_planned_inline_evidence_prompt(question, evidence, plan_points)
            )
        ],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    answer_end = perf_counter()
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(answer_response),
        strategy_name="planned_inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
        raw_response={
            "answer_plan": plan_points,
            "planner_model": settings.cohere_query_planner_model,
            "answer_model": settings.cohere_chat_model,
        },
        timings={
            "answer_plan_seconds": plan_end - plan_start,
            "final_answer_seconds": answer_end - plan_end,
        },
    )


def mode_aware_planned_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    plan_start = perf_counter()
    plan_response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[ct.UserChatMessageV2(content=_build_mode_aware_answer_plan_prompt(question, evidence))],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=500,
    )
    plan_text = _extract_text_from_chat_response(plan_response)
    plan = _normalize_mode_aware_answer_plan(plan_text)
    plan_end = perf_counter()
    answer_response = client.chat(
        model=settings.cohere_chat_model,
        messages=[
            ct.UserChatMessageV2(
                content=_build_mode_aware_planned_inline_evidence_prompt(question, evidence, plan)
            )
        ],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    answer_end = perf_counter()
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(answer_response),
        strategy_name="mode_aware_planned_inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
        raw_response={
            "answer_plan": {
                "answer_mode": plan.answer_mode,
                "should_abstain": plan.should_abstain,
                "abstain_reason": plan.abstain_reason,
                "coverage_points": plan.coverage_points,
            },
            "planner_model": settings.cohere_query_planner_model,
            "answer_model": settings.cohere_chat_model,
        },
        timings={
            "answer_plan_seconds": plan_end - plan_start,
            "final_answer_seconds": answer_end - plan_end,
        },
        abstained=plan.should_abstain,
    )


def selective_mode_aware_planned_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    plan_start = perf_counter()
    plan_response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[ct.UserChatMessageV2(content=_build_mode_aware_answer_plan_prompt(question, evidence))],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=500,
    )
    plan_text = _extract_text_from_chat_response(plan_response)
    plan = _normalize_mode_aware_answer_plan(plan_text)
    plan_end = perf_counter()
    route = _select_mode_aware_answer_route(question, evidence, plan)
    answer_response = client.chat(
        model=settings.cohere_chat_model,
        messages=[ct.UserChatMessageV2(content=route.prompt)],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    answer_end = perf_counter()
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(answer_response),
        strategy_name="selective_mode_aware_planned_inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
        raw_response={
            "answer_plan": {
                "answer_mode": plan.answer_mode,
                "should_abstain": plan.should_abstain,
                "abstain_reason": plan.abstain_reason,
                "coverage_points": plan.coverage_points,
            },
            "selected_path": route.selected_path,
            "planner_model": settings.cohere_query_planner_model,
            "answer_model": settings.cohere_chat_model,
        },
        timings={
            "answer_plan_seconds": plan_end - plan_start,
            "final_answer_seconds": answer_end - plan_end,
        },
        abstained=route.abstained,
    )


def selective_mode_aware_compact_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    plan_start = perf_counter()
    plan_response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[ct.UserChatMessageV2(content=_build_compact_mode_aware_answer_plan_prompt(question, evidence))],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=450,
    )
    plan_text = _extract_text_from_chat_response(plan_response)
    plan = _normalize_mode_aware_answer_plan(plan_text, max_points=5)
    plan_end = perf_counter()
    route = _select_compact_mode_aware_answer_route(question, evidence, plan)
    answer_response = client.chat(
        model=settings.cohere_chat_model,
        messages=[ct.UserChatMessageV2(content=route.prompt)],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    answer_end = perf_counter()
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(answer_response),
        strategy_name="selective_mode_aware_compact_inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
        raw_response={
            "answer_plan": {
                "answer_mode": plan.answer_mode,
                "should_abstain": plan.should_abstain,
                "abstain_reason": plan.abstain_reason,
                "coverage_points": plan.coverage_points,
            },
            "selected_path": route.selected_path,
            "planner_model": settings.cohere_query_planner_model,
            "answer_model": settings.cohere_chat_model,
        },
        timings={
            "answer_plan_seconds": plan_end - plan_start,
            "final_answer_seconds": answer_end - plan_end,
        },
        abstained=route.abstained,
    )


def selective_mode_aware_answer_repair_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    plan_start = perf_counter()
    plan_response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[ct.UserChatMessageV2(content=_build_mode_aware_answer_plan_prompt(question, evidence))],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=500,
    )
    plan_text = _extract_text_from_chat_response(plan_response)
    plan = _normalize_mode_aware_answer_plan(plan_text)
    plan_end = perf_counter()
    route = _select_mode_aware_answer_route(question, evidence, plan)
    draft_response = client.chat(
        model=settings.cohere_chat_model,
        messages=[ct.UserChatMessageV2(content=route.prompt)],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    draft_end = perf_counter()
    draft_answer = _extract_text_from_chat_response(draft_response)
    repair_response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[
            ct.UserChatMessageV2(
                content=_build_answer_repair_plan_prompt(question, evidence, plan, draft_answer)
            )
        ],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=400,
    )
    repair_text = _extract_text_from_chat_response(repair_response)
    repair_plan = _normalize_answer_repair_plan(repair_text)
    repair_plan_end = perf_counter()
    final_answer = draft_answer
    final_response = None
    if repair_plan.needs_revision:
        final_response = client.chat(
            model=settings.cohere_chat_model,
            messages=[
                ct.UserChatMessageV2(
                    content=_build_answer_revision_prompt(
                        question,
                        evidence,
                        plan,
                        draft_answer,
                        repair_plan,
                    )
                )
            ],
            temperature=settings.chat_temperature,
            max_tokens=settings.chat_max_output_tokens,
        )
        final_answer = _extract_text_from_chat_response(final_response)
    final_end = perf_counter()
    return AnswerResult(
        question=question,
        answer_text=final_answer,
        strategy_name="selective_mode_aware_answer_repair_inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
        raw_response={
            "answer_plan": {
                "answer_mode": plan.answer_mode,
                "should_abstain": plan.should_abstain,
                "abstain_reason": plan.abstain_reason,
                "coverage_points": plan.coverage_points,
            },
            "selected_path": route.selected_path,
            "repair_plan": {
                "needs_revision": repair_plan.needs_revision,
                "missing_supported_points": repair_plan.missing_supported_points,
                "unsupported_or_overstated_points": repair_plan.unsupported_or_overstated_points,
            },
            "planner_model": settings.cohere_query_planner_model,
            "answer_model": settings.cohere_chat_model,
            "draft_answer_text": draft_answer,
            "revised_answer": final_response is not None,
        },
        timings={
            "answer_plan_seconds": plan_end - plan_start,
            "draft_answer_seconds": draft_end - plan_end,
            "repair_plan_seconds": repair_plan_end - draft_end,
            "final_answer_seconds": final_end - repair_plan_end,
        },
        abstained=route.abstained,
    )


def structured_contract_deterministic_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    contract_start = perf_counter()
    contract, contract_payload = _extract_cited_structured_answer_contract(settings, question, evidence)
    contract_end = perf_counter()
    answer_text = _render_cited_structured_contract_answer(contract)
    render_end = perf_counter()
    return AnswerResult(
        question=question,
        answer_text=answer_text,
        strategy_name="structured_contract_deterministic_inline_evidence_chat",
        model_name=settings.cohere_query_planner_model,
        citations=_collect_contract_citations(contract, evidence.packed_chunks),
        evidence_bundle=evidence,
        raw_response={
            "structured_contract": {
                "answer_mode": contract.answer_mode,
                "should_abstain": contract.should_abstain,
                "abstain_reason": contract.abstain_reason,
                "slots": {
                    key: {
                        "text": value.text,
                        "citation_chunk_ids": value.citation_chunk_ids,
                    }
                    for key, value in contract.slots.items()
                },
            },
            "structured_contract_payload": contract_payload.model_dump(),
            "planner_model": settings.cohere_query_planner_model,
            "planner_framework": "instructor_cohere_pydantic",
            "render_mode": "deterministic",
        },
        timings={
            "structured_contract_seconds": contract_end - contract_start,
            "deterministic_render_seconds": render_end - contract_end,
        },
        abstained=contract.should_abstain,
    )


def selective_workflow_contract_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    plan_start = perf_counter()
    plan_response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[ct.UserChatMessageV2(content=_build_compact_mode_aware_answer_plan_prompt(question, evidence))],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=500,
    )
    plan_text = _extract_text_from_chat_response(plan_response)
    plan = _normalize_mode_aware_answer_plan(plan_text, max_points=6)
    plan_end = perf_counter()

    if plan.answer_mode in {"workflow", "missing_detail"} or plan.should_abstain:
        contract_start = perf_counter()
        contract, contract_payload = _extract_cited_structured_answer_contract(settings, question, evidence)
        contract_end = perf_counter()
        answer_text = _render_cited_structured_contract_answer(contract)
        render_end = perf_counter()
        return AnswerResult(
            question=question,
            answer_text=answer_text,
            strategy_name="selective_workflow_contract_inline_evidence_chat",
            model_name=settings.cohere_query_planner_model,
            citations=_collect_contract_citations(contract, evidence.packed_chunks),
            evidence_bundle=evidence,
            raw_response={
                "answer_plan": {
                    "answer_mode": plan.answer_mode,
                    "should_abstain": plan.should_abstain,
                    "abstain_reason": plan.abstain_reason,
                    "coverage_points": plan.coverage_points,
                },
                "selected_path": "structured_contract_deterministic",
                "structured_contract": {
                    "answer_mode": contract.answer_mode,
                    "should_abstain": contract.should_abstain,
                    "abstain_reason": contract.abstain_reason,
                    "slots": {
                        key: {
                            "text": value.text,
                            "citation_chunk_ids": value.citation_chunk_ids,
                        }
                        for key, value in contract.slots.items()
                    },
                },
                "structured_contract_payload": contract_payload.model_dump(),
                "planner_framework": "instructor_cohere_pydantic",
                "planner_model": settings.cohere_query_planner_model,
                "render_mode": "deterministic",
            },
            timings={
                "answer_plan_seconds": plan_end - plan_start,
                "structured_contract_seconds": contract_end - contract_start,
                "deterministic_render_seconds": render_end - contract_end,
            },
            abstained=contract.should_abstain,
        )

    baseline_result = inline_evidence_chat(settings, question, evidence)
    raw_response = dict(baseline_result.raw_response or {})
    raw_response["answer_plan"] = {
        "answer_mode": plan.answer_mode,
        "should_abstain": plan.should_abstain,
        "abstain_reason": plan.abstain_reason,
        "coverage_points": plan.coverage_points,
    }
    raw_response["selected_path"] = "inline_evidence_baseline"
    timings = dict(baseline_result.timings)
    timings["answer_plan_seconds"] = plan_end - plan_start
    return AnswerResult(
        question=baseline_result.question,
        answer_text=baseline_result.answer_text,
        strategy_name="selective_workflow_contract_inline_evidence_chat",
        model_name=baseline_result.model_name,
        citations=baseline_result.citations,
        evidence_bundle=baseline_result.evidence_bundle,
        raw_response=raw_response,
        timings=timings,
        abstained=baseline_result.abstained,
    )


def verifier_gated_structured_contract_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    baseline_start = perf_counter()
    baseline_result = inline_evidence_chat(settings, question, evidence)
    baseline_end = perf_counter()

    verifier_start = perf_counter()
    verdict, verdict_payload = _extract_answer_rewrite_verdict(
        settings,
        question,
        evidence,
        baseline_result.answer_text,
    )
    verifier_end = perf_counter()

    should_rewrite = (
        verdict.action == "rewrite_structured_contract"
        and verdict.confidence in {"medium", "high"}
        and (
            verdict.omission_risk
            or verdict.exact_detail_abstain_risk
            or verdict.unsupported_detail_risk
        )
    )

    if should_rewrite:
        contract_start = perf_counter()
        contract, contract_payload = _extract_cited_structured_answer_contract(settings, question, evidence)
        contract_end = perf_counter()
        answer_text = _render_cited_structured_contract_answer(contract)
        render_end = perf_counter()
        return AnswerResult(
            question=question,
            answer_text=answer_text,
            strategy_name="verifier_gated_structured_contract_inline_evidence_chat",
            model_name=settings.cohere_query_planner_model,
            citations=_collect_contract_citations(contract, evidence.packed_chunks),
            evidence_bundle=evidence,
            raw_response={
                "baseline_answer_text": baseline_result.answer_text,
                "selected_path": "rewrite_structured_contract",
                "rewrite_verdict": {
                    "action": verdict.action,
                    "confidence": verdict.confidence,
                    "rationale": verdict.rationale,
                    "omission_risk": verdict.omission_risk,
                    "exact_detail_abstain_risk": verdict.exact_detail_abstain_risk,
                    "unsupported_detail_risk": verdict.unsupported_detail_risk,
                },
                "rewrite_verdict_payload": verdict_payload.model_dump(),
                "structured_contract": {
                    "answer_mode": contract.answer_mode,
                    "should_abstain": contract.should_abstain,
                    "abstain_reason": contract.abstain_reason,
                    "slots": {
                        key: {
                            "text": value.text,
                            "citation_chunk_ids": value.citation_chunk_ids,
                        }
                        for key, value in contract.slots.items()
                    },
                },
                "structured_contract_payload": contract_payload.model_dump(),
                "planner_framework": "instructor_cohere_pydantic",
                "planner_model": settings.cohere_query_planner_model,
                "baseline_model": baseline_result.model_name,
                "render_mode": "deterministic",
            },
            timings={
                "baseline_answer_seconds": baseline_end - baseline_start,
                "rewrite_verdict_seconds": verifier_end - verifier_start,
                "structured_contract_seconds": contract_end - contract_start,
                "deterministic_render_seconds": render_end - contract_end,
            },
            abstained=contract.should_abstain,
        )

    raw_response = dict(baseline_result.raw_response or {})
    raw_response["baseline_answer_text"] = baseline_result.answer_text
    raw_response["selected_path"] = "baseline_keep"
    raw_response["rewrite_verdict"] = {
        "action": verdict.action,
        "confidence": verdict.confidence,
        "rationale": verdict.rationale,
        "omission_risk": verdict.omission_risk,
        "exact_detail_abstain_risk": verdict.exact_detail_abstain_risk,
        "unsupported_detail_risk": verdict.unsupported_detail_risk,
    }
    raw_response["rewrite_verdict_payload"] = verdict_payload.model_dump()
    timings = dict(baseline_result.timings)
    timings["baseline_answer_seconds"] = baseline_end - baseline_start
    timings["rewrite_verdict_seconds"] = verifier_end - verifier_start
    return AnswerResult(
        question=baseline_result.question,
        answer_text=baseline_result.answer_text,
        strategy_name="verifier_gated_structured_contract_inline_evidence_chat",
        model_name=baseline_result.model_name,
        citations=baseline_result.citations,
        evidence_bundle=baseline_result.evidence_bundle,
        raw_response=raw_response,
        timings=timings,
        abstained=baseline_result.abstained,
    )


def contract_aware_verifier_gated_structured_contract_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    baseline_start = perf_counter()
    baseline_result = inline_evidence_chat(settings, question, evidence)
    baseline_end = perf_counter()

    contract_start = perf_counter()
    contract, contract_payload = _extract_cited_structured_answer_contract(settings, question, evidence)
    contract_end = perf_counter()

    verifier_start = perf_counter()
    verdict, verdict_payload = _extract_contract_aware_answer_rewrite_verdict(
        settings,
        question,
        evidence,
        contract,
        baseline_result.answer_text,
    )
    verifier_end = perf_counter()

    should_rewrite = (
        verdict.action == "rewrite_structured_contract"
        and verdict.confidence in {"medium", "high"}
        and (
            verdict.omission_risk
            or verdict.exact_detail_abstain_risk
            or verdict.unsupported_detail_risk
        )
    )

    contract_snapshot = {
        "answer_mode": contract.answer_mode,
        "should_abstain": contract.should_abstain,
        "abstain_reason": contract.abstain_reason,
        "slots": {
            key: {
                "text": value.text,
                "citation_chunk_ids": value.citation_chunk_ids,
            }
            for key, value in contract.slots.items()
        },
    }

    if should_rewrite:
        render_start = perf_counter()
        answer_text = _render_cited_structured_contract_answer(contract)
        render_end = perf_counter()
        return AnswerResult(
            question=question,
            answer_text=answer_text,
            strategy_name="contract_aware_verifier_gated_structured_contract_inline_evidence_chat",
            model_name=settings.cohere_query_planner_model,
            citations=_collect_contract_citations(contract, evidence.packed_chunks),
            evidence_bundle=evidence,
            raw_response={
                "baseline_answer_text": baseline_result.answer_text,
                "selected_path": "rewrite_structured_contract",
                "rewrite_verdict": {
                    "action": verdict.action,
                    "confidence": verdict.confidence,
                    "rationale": verdict.rationale,
                    "omission_risk": verdict.omission_risk,
                    "exact_detail_abstain_risk": verdict.exact_detail_abstain_risk,
                    "unsupported_detail_risk": verdict.unsupported_detail_risk,
                },
                "rewrite_verdict_payload": verdict_payload.model_dump(),
                "structured_contract": contract_snapshot,
                "structured_contract_payload": contract_payload.model_dump(),
                "planner_framework": "instructor_cohere_pydantic",
                "planner_model": settings.cohere_query_planner_model,
                "baseline_model": baseline_result.model_name,
                "render_mode": "deterministic",
            },
            timings={
                "baseline_answer_seconds": baseline_end - baseline_start,
                "structured_contract_seconds": contract_end - contract_start,
                "rewrite_verdict_seconds": verifier_end - verifier_start,
                "deterministic_render_seconds": render_end - render_start,
            },
            abstained=contract.should_abstain,
        )

    raw_response = dict(baseline_result.raw_response or {})
    raw_response["baseline_answer_text"] = baseline_result.answer_text
    raw_response["selected_path"] = "baseline_keep"
    raw_response["rewrite_verdict"] = {
        "action": verdict.action,
        "confidence": verdict.confidence,
        "rationale": verdict.rationale,
        "omission_risk": verdict.omission_risk,
        "exact_detail_abstain_risk": verdict.exact_detail_abstain_risk,
        "unsupported_detail_risk": verdict.unsupported_detail_risk,
    }
    raw_response["rewrite_verdict_payload"] = verdict_payload.model_dump()
    raw_response["structured_contract"] = contract_snapshot
    raw_response["structured_contract_payload"] = contract_payload.model_dump()
    timings = dict(baseline_result.timings)
    timings["baseline_answer_seconds"] = baseline_end - baseline_start
    timings["structured_contract_seconds"] = contract_end - contract_start
    timings["rewrite_verdict_seconds"] = verifier_end - verifier_start
    return AnswerResult(
        question=baseline_result.question,
        answer_text=baseline_result.answer_text,
        strategy_name="contract_aware_verifier_gated_structured_contract_inline_evidence_chat",
        model_name=baseline_result.model_name,
        citations=baseline_result.citations,
        evidence_bundle=baseline_result.evidence_bundle,
        raw_response=raw_response,
        timings=timings,
        abstained=baseline_result.abstained,
    )


def contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    baseline_start = perf_counter()
    baseline_result = inline_evidence_chat(settings, question, evidence)
    baseline_end = perf_counter()

    contract_start = perf_counter()
    contract, contract_payload = _extract_cited_structured_answer_contract(settings, question, evidence)
    contract_end = perf_counter()

    coverage_start = perf_counter()
    coverage_verdict, coverage_payload = _extract_contract_slot_coverage_verdict(
        settings,
        question,
        evidence,
        contract,
        baseline_result.answer_text,
    )
    coverage_end = perf_counter()

    populated_slots = set(contract.slots)
    critical_slots = set(populated_slots)
    if contract.answer_mode == "direct_rule":
        critical_slots &= {"bottom_line", "general_rule", "conditions", "exception", "consequence"}
    elif contract.answer_mode == "navigation":
        critical_slots &= {"start_page", "parent_stage", "child_page_with_rule", "what_that_page_covers", "direct_url"}
    missing_critical_slots = [slot for slot in coverage_verdict.missing_or_weakened_slots if slot in critical_slots]
    should_rewrite = bool(missing_critical_slots or coverage_verdict.unsupported_detail_risk)

    contract_snapshot = {
        "answer_mode": contract.answer_mode,
        "should_abstain": contract.should_abstain,
        "abstain_reason": contract.abstain_reason,
        "slots": {
            key: {
                "text": value.text,
                "citation_chunk_ids": value.citation_chunk_ids,
            }
            for key, value in contract.slots.items()
        },
    }

    if should_rewrite:
        render_start = perf_counter()
        answer_text = _render_cited_structured_contract_answer(contract)
        render_end = perf_counter()
        return AnswerResult(
            question=question,
            answer_text=answer_text,
            strategy_name="contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat",
            model_name=settings.cohere_query_planner_model,
            citations=_collect_contract_citations(contract, evidence.packed_chunks),
            evidence_bundle=evidence,
            raw_response={
                "baseline_answer_text": baseline_result.answer_text,
                "selected_path": "rewrite_structured_contract",
                "slot_coverage_verdict": {
                    "confidence": coverage_verdict.confidence,
                    "rationale": coverage_verdict.rationale,
                    "missing_or_weakened_slots": coverage_verdict.missing_or_weakened_slots,
                    "missing_critical_slots": missing_critical_slots,
                    "unsupported_detail_risk": coverage_verdict.unsupported_detail_risk,
                },
                "slot_coverage_verdict_payload": coverage_payload.model_dump(),
                "structured_contract": contract_snapshot,
                "structured_contract_payload": contract_payload.model_dump(),
                "planner_framework": "instructor_cohere_pydantic",
                "planner_model": settings.cohere_query_planner_model,
                "baseline_model": baseline_result.model_name,
                "render_mode": "deterministic",
            },
            timings={
                "baseline_answer_seconds": baseline_end - baseline_start,
                "structured_contract_seconds": contract_end - contract_start,
                "slot_coverage_verdict_seconds": coverage_end - coverage_start,
                "deterministic_render_seconds": render_end - render_start,
            },
            abstained=contract.should_abstain,
        )

    raw_response = dict(baseline_result.raw_response or {})
    raw_response["baseline_answer_text"] = baseline_result.answer_text
    raw_response["selected_path"] = "baseline_keep"
    raw_response["slot_coverage_verdict"] = {
        "confidence": coverage_verdict.confidence,
        "rationale": coverage_verdict.rationale,
        "missing_or_weakened_slots": coverage_verdict.missing_or_weakened_slots,
        "missing_critical_slots": missing_critical_slots,
        "unsupported_detail_risk": coverage_verdict.unsupported_detail_risk,
    }
    raw_response["slot_coverage_verdict_payload"] = coverage_payload.model_dump()
    raw_response["structured_contract"] = contract_snapshot
    raw_response["structured_contract_payload"] = contract_payload.model_dump()
    timings = dict(baseline_result.timings)
    timings["baseline_answer_seconds"] = baseline_end - baseline_start
    timings["structured_contract_seconds"] = contract_end - contract_start
    timings["slot_coverage_verdict_seconds"] = coverage_end - coverage_start
    return AnswerResult(
        question=baseline_result.question,
        answer_text=baseline_result.answer_text,
        strategy_name="contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat",
        model_name=baseline_result.model_name,
        citations=baseline_result.citations,
        evidence_bundle=baseline_result.evidence_bundle,
        raw_response=raw_response,
        timings=timings,
        abstained=baseline_result.abstained,
    )


def narrow_contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    baseline_start = perf_counter()
    baseline_result = inline_evidence_chat(settings, question, evidence)
    baseline_end = perf_counter()

    contract_start = perf_counter()
    contract, contract_payload = _extract_cited_structured_answer_contract(settings, question, evidence)
    contract_end = perf_counter()

    coverage_start = perf_counter()
    coverage_verdict, coverage_payload = _extract_contract_slot_coverage_verdict(
        settings,
        question,
        evidence,
        contract,
        baseline_result.answer_text,
    )
    coverage_end = perf_counter()

    missing_slots = set(coverage_verdict.missing_or_weakened_slots)
    branch_slots = {"branch_if_all", "branch_if_some", "branch_if_none"}
    missing_branch_slots = sorted(missing_slots & branch_slots)
    missing_detail_context_slots = {"closest_supported_context", "page_or_location", "supporting_rule"}
    missing_missing_detail_context_slots = sorted(missing_slots & missing_detail_context_slots)
    baseline_missing_detail_abstains = _looks_like_missing_detail_abstention(baseline_result.answer_text)
    baseline_corrupted = _looks_corrupted(baseline_result.answer_text)

    should_rewrite = False
    activation_reason = "baseline_keep"
    if contract.answer_mode == "workflow":
        if len(missing_branch_slots) >= 2:
            should_rewrite = True
            activation_reason = "workflow_multi_branch_gap"
    elif contract.answer_mode == "missing_detail":
        if baseline_corrupted and missing_slots:
            should_rewrite = True
            activation_reason = "missing_detail_corrupted_answer"
        elif not baseline_missing_detail_abstains:
            if "exact_detail_status" in missing_slots:
                should_rewrite = True
                activation_reason = "missing_detail_failed_abstention"
            elif len(missing_missing_detail_context_slots) >= 2:
                should_rewrite = True
                activation_reason = "missing_detail_missing_context"

    contract_snapshot = {
        "answer_mode": contract.answer_mode,
        "should_abstain": contract.should_abstain,
        "abstain_reason": contract.abstain_reason,
        "slots": {
            key: {
                "text": value.text,
                "citation_chunk_ids": value.citation_chunk_ids,
            }
            for key, value in contract.slots.items()
        },
    }

    verdict_snapshot = {
        "confidence": coverage_verdict.confidence,
        "rationale": coverage_verdict.rationale,
        "missing_or_weakened_slots": coverage_verdict.missing_or_weakened_slots,
        "missing_branch_slots": missing_branch_slots,
        "missing_missing_detail_context_slots": missing_missing_detail_context_slots,
        "unsupported_detail_risk": coverage_verdict.unsupported_detail_risk,
        "baseline_missing_detail_abstains": baseline_missing_detail_abstains,
        "baseline_corrupted": baseline_corrupted,
        "activation_reason": activation_reason,
    }

    if should_rewrite:
        render_start = perf_counter()
        answer_text = _render_cited_structured_contract_answer(contract)
        render_end = perf_counter()
        return AnswerResult(
            question=question,
            answer_text=answer_text,
            strategy_name="narrow_contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat",
            model_name=settings.cohere_query_planner_model,
            citations=_collect_contract_citations(contract, evidence.packed_chunks),
            evidence_bundle=evidence,
            raw_response={
                "baseline_answer_text": baseline_result.answer_text,
                "selected_path": "rewrite_structured_contract",
                "slot_coverage_verdict": verdict_snapshot,
                "slot_coverage_verdict_payload": coverage_payload.model_dump(),
                "structured_contract": contract_snapshot,
                "structured_contract_payload": contract_payload.model_dump(),
                "planner_framework": "instructor_cohere_pydantic",
                "planner_model": settings.cohere_query_planner_model,
                "baseline_model": baseline_result.model_name,
                "render_mode": "deterministic",
            },
            timings={
                "baseline_answer_seconds": baseline_end - baseline_start,
                "structured_contract_seconds": contract_end - contract_start,
                "slot_coverage_verdict_seconds": coverage_end - coverage_start,
                "deterministic_render_seconds": render_end - render_start,
            },
            abstained=contract.should_abstain,
        )

    raw_response = dict(baseline_result.raw_response or {})
    raw_response["baseline_answer_text"] = baseline_result.answer_text
    raw_response["selected_path"] = "baseline_keep"
    raw_response["slot_coverage_verdict"] = verdict_snapshot
    raw_response["slot_coverage_verdict_payload"] = coverage_payload.model_dump()
    raw_response["structured_contract"] = contract_snapshot
    raw_response["structured_contract_payload"] = contract_payload.model_dump()
    timings = dict(baseline_result.timings)
    timings["baseline_answer_seconds"] = baseline_end - baseline_start
    timings["structured_contract_seconds"] = contract_end - contract_start
    timings["slot_coverage_verdict_seconds"] = coverage_end - coverage_start
    return AnswerResult(
        question=baseline_result.question,
        answer_text=baseline_result.answer_text,
        strategy_name="narrow_contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat",
        model_name=baseline_result.model_name,
        citations=baseline_result.citations,
        evidence_bundle=baseline_result.evidence_bundle,
        raw_response=raw_response,
        timings=timings,
        abstained=baseline_result.abstained,
    )


def missing_detail_exactness_verifier_gated_structured_contract_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    baseline_start = perf_counter()
    baseline_result = inline_evidence_chat(settings, question, evidence)
    baseline_end = perf_counter()

    contract_start = perf_counter()
    contract, contract_payload = _extract_cited_structured_answer_contract(settings, question, evidence)
    contract_end = perf_counter()

    contract_snapshot = {
        "answer_mode": contract.answer_mode,
        "should_abstain": contract.should_abstain,
        "abstain_reason": contract.abstain_reason,
        "slots": {
            key: {
                "text": value.text,
                "citation_chunk_ids": value.citation_chunk_ids,
            }
            for key, value in contract.slots.items()
        },
    }

    if contract.answer_mode != "missing_detail":
        raw_response = dict(baseline_result.raw_response or {})
        raw_response["baseline_answer_text"] = baseline_result.answer_text
        raw_response["selected_path"] = "baseline_keep"
        raw_response["structured_contract"] = contract_snapshot
        raw_response["structured_contract_payload"] = contract_payload.model_dump()
        timings = dict(baseline_result.timings)
        timings["baseline_answer_seconds"] = baseline_end - baseline_start
        timings["structured_contract_seconds"] = contract_end - contract_start
        timings["slot_coverage_verdict_seconds"] = 0.0
        timings["missing_detail_exactness_verdict_seconds"] = 0.0
        return AnswerResult(
            question=baseline_result.question,
            answer_text=baseline_result.answer_text,
            strategy_name="missing_detail_exactness_verifier_gated_structured_contract_inline_evidence_chat",
            model_name=baseline_result.model_name,
            citations=baseline_result.citations,
            evidence_bundle=baseline_result.evidence_bundle,
            raw_response=raw_response,
            timings=timings,
            abstained=baseline_result.abstained,
        )

    coverage_start = perf_counter()
    coverage_verdict, coverage_payload = _extract_contract_slot_coverage_verdict(
        settings,
        question,
        evidence,
        contract,
        baseline_result.answer_text,
    )
    coverage_end = perf_counter()

    missing_slots = set(coverage_verdict.missing_or_weakened_slots)
    exactness_start = perf_counter()
    exactness_verdict, exactness_payload = _extract_missing_detail_exactness_verdict(
        settings,
        question,
        evidence,
        contract,
        baseline_result.answer_text,
    )
    exactness_end = perf_counter()

    should_rewrite, activation_reason = _missing_detail_exactness_rewrite_decision(
        missing_slots=missing_slots,
        baseline_answer_text=baseline_result.answer_text,
        exactness_verdict=exactness_verdict,
    )

    verdict_snapshot = {
        "confidence": coverage_verdict.confidence,
        "rationale": coverage_verdict.rationale,
        "missing_or_weakened_slots": coverage_verdict.missing_or_weakened_slots,
        "unsupported_detail_risk": coverage_verdict.unsupported_detail_risk,
        "activation_reason": activation_reason,
        "exactness_verdict": {
            "confidence": exactness_verdict.confidence,
            "rationale": exactness_verdict.rationale,
            "exact_detail_overstatement_risk": exactness_verdict.exact_detail_overstatement_risk,
            "offending_details": exactness_verdict.offending_details,
        },
    }

    if should_rewrite:
        selection_start = perf_counter()
        selected_slot_keys = _minimal_missing_detail_exactness_keep_set(
            contract,
            selector_keep_slot_keys=None,
            missing_slots=missing_slots,
        )
        render_contract = _prune_cited_structured_answer_contract(
            contract,
            keep_slot_keys=selected_slot_keys,
        )
        selection_end = perf_counter()
        render_start = perf_counter()
        answer_text = _render_cited_structured_contract_answer(render_contract)
        render_end = perf_counter()
        return AnswerResult(
            question=question,
            answer_text=answer_text,
            strategy_name="missing_detail_exactness_verifier_gated_structured_contract_inline_evidence_chat",
            model_name=settings.cohere_query_planner_model,
            citations=_collect_contract_citations(render_contract, evidence.packed_chunks),
            evidence_bundle=evidence,
            raw_response={
                "baseline_answer_text": baseline_result.answer_text,
                "selected_path": "rewrite_structured_contract",
                "slot_coverage_verdict": verdict_snapshot,
                "slot_coverage_verdict_payload": coverage_payload.model_dump(),
                "missing_detail_exactness_verdict_payload": exactness_payload.model_dump(),
                "structured_contract": contract_snapshot,
                "structured_contract_payload": contract_payload.model_dump(),
                "slot_selection": {
                    "selector_keep_slot_keys": [],
                    "final_keep_slot_keys": [
                        key
                        for key in _allowed_contract_slots(contract.answer_mode)
                        if key in selected_slot_keys and key in contract.slots
                    ],
                    "rationale": "exactness_minimal_keep_set",
                    "required_missing_slots": sorted(missing_slots),
                    "core_slots": sorted(_core_contract_slot_keys(contract)),
                },
                "render_contract": {
                    "answer_mode": render_contract.answer_mode,
                    "should_abstain": render_contract.should_abstain,
                    "abstain_reason": render_contract.abstain_reason,
                    "slots": {
                        key: {
                            "text": value.text,
                            "citation_chunk_ids": value.citation_chunk_ids,
                        }
                        for key, value in render_contract.slots.items()
                    },
                },
                "planner_framework": "instructor_cohere_pydantic",
                "planner_model": settings.cohere_query_planner_model,
                "baseline_model": baseline_result.model_name,
                "render_mode": "deterministic",
            },
            timings={
                "baseline_answer_seconds": baseline_end - baseline_start,
                "structured_contract_seconds": contract_end - contract_start,
                "slot_coverage_verdict_seconds": coverage_end - coverage_start,
                "missing_detail_exactness_verdict_seconds": exactness_end - exactness_start,
                "slot_selection_seconds": selection_end - selection_start,
                "deterministic_render_seconds": render_end - render_start,
            },
            abstained=render_contract.should_abstain,
        )

    raw_response = dict(baseline_result.raw_response or {})
    raw_response["baseline_answer_text"] = baseline_result.answer_text
    raw_response["selected_path"] = "baseline_keep"
    raw_response["slot_coverage_verdict"] = verdict_snapshot
    raw_response["slot_coverage_verdict_payload"] = coverage_payload.model_dump()
    raw_response["missing_detail_exactness_verdict_payload"] = exactness_payload.model_dump()
    raw_response["structured_contract"] = contract_snapshot
    raw_response["structured_contract_payload"] = contract_payload.model_dump()
    timings = dict(baseline_result.timings)
    timings["baseline_answer_seconds"] = baseline_end - baseline_start
    timings["structured_contract_seconds"] = contract_end - contract_start
    timings["slot_coverage_verdict_seconds"] = coverage_end - coverage_start
    timings["missing_detail_exactness_verdict_seconds"] = exactness_end - exactness_start
    return AnswerResult(
        question=baseline_result.question,
        answer_text=baseline_result.answer_text,
        strategy_name="missing_detail_exactness_verifier_gated_structured_contract_inline_evidence_chat",
        model_name=baseline_result.model_name,
        citations=baseline_result.citations,
        evidence_bundle=baseline_result.evidence_bundle,
        raw_response=raw_response,
        timings=timings,
        abstained=baseline_result.abstained,
    )


def structured_contract_mode_aware_inline_evidence_chat(
    settings: Settings,
    question: str,
    evidence: EvidenceBundle,
) -> AnswerResult:
    client = cohere.ClientV2(settings.cohere_api_key)
    plan_start = perf_counter()
    contract_response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[ct.UserChatMessageV2(content=_build_structured_answer_contract_prompt(question, evidence))],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=700,
    )
    contract_text = _extract_text_from_chat_response(contract_response)
    contract = _normalize_structured_answer_contract(contract_text)
    plan_end = perf_counter()
    route = _select_structured_contract_answer_route(question, evidence, contract)
    answer_response = client.chat(
        model=settings.cohere_chat_model,
        messages=[ct.UserChatMessageV2(content=route.prompt)],
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_output_tokens,
    )
    answer_end = perf_counter()
    return AnswerResult(
        question=question,
        answer_text=_extract_text_from_chat_response(answer_response),
        strategy_name="structured_contract_mode_aware_inline_evidence_chat",
        model_name=settings.cohere_chat_model,
        citations=_build_citations(evidence.packed_chunks),
        evidence_bundle=evidence,
        raw_response={
            "structured_answer_contract": {
                "answer_mode": contract.answer_mode,
                "should_abstain": contract.should_abstain,
                "abstain_reason": contract.abstain_reason,
                "slots": contract.slots,
            },
            "selected_path": route.selected_path,
            "planner_model": settings.cohere_query_planner_model,
            "answer_model": settings.cohere_chat_model,
        },
        timings={
            "answer_plan_seconds": plan_end - plan_start,
            "final_answer_seconds": answer_end - plan_end,
        },
        abstained=route.abstained,
    )


answer_strategy_registry.register("inline_evidence_chat", inline_evidence_chat)
answer_strategy_registry.register("structured_inline_evidence_chat", structured_inline_evidence_chat)
answer_strategy_registry.register("documents_chat", documents_chat)
answer_strategy_registry.register("query_guided_inline_evidence_chat", query_guided_inline_evidence_chat)
answer_strategy_registry.register("planned_inline_evidence_chat", planned_inline_evidence_chat)
answer_strategy_registry.register("mode_aware_planned_inline_evidence_chat", mode_aware_planned_inline_evidence_chat)
answer_strategy_registry.register(
    "selective_mode_aware_planned_inline_evidence_chat",
    selective_mode_aware_planned_inline_evidence_chat,
)
answer_strategy_registry.register(
    "selective_mode_aware_compact_inline_evidence_chat",
    selective_mode_aware_compact_inline_evidence_chat,
)
answer_strategy_registry.register(
    "selective_mode_aware_answer_repair_inline_evidence_chat",
    selective_mode_aware_answer_repair_inline_evidence_chat,
)
answer_strategy_registry.register(
    "structured_contract_deterministic_inline_evidence_chat",
    structured_contract_deterministic_inline_evidence_chat,
)
answer_strategy_registry.register(
    "selective_workflow_contract_inline_evidence_chat",
    selective_workflow_contract_inline_evidence_chat,
)
answer_strategy_registry.register(
    "verifier_gated_structured_contract_inline_evidence_chat",
    verifier_gated_structured_contract_inline_evidence_chat,
)
answer_strategy_registry.register(
    "contract_aware_verifier_gated_structured_contract_inline_evidence_chat",
    contract_aware_verifier_gated_structured_contract_inline_evidence_chat,
)
answer_strategy_registry.register(
    "contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat",
    contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat,
)
answer_strategy_registry.register(
    "narrow_contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat",
    narrow_contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat,
)
answer_strategy_registry.register(
    "missing_detail_exactness_verifier_gated_structured_contract_inline_evidence_chat",
    missing_detail_exactness_verifier_gated_structured_contract_inline_evidence_chat,
)
answer_strategy_registry.register(
    "structured_contract_mode_aware_inline_evidence_chat",
    structured_contract_mode_aware_inline_evidence_chat,
)
