"""Shared domain models for the Buyer’s Guide RAG system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceFamily(str, Enum):
    BUYERS_GUIDE = "buyers_guide"
    BUY_CANADIAN_POLICY = "buy_canadian_policy"
    TBS_DIRECTIVE = "tbs_directive"


class SourceLink(BaseModel):
    title: str
    url: str
    canonical_url: str | None = None
    in_scope: bool = True


class SourceGraph(BaseModel):
    parent_url: str | None = None
    parent_doc_id: str | None = None
    child_urls: list[str] = Field(default_factory=list)
    child_doc_ids: list[str] = Field(default_factory=list)
    lineage_urls: list[str] = Field(default_factory=list)
    lineage_doc_ids: list[str] = Field(default_factory=list)
    depth: int = 0
    incoming_in_scope_links: list[str] = Field(default_factory=list)
    outgoing_in_scope_links: list[str] = Field(default_factory=list)


class StructureBlock(BaseModel):
    block_id: str
    block_type: str
    heading: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    text: str
    order: int
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    source_url: HttpUrl
    fetched_at: datetime
    final_url: HttpUrl | None = None
    status_code: int | None = None
    html: str
    headers: dict[str, str] = Field(default_factory=dict)
    discovered_links: list[str] = Field(default_factory=list)


class NormalizedDocument(BaseModel):
    doc_id: str
    title: str
    source_url: str
    canonical_url: str
    source_family: SourceFamily
    authority_rank: int
    date_modified: str | None = None
    fetched_at: datetime
    content_hash: str
    word_count: int
    extraction_method: str
    breadcrumbs: list[SourceLink] = Field(default_factory=list)
    graph: SourceGraph = Field(default_factory=SourceGraph)
    structure_blocks: list[StructureBlock] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    raw_text: str = ""
    markdown_text: str = ""
    provenance_paths: dict[str, str] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    canonical_url: str
    title: str
    source_family: SourceFamily
    authority_rank: int
    chunker_name: str
    chunk_type: str
    text: str
    heading: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    section_id: str | None = None
    order: int = 0
    token_estimate: int = 0
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RetrievalCandidate(BaseModel):
    chunk: ChunkRecord
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    blended_score: float = 0.0
    source_topology_reason: str | None = None


class EvidenceBundle(BaseModel):
    query: str
    raw_shortlist: list[RetrievalCandidate] = Field(default_factory=list)
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    selected_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    packed_chunks: list[ChunkRecord] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    retrieval_assessment: RetrievalAssessment | None = None
    tool_trace: list[ToolTraceStep] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationState(BaseModel):
    conversation_id: str
    recent_turns: list[ConversationTurn] = Field(default_factory=list)
    active_entities: list[str] = Field(default_factory=list)
    comparison_axes: list[str] = Field(default_factory=list)
    resolved_query: str | None = None


class RetrievalAssessment(BaseModel):
    sufficient_for_answer: bool
    coverage_risk: Literal["low", "medium", "high"]
    exactness_risk: Literal["low", "medium", "high"]
    support_conflict: bool = False
    recommended_next_step: Literal["answer", "retry_retrieve", "browse_official"]
    reasons: list[str] = Field(default_factory=list)
    raw_response: dict[str, object] | None = None


class ToolTraceStep(BaseModel):
    tool: str
    query_or_url: str | None = None
    elapsed_seconds: float | None = None
    result_count: int | None = None
    evidence_origin: str | None = None
    stop_reason: str | None = None


class ServeTrace(BaseModel):
    serve_mode: Literal["indexed_fast", "indexed_retry", "official_browse"]
    escalation_decision: str | None = None
    retrieval_assessment: RetrievalAssessment | None = None
    retry_retrieval_assessment: RetrievalAssessment | None = None
    retry_policy: dict[str, object] | None = None
    conversation_state: ConversationState | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    retrieval_stats: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AnswerCitation(BaseModel):
    chunk_id: str
    canonical_url: str
    snippet: str | None = None


class AnswerResult(BaseModel):
    question: str
    answer_text: str
    strategy_name: str
    model_name: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle | None = None
    raw_response: dict[str, object] | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    abstained: bool = False
    failure_reason: str | None = None
    serve_trace: ServeTrace | None = None


class EvalClaimEvidence(BaseModel):
    claim: str
    evidence_doc_urls: list[str] = Field(default_factory=list)
    evidence_doc_prefixes: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    id: str
    question: str
    persona: str | None = None
    split: str | None = None
    primary_urls: list[str] = Field(default_factory=list)
    supporting_urls: list[str] = Field(default_factory=list)
    expected_primary_source_family: SourceFamily | None = None
    expected_doc_prefixes: list[str] = Field(default_factory=list)
    supporting_doc_prefixes: list[str] = Field(default_factory=list)
    must_include_concepts: list[str] = Field(default_factory=list)
    should_avoid: list[str] = Field(default_factory=list)
    evaluation_focus: str | None = None
    required_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    required_phrase_groups: list[list[str]] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    expect_abstain: bool | None = None
    claim_evidence: list[EvalClaimEvidence] = Field(default_factory=list)
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    restricted_source_valid: bool | None = None
    open_browse_valid: bool | None = None


class EvalCaseResult(BaseModel):
    case: EvalCase
    answer: AnswerResult
    judgment: dict[str, object] | None = None
    metrics: dict[str, float | bool | str | None] = Field(default_factory=dict)


class EvalRunResult(BaseModel):
    run_name: str
    created_at: datetime
    profile_name: str
    answer_model: str
    judge_model: str
    run_manifest: dict[str, object] = Field(default_factory=dict)
    cases: list[EvalCaseResult] = Field(default_factory=list)
    overall_metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RagasCaseResult(BaseModel):
    case_id: str
    split: str | None = None
    question: str
    answer_strategy: str
    answer_text: str
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    packed_chunk_count: int = 0
    candidate_chunk_count: int = 0
    evaluated: bool = True
    skip_reason: str | None = None


class RagasRunResult(BaseModel):
    run_name: str
    created_at: datetime
    profile_name: str
    eval_model: str
    run_manifest: dict[str, object] = Field(default_factory=dict)
    cases: list[RagasCaseResult] = Field(default_factory=list)
    overall_metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PairwiseJudgeVerdict(BaseModel):
    winner: Literal["answer_a", "answer_b", "tie"]
    confidence: Literal["low", "medium", "high"]
    coverage_winner: Literal["answer_a", "answer_b", "tie"]
    faithfulness_winner: Literal["answer_a", "answer_b", "tie"]
    safety_winner: Literal["answer_a", "answer_b", "tie"]
    rationale: str


class PairwiseCaseResult(BaseModel):
    case_id: str
    split: str | None = None
    question: str
    control_run_name: str
    candidate_run_name: str
    answer_a_source: str
    answer_b_source: str
    overall_winner: str
    confidence: str
    coverage_winner: str
    faithfulness_winner: str
    safety_winner: str
    rationale: str
    control_answer_text: str
    candidate_answer_text: str
    cache_hit: bool = False


class PairwiseRunResult(BaseModel):
    run_name: str
    created_at: datetime
    control_run_path: str
    candidate_run_path: str
    judge_model: str
    run_manifest: dict[str, object] = Field(default_factory=dict)
    cases: list[PairwiseCaseResult] = Field(default_factory=list)
    overall_metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
