"""Typed runtime profile models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalProfile(BaseModel):
    source_topology: str = "bg_primary_support_fallback"
    retrieval_mode: str = "hybrid_es_rerank"
    dense_retrieval_backend: str = "local_embedding_store"
    top_k: int = 16
    candidate_k: int = 48
    retrieval_alpha: float = 0.7
    rerank_top_n: int = 16
    es_knn_num_candidates: int = 120
    enable_mmr_diversity: bool = False
    mmr_lambda: float = 0.75
    enable_ranked_chunk_diversity: bool = False
    diversity_cover_fraction: float = 0.5
    max_chunks_per_document: int = 8
    max_chunks_per_heading: int = 4
    seed_chunks_per_heading: int = 2
    enable_query_decomposition: bool = False
    max_expanded_queries: int = 2
    per_query_candidate_k: int = 24
    query_fusion_rrf_k: int = 60
    enable_parallel_query_branches: bool = False
    enable_retrieval_mode_selection: bool = False
    retrieval_mode_selector_max_chunks: int = 10
    enable_page_intro_expansion: bool = False
    page_intro_candidate_k: int = 8
    page_intro_max_order: int = 10
    enable_document_context_expansion: bool = False
    document_context_seed_docs: int = 2
    document_context_candidate_k: int = 12
    document_context_neighbor_docs: int = 2
    enable_structural_context_augmentation: bool = False
    structural_context_seed_docs: int = 2
    structural_context_intro_max_order: int = 10
    structural_context_same_heading_k: int = 2
    structural_context_nearby_k: int = 3
    structural_context_nearby_window: int = 12
    structural_context_neighbor_docs: int = 2
    enable_document_seed_retrieval: bool = False
    document_seed_ranking_mode: str = "intro_pool"
    document_seed_scope: str = "corpus"
    document_seed_scope_docs: int = 4
    document_seed_docs: int = 3
    document_seed_intro_max_order: int = 10
    document_seed_intro_chunks: int = 3
    document_seed_candidate_k: int = 12
    document_seed_max_chars: int = 1400


class ChunkingProfile(BaseModel):
    chunker: str = "section_chunker"
    metadata_enrichers: list[str] = Field(
        default_factory=lambda: [
            "authority_metadata",
            "lineage_metadata",
            "scope_tag_metadata",
            "source_topology_metadata",
        ]
    )
    sliding_window_chars: int = 1200
    sliding_window_overlap: int = 200


class AnswerProfile(BaseModel):
    strategy: str = "inline_evidence_chat"
    max_packed_docs: int = 24
    max_doc_chars: int = 1600
    model_name: str = "command-a-03-2025"
    planner_model_name: str | None = None
    evidence_unit: str = "chunk"
    span_max_chars: int = 320
    span_candidate_chunks: int = 8
    span_max_per_chunk: int = 2
    span_rerank_top_n: int = 0


class EvalProfile(BaseModel):
    suite_name: str = "parity_19"
    judge_model: str = "command-a-03-2025"
    chat_temperature: float = 0.0


class RuntimeProfile(BaseModel):
    name: str
    description: str
    collection_strategy: str = "default_web_collect"
    retrieval: RetrievalProfile = Field(default_factory=RetrievalProfile)
    chunking: ChunkingProfile = Field(default_factory=ChunkingProfile)
    answering: AnswerProfile = Field(default_factory=AnswerProfile)
    evaluation: EvalProfile = Field(default_factory=EvalProfile)
