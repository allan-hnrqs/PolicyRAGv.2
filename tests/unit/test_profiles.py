from pathlib import Path

from bgrag.config import Settings
from bgrag.profiles.loader import list_profiles, load_profile

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_baseline_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline", settings)
    assert profile.name == "baseline"
    assert profile.retrieval.source_topology == "bg_primary_support_fallback"


def test_profile_listing_includes_baseline() -> None:
    settings = Settings(project_root=REPO_ROOT)
    assert "baseline" in list_profiles(settings)


def test_mode_aware_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("mode_aware_planned_answering", settings)
    assert profile.answering.strategy == "mode_aware_planned_inline_evidence_chat"


def test_selective_mode_aware_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("selective_mode_aware_planned_answering", settings)
    assert profile.answering.strategy == "selective_mode_aware_planned_inline_evidence_chat"


def test_selective_mode_aware_compact_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("selective_mode_aware_compact_answering", settings)
    assert profile.answering.strategy == "selective_mode_aware_compact_inline_evidence_chat"


def test_structured_contract_mode_aware_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("structured_contract_mode_aware_answering", settings)
    assert profile.answering.strategy == "structured_contract_mode_aware_inline_evidence_chat"


def test_structured_contract_deterministic_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("structured_contract_deterministic_answering", settings)
    assert profile.answering.strategy == "structured_contract_deterministic_inline_evidence_chat"


def test_selective_workflow_contract_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("selective_workflow_contract_answering", settings)
    assert profile.answering.strategy == "selective_workflow_contract_inline_evidence_chat"


def test_verifier_gated_structured_contract_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("verifier_gated_structured_contract_answering", settings)
    assert profile.answering.strategy == "verifier_gated_structured_contract_inline_evidence_chat"


def test_contract_aware_verifier_gated_structured_contract_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("contract_aware_verifier_gated_structured_contract_answering", settings)
    assert profile.answering.strategy == "contract_aware_verifier_gated_structured_contract_inline_evidence_chat"


def test_contract_slot_coverage_verifier_gated_structured_contract_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("contract_slot_coverage_verifier_gated_structured_contract_answering", settings)
    assert profile.answering.strategy == "contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat"


def test_narrow_contract_slot_coverage_verifier_gated_structured_contract_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("narrow_contract_slot_coverage_verifier_gated_structured_contract_answering", settings)
    assert profile.answering.strategy == "narrow_contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat"


def test_missing_detail_exactness_verifier_gated_structured_contract_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("missing_detail_exactness_verifier_gated_structured_contract_answering", settings)
    assert profile.answering.strategy == "missing_detail_exactness_verifier_gated_structured_contract_inline_evidence_chat"


def test_selective_mode_aware_answer_repair_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("selective_mode_aware_answer_repair", settings)
    assert profile.answering.strategy == "selective_mode_aware_answer_repair_inline_evidence_chat"


def test_demo_documents_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("demo_documents", settings)
    assert profile.answering.strategy == "documents_chat"
    assert profile.retrieval.enable_query_decomposition is False


def test_baseline_documents_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_documents", settings)
    assert profile.answering.strategy == "documents_chat"
    assert profile.retrieval.enable_query_decomposition is True


def test_baseline_vector_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.retrieval.rerank_top_n == 0


def test_baseline_vector_rerank_all_corpus_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_all_corpus", settings)
    assert profile.retrieval.retrieval_mode == "rerank_all_corpus"
    assert profile.retrieval.enable_query_decomposition is False
    assert profile.retrieval.rerank_top_n == 0


def test_demo_vector_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("demo_vector", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.retrieval.enable_query_decomposition is False


def test_baseline_vector_rerank_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.retrieval.rerank_top_n == 24
    assert profile.retrieval.enable_parallel_query_branches is True


def test_baseline_vector_rerank_shortlist_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.retrieval.rerank_top_n == 48
    assert profile.retrieval.enable_parallel_query_branches is False


def test_baseline_vector_rerank_shortlist_hybrid_retry_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_hybrid_retry", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.retrieval.rerank_top_n == 48
    assert profile.retrieval.enable_retrieval_assessment is True
    assert profile.retrieval.enable_hybrid_retry_trigger is True
    assert profile.retrieval.enable_official_site_escalation is False
    assert profile.retrieval.retry_candidate_k == 64
    assert profile.retrieval.retry_rerank_top_n == 64
    assert profile.retrieval.retry_per_query_candidate_k == 32


def test_baseline_vector_rerank_shortlist_ranked_diverse_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_ranked_diverse", settings)
    assert profile.retrieval.source_topology == "ranked_passthrough"
    assert profile.retrieval.rerank_top_n == 48
    assert profile.retrieval.enable_ranked_chunk_diversity is True
    assert profile.retrieval.max_chunks_per_heading == 3


def test_baseline_vector_rerank_shortlist_opensearch_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_opensearch", settings)
    assert profile.retrieval.search_backend == "opensearch"
    assert profile.retrieval.dense_retrieval_backend == "opensearch_knn"
    assert profile.retrieval.rerank_top_n == 48


def test_baseline_vector_rerank_shortlist_agentic_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_agentic", settings)
    assert profile.retrieval.hybrid_fusion_mode == "es_rrf"
    assert profile.retrieval.enable_retrieval_assessment is True
    assert profile.retrieval.enable_official_site_escalation is True


def test_demo_vector_rerank_shortlist_agentic_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("demo_vector_rerank_shortlist_agentic", settings)
    assert profile.retrieval.hybrid_fusion_mode == "es_rrf"
    assert profile.retrieval.enable_query_decomposition is False
    assert profile.retrieval.enable_official_site_escalation is True


def test_baseline_vector_rerank_wide_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_wide", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.retrieval.candidate_k == 96
    assert profile.retrieval.rerank_top_n == 96
    assert profile.retrieval.per_query_candidate_k == 48


def test_baseline_vector_rerank_wide_answer_repair_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_wide_answer_repair", settings)
    assert profile.retrieval.candidate_k == 96
    assert profile.retrieval.rerank_top_n == 96
    assert profile.answering.strategy == "selective_mode_aware_answer_repair_inline_evidence_chat"


def test_baseline_vector_rerank_shortlist_structured_contract_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_structured_contract", settings)
    assert profile.retrieval.rerank_top_n == 48
    assert profile.answering.strategy == "structured_contract_deterministic_inline_evidence_chat"


def test_baseline_vector_rerank_shortlist_selective_workflow_contract_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_selective_workflow_contract", settings)
    assert profile.retrieval.rerank_top_n == 48
    assert profile.answering.strategy == "selective_workflow_contract_inline_evidence_chat"


def test_baseline_vector_rerank_shortlist_narrow_contract_gate_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_narrow_contract_gate", settings)
    assert profile.retrieval.rerank_top_n == 48
    assert (
        profile.answering.strategy
        == "narrow_contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat"
    )


def test_demo_vector_rerank_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("demo_vector_rerank", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.retrieval.rerank_top_n == 18
    assert profile.retrieval.enable_parallel_query_branches is True


def test_baseline_vector_spans_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_spans", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.answering.evidence_unit == "span"
    assert profile.answering.span_rerank_top_n == 16


def test_demo_vector_spans_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("demo_vector_spans", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.answering.evidence_unit == "span"
    assert profile.answering.span_rerank_top_n == 12


def test_baseline_vector_documents_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_documents", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.answering.strategy == "documents_chat"
    assert profile.answering.planner_model_name == "command-r7b-12-2024"


def test_demo_vector_documents_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("demo_vector_documents", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.answering.strategy == "documents_chat"
    assert profile.answering.planner_model_name == "command-r7b-12-2024"


def test_hierarchical_context_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("hierarchical_context_expansion", settings)
    assert profile.retrieval.enable_page_intro_expansion is True
    assert profile.retrieval.enable_document_context_expansion is True


def test_structural_context_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("structural_context_expansion", settings)
    assert profile.retrieval.enable_structural_context_augmentation is True


def test_page_seed_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("page_seed_retrieval", settings)
    assert profile.retrieval.enable_document_seed_retrieval is True


def test_document_rerank_seed_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("document_rerank_seed_retrieval", settings)
    assert profile.retrieval.document_seed_ranking_mode == "rerank_docs"


def test_localized_document_rerank_seed_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("localized_document_rerank_seed_retrieval", settings)
    assert profile.retrieval.document_seed_scope == "local_graph"


def test_lineage_document_rerank_seed_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("lineage_document_rerank_seed_retrieval", settings)
    assert profile.retrieval.document_seed_scope == "local_lineage"


def test_baseline_vector_rerank_shortlist_localized_seed_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_localized_seed", settings)
    assert profile.retrieval.rerank_top_n == 48
    assert profile.retrieval.enable_document_seed_retrieval is True
    assert profile.retrieval.document_seed_scope == "local_graph"


def test_baseline_vector_rerank_shortlist_hierarchical_context_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("baseline_vector_rerank_shortlist_hierarchical_context", settings)
    assert profile.retrieval.rerank_top_n == 48
    assert profile.retrieval.enable_page_intro_expansion is True
    assert profile.retrieval.enable_document_context_expansion is True


def test_selective_localized_document_rerank_seed_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("selective_localized_document_rerank_seed_retrieval", settings)
    assert profile.retrieval.enable_retrieval_mode_selection is True
    assert profile.retrieval.enable_document_seed_retrieval is True
