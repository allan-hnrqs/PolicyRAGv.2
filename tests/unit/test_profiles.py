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


def test_demo_vector_rerank_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("demo_vector_rerank", settings)
    assert profile.retrieval.dense_retrieval_backend == "elasticsearch_knn"
    assert profile.retrieval.rerank_top_n == 18
    assert profile.retrieval.enable_parallel_query_branches is True


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


def test_selective_localized_document_rerank_seed_profile_loads() -> None:
    settings = Settings(project_root=REPO_ROOT)
    profile = load_profile("selective_localized_document_rerank_seed_retrieval", settings)
    assert profile.retrieval.enable_retrieval_mode_selection is True
    assert profile.retrieval.enable_document_seed_retrieval is True
