# Profile Surface

This directory still contains historical experiment profiles because the repo
uses them to reproduce old artifacts. That does **not** mean they are all live
options for new work.

Use this status map instead of guessing from filenames.

## Control

These are the profiles to treat as current control surfaces:

- `baseline`
  - legacy formal control
- `demo`
  - legacy demo control
- `baseline_vector_rerank_shortlist`
  - current serious indexed-RAG control

## Candidate

These are the live candidate profiles for the current tiered architecture work:

- `baseline_vector_rerank_shortlist_agentic`
- `demo_vector_rerank_shortlist_agentic`

## Archived Experiments

These stay loadable for artifact reproduction, but they are not current
promotion candidates.

Documents / grounded-answering:

- `baseline_documents`
- `demo_documents`
- `baseline_vector_documents`
- `demo_vector_documents`

Retrieval experiments:

- `baseline_vector_rerank`
- `baseline_vector_rerank_all_corpus`
- `baseline_vector_rerank_shortlist_hybrid_retry`
- `baseline_vector_rerank_shortlist_authority_reserve`
- `baseline_vector_rerank_shortlist_selective_authority_reserve`
- `baseline_vector_rerank_shortlist_selective_authority_cluster`
- `baseline_vector_rerank_shortlist_opensearch`
- `baseline_vector_rerank_shortlist_ranked_diverse`
- `baseline_vector_rerank_wide`
- `baseline_vector_rerank_wide_answer_repair`
- `baseline_vector_spans`
- `demo_vector_rerank`
- `demo_vector_spans`
- `diverse_packing`
- `unified_source_hybrid`
- `hierarchical_context_expansion`
- `structural_context_expansion`
- `page_seed_retrieval`
- `document_rerank_seed_retrieval`
- `localized_document_rerank_seed_retrieval`
- `lineage_document_rerank_seed_retrieval`
- `selective_localized_document_rerank_seed_retrieval`

Answer-side experiments:

- `planned_answering`
- `mode_aware_planned_answering`
- `query_guided_answering`
- `structured_answering`
- `structured_contract_deterministic_answering`
- `structured_contract_mode_aware_answering`
- `selective_mode_aware_planned_answering`
- `selective_mode_aware_compact_answering`
- `selective_mode_aware_answer_repair`
- `selective_workflow_contract_answering`
- `baseline_vector_rerank_shortlist_structured_contract`
- `baseline_vector_rerank_shortlist_selective_workflow_contract`
- `baseline_vector_rerank_shortlist_narrow_contract_gate`
- `verifier_gated_structured_contract_answering`
- `contract_aware_verifier_gated_structured_contract_answering`
- `contract_slot_coverage_verifier_gated_structured_contract_answering`
- `narrow_contract_slot_coverage_verifier_gated_structured_contract_answering`
- `missing_detail_exactness_verifier_gated_structured_contract_answering`

Older vector/demo variants:

- `baseline_vector`
- `demo_vector`
- `demo_qd`

## Rule

If a profile is not listed under `Control` or `Candidate`, do not treat it as a
current promotion target without first updating:

- [experiment_index.md](../docs/experiment_index.md)
- [decision_log.md](../docs/decision_log.md)
