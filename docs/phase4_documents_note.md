# Phase 4 Comparison Note: Grounded `documents` Answering

This note compares the Phase 1 vector checkpoint `2033b18` against the grounded-document profiles:

- `baseline_vector_documents`
- `demo_vector_documents`

## What changed
- Kept the Phase 1 Elasticsearch native vector retrieval path.
- Switched final answer generation to Cohere grounded `documents`.
- Added profile-scoped runtime settings so answer model, planner model, `max_doc_chars`, and `max_packed_docs` are actually honored.
- Applied the planner-model override to both evaluation and live demo contextualization/query planning.
- Used `command-r7b-12-2024` as the lighter planner model and `command-a-03-2025` as the final answer model in the new profiles.
- Strengthened the grounded-document answer contract to explicitly require multi-part coverage and exactness discipline.

## Artifacts
- Dev judged eval:
  - `datasets/runs/baseline_vector_documents_20260402_175901_174103_07b0.json`
- Holdout judged eval:
  - `datasets/runs/baseline_vector_documents_20260402_180250_893117_2d9d.json`
- Product benchmark:
  - `datasets/runs/product_benchmark/product_benchmark_demo_vector_documents_20260402_180025_290320_20c2.json`
- Multi-turn benchmark:
  - `datasets/runs/product_benchmark/product_benchmark_demo_vector_documents_20260402_175945_904015_e566.json`

## Comparison
Phase 1 vector checkpoint:
- Dev judged eval required-claim recall: `0.8333`
- Holdout judged eval required-claim recall: `0.75`
- Demo product benchmark total request mean: `7.52s`
- Demo multi-turn total request mean: `8.87s`

Phase 4 grounded-document experiment:
- Dev judged eval required-claim recall: `0.5556`
- Holdout judged eval required-claim recall: `0.5083`
- Demo product benchmark total request mean: `5.52s`
- Demo multi-turn total request mean: `4.05s`

## Read
- This is a real serving-latency win.
- It is also the first path that gives acceptable chat-shell behavior without a separate greeting gate.
- The planner-model split lowered contextualization and overall request cost materially.
- But canonical answer quality collapsed.
- The grounded `documents` path is under-covering compound benchmark questions and still overstates some exact details.

Representative behavior:
- `CHAT_001`: now responds naturally with a greeting instead of a 12-citation RAG answer.
- `HR_001`: fast but too narrow; it drops required late-offer handling detail.
- `SO_001` and `SO_002`: faster and cleaner, but too willing to say the documents do not explicitly state distinctions that the benchmark expects the system to synthesize from the evidence.
- `EX_001`: still incorrectly states `Schedule 3 Specimen Signature Card Template` as the exact requested form.

## Decision
- Keep the profile-scoped runtime settings fix.
- Keep the vector+documents profiles as reversible serving experiments.
- Do not promote grounded `documents` answering as the default path yet.
- Move to Phase 5 conversation-state work only if we accept that Phase 4 remains a product-serving experiment rather than a benchmark-quality replacement.

## Follow-Up Postmortem
Additional controlled replay work was run after the original Phase 4 note to remove the main confounds.

What was fixed before replaying:
- citation parsing in the `documents` path now reads Cohere v2 citation `sources`
- inline and `documents` answer paths now share the same core behavioral contract
- answer-only replay benchmarks now freeze the Phase 1 evidence bundle and resolved question so retrieval and planning do not vary

Additional artifacts:
- holdout inline replay:
  - `datasets/runs/answer_replay_benchmark/answer_replay_inline_evidence_chat_20260403_000626_349617_6f62.json`
- holdout grounded `documents` replay with shared contract:
  - `datasets/runs/answer_replay_benchmark/answer_replay_documents_chat_20260403_010651_063654_0dde.json`
- dev inline replay:
  - `datasets/runs/answer_replay_benchmark/answer_replay_inline_evidence_chat_20260403_001144_098795_ee80.json`
- dev grounded `documents` replay with shared contract:
  - `datasets/runs/answer_replay_benchmark/answer_replay_documents_chat_20260403_010555_542873_eb59.json`
- holdout synthetic single-blob grounded-document replay:
  - `datasets/runs/answer_replay_benchmark/answer_replay_documents_inline_blob_chat_20260403_013641_951911_2782.json`
- dev synthetic single-blob grounded-document replay:
  - `datasets/runs/answer_replay_benchmark/answer_replay_documents_inline_blob_chat_20260403_013628_010567_5077.json`
- holdout inline structured-contract replay:
  - `datasets/runs/answer_replay_benchmark/answer_replay_structured_contract_deterministic_inline_evidence_chat_20260403_014902_828516_fbe2.json`
- holdout grounded structured-contract replay:
  - `datasets/runs/answer_replay_benchmark/answer_replay_structured_contract_deterministic_documents_chat_20260403_014852_592549_bb80.json`
- dev inline structured-contract replay:
  - `datasets/runs/answer_replay_benchmark/answer_replay_structured_contract_deterministic_inline_evidence_chat_20260403_014923_442357_9ef2.json`
- dev grounded structured-contract replay:
  - `datasets/runs/answer_replay_benchmark/answer_replay_structured_contract_deterministic_documents_chat_20260403_014904_987660_3a1b.json`

Read from the follow-up:
- the original zero-citation story was partly a repo bug, not a Cohere limitation
- trimming was not the cause on the cleaned corpus
- per-chunk `documents[]` packaging was not the main cause; the single-blob replay did not recover inline quality
- a structured grounded-doc contract also underperformed badly

Follow-up comparison:
- holdout:
  - inline free-form recall: `0.775`
  - grounded `documents` free-form recall: `0.675`
  - inline structured-contract recall: `0.800`
  - grounded structured-contract recall: `0.5833`
- dev:
  - inline free-form recall: `0.8333`
  - grounded `documents` free-form recall: `0.7778`
  - inline structured-contract recall: `0.8333`
  - grounded structured-contract recall: `0.5833`

Revised decision:
- keep `inline_evidence_chat` as the stronger mainline answer path
- keep the citation-parsing fix and replay tooling
- treat Cohere grounded `documents` as an informative but unsuccessful experiment for this repo’s synthesis-heavy task
- do not continue Phase 4 as the mainline serving path unless a future experiment introduces materially new evidence
