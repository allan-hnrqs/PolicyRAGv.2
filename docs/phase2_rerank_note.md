# Phase 2 Comparison Note: Bounded Cohere Rerank and Parallel Query Branches

This note compares the Phase 1 vector checkpoint `2033b18` against the new rerank-backed profiles:

- `baseline_vector_rerank`
- `demo_vector_rerank`

## What changed
- Fixed retriever semantics so `rerank_top_n: 0` truly disables Cohere rerank instead of silently reranking the full candidate slice.
- Added profile-controlled rerank budgets for baseline and demo vector profiles.
- Enabled parallel query branches in the rerank experiment profiles.

## Artifacts
- Dev retrieval benchmark:
  - `datasets/runs/retrieval_benchmark/baseline_vector_rerank_retrieval_profile_20260402_170002_420763_f544.json`
- Holdout retrieval benchmark:
  - `datasets/runs/retrieval_benchmark/baseline_vector_rerank_retrieval_profile_20260402_165957_961778_e5d4.json`
- Dev judged eval:
  - `datasets/runs/baseline_vector_rerank_20260402_170454_996614_ae6a.json`
- Holdout judged eval:
  - `datasets/runs/baseline_vector_rerank_20260402_170515_440075_4b4b.json`
- Product benchmark:
  - `datasets/runs/product_benchmark/product_benchmark_demo_vector_rerank_20260402_171030_587445_7663.json`
- Multi-turn benchmark:
  - `datasets/runs/product_benchmark/product_benchmark_demo_vector_rerank_20260402_170828_310649_e163.json`

## Comparison
Phase 1 vector checkpoint:
- Dev retrieval packed expected URL recall: `0.8889`
- Holdout retrieval packed expected URL recall: `0.75`
- Dev judged eval required-claim recall: `0.8333`
- Holdout judged eval required-claim recall: `0.75`
- Demo product benchmark total request mean: `7.52s`
- Demo product benchmark total request p95: `14.61s`

Phase 2 rerank experiment:
- Dev retrieval packed expected URL recall: `0.7778`
- Holdout retrieval packed expected URL recall: `0.75`
- Dev judged eval required-claim recall: `0.7778`
- Holdout judged eval required-claim recall: `0.725`
- Demo product benchmark total request mean: `23.38s`
- Demo product benchmark total request p95: `57.22s`

## Read
- The semantic retriever fix is correct and should stay.
- The rerank experiment itself is not a win.
- Dev retrieval regressed relative to Phase 1.
- Holdout retrieval did not improve.
- Judged eval regressed on both dev and holdout.
- Product latency became much worse, with answer-generation time dominating the tail.
- Parallel subquery execution did not offset the extra answer-side and rerank cost enough to matter.

## Decision
- Keep the retriever semantics fix for `rerank_top_n`.
- Keep the rerank experiment profiles as reversible comparison artifacts.
- Do not promote reranking as the default serving path.
- Move to Phase 3 span-aware evidence shaping from the Phase 1 vector checkpoint, not from the Phase 2 rerank profiles.
