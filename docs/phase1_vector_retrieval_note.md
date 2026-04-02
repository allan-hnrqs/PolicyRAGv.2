# Phase 1 Comparison Note: Elasticsearch Native Vector Retrieval

This note compares the clean checkpoint `58746ac` against the new vector-backed profiles:

- `baseline_vector`
- `demo_vector`

## What changed
- Added Elasticsearch indexed dense vectors to chunk indices.
- Added an `elasticsearch_knn` dense retrieval backend behind profile configuration.
- Kept the old local embedding-store path intact for rollback.
- Added retrieval stage timings for lexical search, vector search, fusion, rerank, and packing.

## Artifacts
- Dev retrieval benchmark:
  - `datasets/runs/retrieval_benchmark/baseline_vector_retrieval_profile_20260402_164505_428859_c7d0.json`
- Holdout retrieval benchmark:
  - `datasets/runs/retrieval_benchmark/baseline_vector_retrieval_profile_20260402_164530_251244_2f18.json`
- Dev judged eval:
  - `datasets/runs/baseline_vector_20260402_164900_370690_4a15.json`
- Holdout judged eval:
  - `datasets/runs/baseline_vector_20260402_165200_822121_7d87.json`
- Product benchmark:
  - `datasets/runs/product_benchmark/product_benchmark_demo_vector_20260402_165435_391309_df5e.json`
- Multi-turn benchmark:
  - `datasets/runs/product_benchmark/product_benchmark_demo_vector_20260402_165354_890071_a519.json`

## Comparison
Checkpoint anchor:
- Clean holdout judged eval required-claim recall: `0.7167`
- Clean holdout retrieval packed expected URL recall: `0.7833`
- Clean demo product benchmark total request mean: `12.01s`

Phase 1:
- Dev retrieval packed expected URL recall: `0.8889`
- Holdout retrieval packed expected URL recall: `0.75`
- Dev judged eval required-claim recall: `0.8333`
- Holdout judged eval required-claim recall: `0.75`
- Demo product benchmark total request mean: `7.52s`
- Demo product benchmark total request p95: `14.61s`

## Read
- This is a real latency win.
- This is also a real retrieval speed win.
- Dev retrieval improved materially.
- Holdout retrieval did not hold the clean checkpoint level.
- Holdout judged quality improved slightly relative to the clean checkpoint, but not enough to ignore the retrieval regression.
- Chat-shell behavior is still bad, so this is not a product-serving promotion by itself.

## Decision
- Keep the Phase 1 vector path.
- Do not promote it as the default baseline yet.
- Use it as the new reversible checkpoint for Phase 2 rerank work.
