# Phase 3 Comparison Note: Span-Aware Evidence Packing

This note compares the Phase 1 vector checkpoint `2033b18` against the new span-packing profiles:

- `baseline_vector_spans`
- `demo_vector_spans`

## What changed
- Added answer-time span packing behind profile configuration.
- Kept chunk retrieval unchanged.
- Split only the selected candidate pool into smaller spans for packing.
- Used bounded Cohere rerank only at the span-selection stage.
- Preserved original chunk provenance on derived span records.

## Artifacts
- Dev retrieval benchmark:
  - `datasets/runs/retrieval_benchmark/baseline_vector_spans_retrieval_profile_20260402_172534_640309_2957.json`
- Holdout retrieval benchmark:
  - `datasets/runs/retrieval_benchmark/baseline_vector_spans_retrieval_profile_20260402_172621_058681_4ab8.json`
- Dev judged eval:
  - `datasets/runs/baseline_vector_spans_20260402_172933_767092_530f.json`
- Holdout judged eval:
  - `datasets/runs/baseline_vector_spans_20260402_173409_584224_3380.json`
- Product benchmark:
  - `datasets/runs/product_benchmark/product_benchmark_demo_vector_spans_20260402_173627_207405_df57.json`
- Multi-turn benchmark:
  - `datasets/runs/product_benchmark/product_benchmark_demo_vector_spans_20260402_173723_725890_1afd.json`

## Comparison
Phase 1 vector checkpoint:
- Dev retrieval packed expected URL recall: `0.8889`
- Holdout retrieval packed expected URL recall: `0.75`
- Dev judged eval required-claim recall: `0.8333`
- Holdout judged eval required-claim recall: `0.75`
- Demo product benchmark total request mean: `7.52s`
- Demo multi-turn total request mean: `8.87s`

Phase 3 span-packing experiment:
- Dev retrieval packed expected URL recall: `0.7778`
- Holdout retrieval packed expected URL recall: `0.75`
- Dev judged eval required-claim recall: `0.5000`
- Holdout judged eval required-claim recall: `0.6833`
- Demo product benchmark total request mean: `11.59s`
- Demo multi-turn total request mean: `9.18s`

## Read
- This is not a win.
- Packed MRR improved because the packed span set often put one relevant span at rank 1.
- That did not translate into better expected-URL recall or better answer quality.
- Fragmenting the evidence made the answer model miss multi-claim coverage and exact branch conditions.
- Product latency also worsened because answer generation still dominated and the extra packing step did not buy enough.

Representative failures:
- `HR_001`: late-offer proof handling became vague and lost the operational proof requirements.
- `HR_007`: supply-arrangement trade-agreement coverage weakened badly.
- `HR_017`: the system still drifted into unsupported contact guidance despite the narrower span set.

## Decision
- Keep the implementation as a reversible experiment only.
- Do not promote span-aware packing as the default serving path.
- Move to Phase 4 grounded `documents` answering instead of iterating further on this span-packing design.
