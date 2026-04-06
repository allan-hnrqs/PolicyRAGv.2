# Retrieval Budget Sweep

This note records a bounded retrieval-budget sweep over the serious rerank-shortlist
lane on the full `19`-case parity surface.

## Scope

- base profile:
  - `baseline_vector_rerank_shortlist`
- eval surface:
  - `datasets/eval/parity/parity19.jsonl`
- query mode:
  - `profile`
- artifact:
  - [baseline_vector_rerank_shortlist_budget_sweep_20260404_053003_793525_b349.md](../datasets/runs/retrieval_budget_sweep/baseline_vector_rerank_shortlist_budget_sweep_20260404_053003_793525_b349.md)

## Variants

- `top16_c48_r48_p24`
  - end-to-end required-claim recall: `0.8289`
  - mean case seconds: `16.07`
  - packed expected-URL recall: `0.8246`
- `top16_c64_r64_p32`
  - end-to-end required-claim recall: `0.8553`
  - mean case seconds: `19.31`
  - packed expected-URL recall: `0.7982`
- `top16_c96_r96_p48`
  - end-to-end required-claim recall: `0.8509`
  - mean case seconds: `26.83`
  - packed expected-URL recall: `0.7982`
- `top24_c64_r64_p32`
  - end-to-end required-claim recall: `0.8158`
  - mean case seconds: `51.17`
  - packed expected-URL recall: `0.8246`
- `top24_c96_r96_p48`
  - end-to-end required-claim recall: `0.8289`
  - mean case seconds: `20.53`
  - packed expected-URL recall: `0.8246`

## Read

- The current shortlist budget was not cleanly tuned before this sweep.
- A modest shortlist increase helps:
  - `top16_c64_r64_p32` is the best quality point in this bounded sweep.
- More retrieval budget does not keep helping:
  - `96/96/48` did not beat `64/64/32`
  - `top_k=24` was not a win
- The quality gains are real but still not enough:
  - the best bounded point reached `0.8553`, not `0.9+`

## Decision

- Keep this sweep as evidence that the shortlist/rerank budget matters.
- `top16_c64_r64_p32` was the leading next retrieval budget point from the full
  parity19 sweep, so it was then validated on the clean dev/holdout split.
- Split validation result:
  - dev improved:
    - baseline rerank-shortlist: `0.8611`
    - tuned candidate: `0.8889`
  - holdout improved slightly on judged end-to-end recall:
    - baseline rerank-shortlist: `0.7750`
    - tuned candidate: `0.8000`
  - but retrieval-only holdout regressed:
    - baseline rerank-shortlist packed expected-URL recall: `0.7667`
    - tuned candidate packed expected-URL recall: `0.7167`
    - baseline rerank-shortlist packed claim-evidence recall: `0.9750`
    - tuned candidate packed claim-evidence recall: `0.8750`
- Decision:
  - do not promote this tuned budget point as the new default serious profile
  - keep it as evidence that shortlist/rerank budgets affect outcomes
  - treat the split result as mixed rather than a clean architectural win
