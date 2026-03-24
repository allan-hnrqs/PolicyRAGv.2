# Parity Contract

This file records the trustworthy benchmark surfaces and baseline assumptions
for phase 1. It exists to prevent drift across long implementation sessions.

## Canonical phase-1 benchmark surface

The current trustworthy phase-1 benchmark is the original `19`-case
human-realistic Buyer’s Guide suite derived from:

- `cross_eval/human_realistic_buyers_guide_cases.jsonl`
- `cross_eval/generated_cases/feat_retrieval_human_realistic.jsonl`

The canonical split membership is:

- dev:
  - `HR_001`
  - `HR_003`
  - `HR_005`
  - `HR_007`
  - `HR_009`
  - `HR_010`
  - `HR_013`
  - `HR_015`
  - `HR_017`
- frozen holdout:
  - `HR_002`
  - `HR_004`
  - `HR_006`
  - `HR_008`
  - `HR_011`
  - `HR_012`
  - `HR_014`
  - `HR_016`
  - `HR_018`
  - `HR_019`
- final blind acceptance set:
  - not built yet in this repo
  - must be authored separately from dev/holdout and kept sealed until milestone evaluation

The canonical split files in this repo should be derived from `datasets/eval/parity/parity19.jsonl`,
not copied from older `feat` dev/holdout artifacts:

- dev:
  - `datasets/eval/dev/parity19_dev.jsonl`
- frozen holdout:
  - `datasets/eval/holdout/parity19_holdout.jsonl`

Legacy note:

- `datasets/eval/parity/dev_feat.jsonl` is not a trustworthy structured-judge tuning set because many cases there do not contain `required_claims`.
- `datasets/eval/parity/holdout_feat.jsonl` is still useful historically, but the new canonical holdout should come from the fully annotated `parity19.jsonl`.

## Current trustworthy feat reference runs

Use the rebased/current-truth run surfaces, not historical promotion passes:

- `cross_eval/runs/feat_retrieval_human_realistic_fullannotated_v1.json`
- `cross_eval/runs/feat_retrieval_human_realistic_dev_rebased_v3.json`
- `cross_eval/runs/feat_retrieval_human_realistic_holdout_rebased_v2.json`

## Runtime profile assumptions to preserve

Baseline runtime assumptions for parity:

- `top_k=16`
- `retrieval_candidate_k=48`
- `retrieval_alpha=0.70`
- query rewrite off
- rerank on
- `rerank_alpha=0.20`
- `rerank_top_n=0`
- MMR on
- `mmr_lambda=0.75`
- source-priority retrieval on
- source-priority packing on
- `pack_coverage_aware=true`
- `max_packed_docs=24`
- `max_doc_tokens=320`
- `chat_temperature=0.0`

Baseline model assumptions:

- answer model: `command-r-plus-08-2024`
- judge model: `command-a-03-2025`

## Tuning policy

Architectural work is allowed to include aggressive parameter tuning when the
architecture genuinely depends on those parameters. Token cost is not the main
constraint, but tuning still has to stay honest:

- use dev cases or explicitly declared experiment slices for tuning
- do not tune directly on frozen holdout
- record tuned parameters, search ranges, and acceptance rationale
- validate every accepted tuning change on the frozen holdout before promotion

The final blind acceptance set is stricter than holdout:

- do not use it during active tuning
- do not inspect per-case failures while the architecture is still changing rapidly
- use it only for milestone acceptance checks after a candidate architecture has already cleared dev and holdout
- when a final check fails, respond by creating new non-final challenge cases rather than iterating directly on the sealed set

## Explicit non-canonical surfaces

The rebuilt `39`-case path now exists locally and is used as the main expanded
architecture-validation surface. It was rebuilt in this repository rather than
inherited from orphaned lab artifacts.

Current rebuilt-39 artifacts:

- `datasets/eval/parity/parity39_working.jsonl`
- `datasets/eval/dev/parity39_dev_draft.jsonl`
- `datasets/eval/holdout/parity39_holdout_draft.jsonl`
- `docs/eval_authoring.md`
- `datasets/eval/manifests/eval_case_template.json`
- `datasets/eval/manifests/parity39_blueprint.json`
- `datasets/eval/manifests/final_acceptance_blueprint.json`

Important note:

- the rebuilt `39`-case suite is now trustworthy enough for architecture
  validation, but the original `19`-case parity suite still remains the formal
  phase-1 control benchmark for promotion decisions
