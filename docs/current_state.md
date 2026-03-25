# Current State

## Goal

Build a clean Buyer’s Guide-first backend that can match or beat the trustworthy
`feat` parity floor without inheriting its code clutter or ad hoc answer logic.

## Repo Role

- primary tracked repo:
  - `c:\Users\14164\Documents\CohereThing\PolicyRAGv.2`
- legacy local fallback:
  - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean`

Unless a task specifically calls for a disposable experiment copy, ongoing work
should now happen in this Git-backed repo.

The evaluation workflow should converge toward three tiers:

- dev for tuning and rapid iteration
- frozen holdout for promotion checks
- final blind acceptance set for end-of-phase quality assessment

## What is implemented

- repo skeleton
- plan/docs scaffold
- typed settings and core domain models
- runtime profiles
- collector
- normalizer
- metadata enrichers
- chunkers
- corpus persistence
- retrieval / answering / eval runtime skeleton
- parity freeze helper
- CLI wiring
- strict live runtime
- namespaced Elasticsearch index snapshots
- run manifests for eval outputs
- detached background eval launcher
- secondary Ragas eval lane
- secondary pairwise A/B eval lane

## What is canonical right now

- Buyer’s Guide-first source topology
- phase-1 answer default should be `inline_evidence_chat`
- phase-1 parity is anchored to the 19-case suite and canonical dev/holdout split
- canonical dev/holdout files now come from:
  - `datasets/eval/dev/parity19_dev.jsonl`
  - `datasets/eval/holdout/parity19_holdout.jsonl`
- the baseline retrieval profile now includes LLM query decomposition
- the intended baseline runtime is strict:
  - Cohere embeddings are required for indexing/querying
  - Elasticsearch lexical retrieval is required for querying/eval
  - no degraded local fallback retrieval path should silently run
- the active index snapshot namespace is `baseline_embed_english_v3_0_dafb9a0708`
- the repo now has two eval lanes:
- the repo now has three eval lanes:
  - primary:
    - `bgrag eval`
    - deterministic retrieval metrics + structured Cohere judge
  - secondary:
    - `bgrag eval-ragas`
    - Ragas claim-oriented metrics using a repo-native patched Cohere wrapper
  - secondary:
    - `bgrag eval-pairwise`
    - blinded pairwise A/B judging over two existing run artifacts using
      OpenAI `gpt-5.4`
- the best current clean 19-case parity run is:
  - `datasets/runs/query_decomposition_20260323_041110.json`
  - required-claim recall mean: `0.868421052631579`
  - answer failures: `0`
  - forbidden-claim violations: `0`
- the leading canonical dev run is:
  - `datasets/runs/query_decomposition_20260323_035140.json`
  - required-claim recall mean: `0.8888888888888888`
  - packed claim-evidence recall mean annotated: `1.0`
- the leading canonical holdout runs are:
  - `datasets/runs/query_decomposition_20260323_040327.json`
  - `datasets/runs/query_decomposition_20260323_040734.json`
  - required-claim recall mean: `0.825`
  - packed claim-evidence recall mean annotated: `0.975`
  - answer failures: `0`
  - forbidden-claim violations: `0`
- the current answer-side experimental runs are:
  - `datasets/runs/query_guided_answering_20260323_042935.json`
  - `datasets/runs/query_guided_answering_20260323_043001.json`
  - `datasets/runs/planned_answering_20260323_045225.json`
  - `datasets/runs/planned_answering_20260323_045329.json`
  - `datasets/runs/mode_aware_planned_answering_20260323_065108.json`
  - `datasets/runs/mode_aware_planned_answering_20260323_065201.json`
  - `datasets/runs/mode_aware_planned_answering_20260323_072358.json`
  - `datasets/runs/mode_aware_planned_answering_20260323_074615.json`
  - `datasets/runs/mode_aware_planned_answering_20260323_080021.json`
  - `datasets/runs/mode_aware_planned_answering_20260323_081625.json`
  These are informative but not promoted. The query-guided and planned answer
  strategies both improved some holdout workflow cases, but neither cleanly
  beat the baseline on both canonical dev and holdout.
  The new mode-aware branch is the strongest rebuilt-39 answer experiment so
  far, but it is still not promoted because it improves the rebuilt `39`-case
  surface while regressing the original `19`-case full parity run and adding
  substantial latency.
- the main experimental retrieval branches currently under evaluation are:
  - `diverse_packing`
- `query_decomposition` has now been promoted into `baseline`
- the old `datasets/eval/parity/dev_feat.jsonl` file is not a trustworthy
  structured-judge tuning surface because only 3 of its 9 cases contain
  `required_claims`
- a final blind acceptance set does not exist yet and still needs to be authored
- the eval-bank rebuild now has explicit planning artifacts:
  - `docs/eval_authoring.md`
  - `docs/experiment_log.md`
  - `datasets/eval/manifests/eval_case_template.json`
  - `datasets/eval/manifests/parity39_blueprint.json`
  - `datasets/eval/manifests/final_acceptance_blueprint.json`
- the rebuilt `39`-case suite is now complete and validated:
  - `datasets/eval/parity/parity39_working.jsonl`
  - `datasets/eval/dev/parity39_dev_draft.jsonl`
  - `datasets/eval/holdout/parity39_holdout_draft.jsonl`
  - current counts: 39 working / 19 dev / 20 holdout
- the current full rebuilt-39 control run is:
  - `datasets/runs/baseline_20260323_061530.json`
  - required-claim recall mean: `0.7414529914529915`
  - answer failures: `0`
  - forbidden-claim violations: `1`
- the current leading rebuilt-39 experimental run is:
  - `datasets/runs/mode_aware_planned_answering_20260323_072358.json`
  - required-claim recall mean: `0.8047008547008547`
  - answer failures: `0`
  - forbidden-claim violations: `0`
- repeated rebuilt-39 split validation now shows:
  - dev mean:
    - baseline `0.7215`
    - mode-aware `0.8092`
  - holdout mean:
    - baseline `0.7204`
    - mode-aware `0.7304`
  - holdout forbidden-claim violations:
    - baseline `1` in both repeats
    - mode-aware `0` in both repeats
- the current full original-19 experimental run for the mode-aware branch is:
  - `datasets/runs/mode_aware_planned_answering_20260323_081625.json`
  - required-claim recall mean: `0.8157894736842105`
  - answer failures: `0`
  - forbidden-claim violations: `0`
  This is below the promoted `19`-case control, so the branch remains
  experimental.
- the current Ragas smoke artifact is:
  - `datasets/runs/baseline_ragas_20260324_003708.json`
  - suite: `datasets/eval/parity/smoke_one.jsonl`
  - overall metrics:
    - `context_recall_mean = 1.0`
    - `faithfulness_mean = 1.0`
    - `correctness_precision_mean = 0.57`
    - `coverage_recall_mean = 0.67`
  - timing:
    - `answer_phase_seconds = 27.39`
    - `ragas_phase_seconds = 89.94`
  Interpretation:
  - the secondary lane is live and useful
  - it is materially slower than the current judge path
  - it should be used for spot checks, promotion cross-checks, and methodology
    hardening rather than every inner-loop experiment
- the current pairwise validation artifact is:
  - `datasets/runs/pairwise_query_decomposition_20260323_041110_vs_mode_aware_planned_answering_20260323_081625_20260324_011849.json`
  - exact cached replay:
    - `datasets/runs/pairwise_query_decomposition_20260323_041110_vs_mode_aware_planned_answering_20260323_081625_20260324_011934.json`
  - result on the original full `19`-case control suite:
    - control wins `11`
    - candidate wins `6`
    - ties `2`
  - cached replay:
    - `cache_hit_count = 19`
  Interpretation:
  - pairwise A/B judging is now live
  - it agrees with the current choice to keep baseline as the formal control
  - the local cache is working, so repeat comparisons should be much cheaper
- the first rebuilt-39 pairwise validation artifact is:
  - `datasets/runs/pairwise_baseline_20260323_061530_vs_mode_aware_planned_answering_20260323_072358_20260324_012451.json`
  - result on the rebuilt full `39`-case suite:
    - control wins `22`
    - candidate wins `14`
    - ties `3`
  Interpretation:
  - pairwise judging currently still prefers baseline overall on the rebuilt
    `39`, even though the scalar required-claim recall metric favored the
    mode-aware branch there
  - this makes pairwise A/B evaluation a meaningful methodological check, not
    just a redundant second opinion
- the newer `selective_mode_aware_planned_answering` branch remains the current
  leading answer-side candidate, but it is still not promoted by default.
  It has strong single-run numbers on both the original `19`-case suite and
  the rebuilt `39`-case suite, and overnight repeat-validation in the isolated
  autonomy copy also supported it on `parity19_holdout`, but the autonomy pass
  did not leave a clean enough final milestone to treat as a completed
  promotion decision on its own.
- clean control-side repeat validation of the selective branch now exists:
  - `datasets/runs/profile_compare_20260323_194102.md`
  - `datasets/runs/profile_compare_20260323_201115.md`
  - on canonical `parity19_dev`:
    - baseline mean `0.9028`
    - selective mean `0.8750`
  - on canonical `parity19_holdout`:
    - baseline mean `0.8000`
    - selective mean `0.8333`
  - interpretation:
    - selective remains promising
    - but the signal is still mixed on the formal control surface
    - so it is not yet promoted as the new default
- two later structured-contract answer branches have now been tested and both
  remain experimental only:
  - `structured_contract_deterministic_answering`
    - strong on the omission-heavy weak slice
    - not broad-promotable because it regresses canonical `parity19_dev`
  - `selective_workflow_contract_answering`
    - preserves most weak-slice gains
    - but its current planner-only selector is not selective enough and still
      loses the canonical dev pairwise gate
  Interpretation:
  - structured slot extraction plus deterministic rendering is a real method
    for omission-heavy workflow or missing-detail cases
  - but the repo still does not have a trustworthy broad selector for when to
    apply it
- a verifier-gated post-draft specialization branch now exists as another
  answer-side experiment:
  - profile:
    - `profiles/verifier_gated_structured_contract_answering.yaml`
  - weak-slice artifact:
    - `datasets/runs/verifier_gated_structured_contract_answering_20260324_165622.json`
    - baseline `0.3571`
    - verifier-gated `0.3810`
    - pairwise baseline wins `3`, candidate `1`, ties `3`
  - canonical `parity19_dev` artifact:
    - `datasets/runs/verifier_gated_structured_contract_answering_20260324_170544.json`
    - baseline `0.9444`
    - verifier-gated `0.8889`
    - pairwise `2-2-5`
  Interpretation:
  - the post-draft verifier family is more methodologically aligned than
    pre-answer routing
  - the first implementation is safe but underfiring
  - it is not promoted
- two stronger verifier-gated descendants now exist:
  - `contract_aware_verifier_gated_structured_contract_answering`
    - holds canonical dev scalar quality flat relative to baseline in its best
      run
    - still underfires badly, so it is not promoted
  - `contract_slot_coverage_verifier_gated_structured_contract_answering`
    - strongest weak-slice verifier result so far:
      - `datasets/runs/profile_compare_20260324_180252.md`
      - baseline `0.2619`
      - slot-coverage verifier `0.5000`
      - forbidden violations improved from `1` to `0`
      - pairwise:
        - `datasets/runs/pairwise_baseline_20260324_175348_vs_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_180252_20260324_180954.json`
        - baseline wins `0`
        - candidate wins `6`
        - ties `1`
    - but canonical dev is still too broad:
      - `datasets/runs/profile_compare_20260324_180702.md`
      - baseline `0.9167`
      - slot-coverage verifier `0.8333`
      - route counts:
        - `rewrite_structured_contract = 8`
        - `baseline_keep = 1`
      - pairwise:
        - `datasets/runs/pairwise_baseline_20260324_175552_vs_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_180702_20260324_181006.json`
        - baseline wins `5`
        - candidate wins `3`
        - ties `1`
    - interpretation:
      - omission detection is now materially better
      - selective activation is still the blocker
      - branch remains experimental only
- the current leading verifier-gated branch is now:
  - `narrow_contract_slot_coverage_verifier_gated_structured_contract_answering`
  - canonical weak slice:
    - `datasets/runs/profile_compare_20260324_183103.md`
    - baseline `0.2857`
    - narrow verifier `0.5357`
    - pairwise:
      - `datasets/runs/pairwise_baseline_20260324_182254_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_183103_20260324_183609.json`
      - baseline wins `0`
      - candidate wins `3`
      - ties `4`
  - canonical `parity19_dev`:
    - `datasets/runs/profile_compare_20260324_183449.md`
    - baseline `0.8889`
    - narrow verifier `0.8611`
    - route counts:
      - `baseline_keep = 9`
  - canonical `parity19_holdout`:
    - `datasets/runs/profile_compare_20260324_185113.md`
    - baseline `0.8000`
    - narrow verifier `0.8000`
    - route counts:
      - `baseline_keep = 10`
  - rebuilt `39` dev:
    - `datasets/runs/profile_compare_20260324_194042.md`
    - baseline `0.7719`
    - narrow verifier `0.7939`
    - route counts:
      - `baseline_keep = 17`
      - `rewrite_structured_contract = 2`
    - pairwise:
      - `datasets/runs/pairwise_baseline_20260324_192403_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_194042_20260324_202555.json`
      - baseline wins `6`
      - candidate wins `4`
      - ties `9`
  - rebuilt `39` holdout:
    - `datasets/runs/profile_compare_20260324_201649.md`
    - baseline `0.6858`
    - narrow verifier `0.7333`
    - forbidden violations:
      - baseline `2`
      - narrow verifier `1`
    - route counts:
      - `baseline_keep = 19`
      - `rewrite_structured_contract = 1`
    - pairwise:
      - `datasets/runs/pairwise_baseline_20260324_195256_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_201649_20260324_202401.json`
      - baseline wins `8`
      - candidate wins `6`
      - ties `6`
  Interpretation:
  - this is the healthiest verifier-gated branch so far
  - it is largely neutral on the formal `19`-case control surface
  - it helps the rebuilt `39` scalar surfaces
  - pairwise on rebuilt `39` is still mixed, so it remains experimental
- two additional generic workflow-route refinements were tried after that
  control-side repeat validation and both were rejected:
  - a tighter workflow prompt
  - a planner-driven structured-workflow flag
  both regressed a targeted mixed slice and were reverted
- follow-up retrieval-side probes on the low-coverage slice were also rejected:
  - `diverse_packing` did not improve answer quality and reduced URL coverage
  - a baseline-aligned `unified_source_hybrid` run also failed to improve the
    slice
- a later hard-cluster retrieval probe isolated a narrower remaining failure
  mode on:
  - `HR_011`
  - `HR_021`
  - `HR_022`
  - `HR_028`
  On that slice, the plain baseline reached:
  - `datasets/runs/baseline_20260323_210002.json`
  - mean required-claim recall `0.2708`
- two context-expansion retrieval experiments were then validated on that same
  hard cluster:
  - `hierarchical_context_expansion`
    - `datasets/runs/hierarchical_context_expansion_20260323_210432.json`
    - mean required-claim recall `0.3542`
  - `structural_context_expansion`
    - `datasets/runs/structural_context_expansion_20260323_211338.json`
    - mean required-claim recall `0.3542`
    - lower latency than the hierarchical variant
  Interpretation:
  - both are real but limited improvements
  - `structural_context_expansion` is the cleaner mechanism and stays available
    as an experiment
  - neither is strong enough to replace the control path
  - the remaining bottleneck now looks more like page-first or document-first
    selection than another chunk-only context add-on
- a first page-first retrieval attempt was then tested:
  - `datasets/runs/page_seed_retrieval_20260323_212235.json`
  - mean required-claim recall `0.3542`
  - lower latency than the other hard-cluster retrieval experiments
  Interpretation:
  - the direction is still plausible
  - but this first intro-chunk page-seeding implementation did not outperform
    `structural_context_expansion`
  Decision:
  - reject that implementation
  - the next page-first attempt should use a stronger page-ranking mechanism
    than intro-chunk aggregation alone
- stronger document-reranked page seeding was then explored in three variants:
  - global:
    - `datasets/runs/document_rerank_seed_retrieval_20260323_212856.json`
    - hard-cluster mean `0.7708`
  - localized graph scope:
    - `datasets/runs/localized_document_rerank_seed_retrieval_20260323_213426.json`
    - hard-cluster mean `0.7708`
  - lineage-only scope:
    - `datasets/runs/lineage_document_rerank_seed_retrieval_20260323_215406.json`
    - hard-cluster mean `0.7292`
  Interpretation:
  - page-rerank seeding is a real targeted capability for page-coverage
    workflow cases
  - but broad validation on canonical `parity19` still failed:
    - localized graph scope:
      - dev `0.8333`
      - holdout `0.6167`
    - lineage-only scope:
      - dev `0.8611`
      - holdout `0.6167`
  Decision:
  - do not promote any page-rerank variant as the default retriever
  - keep them as targeted experimental profiles only
  - the next sound step is a principled selective activation gate or a return
    to answer-side improvement with the baseline retriever preserved
- an LLM-based selective retrieval gate was then built to choose between:
  - `baseline`
  - `page_family_expansion`
  based on the question plus the baseline evidence summary
- the stronger selective prompt produced:
  - `datasets/runs/selective_localized_document_rerank_seed_retrieval_20260323_222454.json`
    - hard-cluster mean `0.6875`
  - `datasets/runs/selective_localized_document_rerank_seed_retrieval_20260323_223026.json`
    - canonical `parity19_dev`: `0.8611`
  - `datasets/runs/selective_localized_document_rerank_seed_retrieval_20260323_223602.json`
    - canonical `parity19_holdout`: `0.7417`
  Interpretation:
  - selective activation is materially better than broad page-rerank activation
  - but it still does not beat the promoted baseline on the formal control set
  Decision:
  - keep `selective_localized_document_rerank_seed_retrieval` as an
    experimental profile only
  - do not promote selective page-rerank activation as the default retriever
- the overnight autonomy audit produced a few accepted low-risk control-repo
  improvements:
  - richer human-readable repeat summaries in `scripts/compare_profiles.py`
  - repo-relative unit test paths
  - a self-contained parity integration test fixture
  - a small wording cleanup in `src/bgrag/config.py`
- a structured-contract answer branch was then tested as a replacement for the
  free-form selective planner:
  - weak answer-layer slice:
    - baseline `0.3333`
    - current selective branch `0.5595`
    - first structured-contract run `0.6310`
  - canonical `parity19_dev`:
    - `0.8611`
  - canonical `parity19_holdout`:
    - `0.7917`
  - a second schema iteration added workflow slots for:
    - `prerequisite_or_scope`
    - `required_document_or_input`
    - `follow_on_requirement`
    but the weak-slice score stayed flat at `0.6310`
  Interpretation:
  - the structured contract is a real targeted lift on omission-heavy weak
    cases
  - it still does not clear the canonical `parity19` promotion gate
  Decision:
  - keep `structured_contract_mode_aware_answering` experimental only
  - do not promote it
  - the stronger current experimental answer branch is still
    `selective_mode_aware_planned_answering`
- pairwise-vs-scalar disagreement analysis is now in place:
  - original `19` analysis:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_query_decomposition_20260323_041110_vs_mode_aware_planned_answering_20260323_081625_20260324_011849_20260324_013105_610347.md`
  - rebuilt `39` analysis:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260323_061530_vs_mode_aware_planned_answering_20260323_072358_20260324_012451_20260324_013105_619678.md`
  Interpretation:
  - the main disagreement pattern is answer verbosity / embellishment, not
    retrieval coverage
  - candidates often improve required-claim recall by being longer
  - pairwise judging then prefers the shorter baseline because it is more
    faithful and direct
- a narrow answer-repair branch has now been validated on the formal original
  `19`-case control:
  - `datasets/runs/profile_compare_20260324_021300.md`
    - baseline: `0.8553`
    - repair: `0.8684`
  - pairwise cross-check:
    - `datasets/runs/pairwise_baseline_20260324_020025_vs_selective_mode_aware_answer_repair_20260324_021259_20260324_021545.json`
    - control wins `13`, candidate wins `5`, ties `1`
  Interpretation:
  - the repair branch is the strongest scalar-only answer-side candidate so far
  - it is still not promotable because pairwise judging rejects it on the
    formal control surface
- a refinement branch that tried to remove unnecessary supported detail was
  tested and rejected on the answer-precision slice:
  - `datasets/runs/profile_compare_20260324_022912.md`
  - repair `0.7000`
  - refinement `0.5500`
  - the control repo code for that rejected branch has been reverted
- the strongest new answer-side branch after that is now:
  - `selective_mode_aware_compact_answering`
  - best original `19` full run:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_032955.json`
    - required-claim recall mean: `0.9078947368421053`
    - answer failures: `0`
    - forbidden-claim violations: `0`
  - best rebuilt `39` full run:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_034912.json`
    - required-claim recall mean: `0.8217948717948718`
    - answer failures: `0`
    - forbidden-claim violations: `0`
  - pairwise cross-checks:
    - original `19`:
      - `datasets/runs/pairwise_baseline_20260324_020025_vs_selective_mode_aware_compact_answering_20260324_032955_20260324_033148.json`
      - control wins `10`, candidate wins `9`, ties `0`
    - rebuilt `39`:
      - `datasets/runs/pairwise_baseline_20260323_061530_vs_selective_mode_aware_compact_answering_20260324_034912_20260324_035222.json`
      - control wins `18`, candidate wins `19`, ties `2`
  Interpretation:
  - this branch materially improves the earlier verbosity / truncation problem
  - but it is still not promoted because later repeats regressed:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_040035.json`
      - original `19`: `0.8552631578947368`
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_040651.json`
      - canonical dev: `0.8333333333333334`
  - the instability appears to come from the compact planner dropping
    indispensable points, not from retrieval failure
- a canonical targeted answer-side precision gate now exists:
  - slice:
    - `datasets/eval/generated/answer_pairwise_precision_slice.jsonl`
  - manifest:
    - `datasets/eval/manifests/answer_pairwise_precision_slice_manifest.json`
  - builder:
    - `scripts/build_answer_pairwise_precision_slice.py`
  - current validated size:
    - `13` cases
  - use:
    - targeted promotion check for answer-structure / planning changes on the
      exact cases where scalar recall and pairwise judgment tend to disagree
- the first control-side validation on that new precision gate now exists:
  - scalar runs:
    - `datasets/runs/baseline_20260324_041922.json`
      - `0.8500`
    - `datasets/runs/baseline_20260324_042616.json`
      - `0.8500`
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_044115.json`
      - `0.8500`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_042616_vs_selective_mode_aware_compact_answering_20260324_044115_20260324_044745.json`
    - baseline wins `6`
    - compact wins `7`
    - ties `0`
  - pairwise-vs-scalar analysis:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_042616_vs_selective_mode_aware_compact_answering_20260324_044115_20260324_044745_20260324_052605_625658.json`
    - candidate truncation-risk count `0`
    - control truncation-risk count `3`
  - secondary Ragas cross-check:
    - `datasets/runs/baseline_ragas_20260324_051602.json`
    - `datasets/runs/selective_mode_aware_compact_answering_ragas_20260324_052552.json`
    - baseline vs compact:
      - `context_recall_mean`: tie at `0.9615`
      - `faithfulness_mean`: compact better (`0.9722` vs `0.9026`)
      - `correctness_precision_mean`: baseline better (`0.6331` vs `0.5408`)
      - `coverage_recall_mean`: baseline better (`0.7138` vs `0.6038`)
  Interpretation:
  - compact now looks qualitatively healthier on the disagreement-heavy slice
  - but it is still not a clean promotion because the concise branch is
    leaving too many supported points on the table
- a narrow compact-branch tightening pass has now improved that answer-side
  lane further:
  - canonical precision slice:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_054426.json`
    - scalar `0.8462`, `0` failures, `0` forbidden violations
    - pairwise vs baseline:
      - `datasets/runs/pairwise_baseline_20260324_042616_vs_selective_mode_aware_compact_answering_20260324_054426_20260324_054554.json`
      - compact wins `8`
      - baseline wins `5`
  - canonical `parity19_dev`:
    - `datasets/runs/profile_compare_20260324_055800.md`
    - baseline `0.9167`
    - compact `0.8889`
    - pairwise tie `4-4-1`
  - canonical `parity19_holdout`:
    - `datasets/runs/profile_compare_20260324_055940.md`
    - baseline `0.7417`
    - compact `0.8750`
    - pairwise tie `5-5`
  Interpretation:
  - the compact branch is materially stronger than before
  - it is now:
    - winning the precision gate more clearly
    - substantially stronger on canonical holdout scalar quality
    - no longer losing canonical pairwise checks outright
  - but it is still not the clean default:
    - canonical `parity19_dev` scalar quality still favors baseline
    - the branch still needs one more generic precision pass before default
      promotion
- two follow-up branches have now been rejected and removed:
  - a stricter compact prompt variant that reused the contextual
    missing-detail route
    - canonical `parity19_dev` dropped to `0.8333`
    - it still did not fix the remaining `HR_001` omission
    - it regressed `HR_017`
  - a compact-plus-repair branch
    - canonical `parity19_dev` also dropped to `0.8333`
    - the repair planner tried to add unrelated nearby supported detail on
      `HR_001`, which makes the current repair lane methodologically unsound
  Interpretation:
  - the control repo is back at the previous known-good compact state
  - the next move should not be more prompt rescue or naive repair reuse
- the Ragas lane itself has been hardened since the earlier disagreement-slice
  failures:
  - it now uses explicit `RunConfig`
  - defaults:
    - `ragas_timeout_seconds = 480`
    - `ragas_max_workers = 4`
  - metric-level timeouts now degrade to missing values instead of aborting the
    whole run

## What still needs honest rebuild work

- final blind acceptance set
- source-site research synthesis
- stronger answer-side precision improvements on the remaining low cases
- broader end-to-end parity and holdout comparisons against `feat`
- author the sealed final blind acceptance set after the 39-case suite is stable
- stronger tests around retrieval, indexing, and answer packaging

## Immediate next step

Keep the promoted query-decomposition baseline as the control. Use it while:

- treating `selective_mode_aware_compact_answering` as the leading but still
  unpromoted answer-side branch
- using `datasets/eval/generated/answer_pairwise_precision_slice.jsonl` as the
  next targeted promotion gate before any new answer-side branch is trusted
- requiring answer-side branches to clear:
  - the pairwise-precision slice
  - canonical `parity19` dev/holdout
  - rebuilt `parity39` dev/holdout
  - pairwise A/B
- using Ragas `faithfulness` and `correctness_precision` as spot checks on the
  disagreement-heavy slice rather than only on smoke cases
- avoiding more prompt-only rescue tweaks unless they improve stability, not
  just a single full-suite headline run
- avoiding repair-based answer promotion work until the repair planner can
  distinguish required answer content from adjacent supported context much more
  reliably
- focusing the next answer-side work on the compact branch's remaining
  precision/dev losses:
  - `HR_005`
  - `HR_009`
  - `HR_011`
  - `HR_013`
  - `HR_015`
  - `HR_017`
- if continuing the verifier-gated lane, limiting it to one bounded follow-up:
  - use the narrow slot-coverage verifier as the current base
  - analyze rebuilt-39 disagreement cases where scalar improves but pairwise
    still favors baseline
  - do not return to opaque pre-answer routing
- building the sealed final acceptance set only after the rebuilt 39-case suite
  and the answer-side promotion gate are both stable

## Parallel Worker State (2026-03-24)

- active disposable worker lanes:
  - disagreement:
    - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean-worker_disagreement_20260324_163228`
  - research:
    - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean-worker_research_20260324_163228`
  - infra:
    - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean-worker_infra_20260324_163228`
- shared Elasticsearch policy:
  - the baseline namespace `baseline_embed_english_v3_0_dafb9a0708` is treated as
    read-only
  - any worker that needs a new index must create and document a unique lane-specific
    namespace

## Rebuilt-39 Disagreement Read

- fresh pairwise-vs-scalar analyses were generated for the current leading verifier
  branch:
  - dev:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_192403_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_194042_20260324_202555_20260324_203929_290278.md`
  - holdout:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_195256_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_201649_20260324_202401_20260324_203928_965117.md`
- current interpretation:
  - the branch's real wins are still the rewritten omission-heavy cases
    - dev examples:
      - `HR_030`
      - `HR_038`
    - holdout examples:
      - `HR_006`
      - `HR_016`
      - `HR_035`
  - most pairwise control wins are not "control beats candidate despite higher
    candidate recall"
  - instead, they are mostly equal-recall cases where the judge prefers baseline
    style/faithfulness or dislikes small extra details in candidate wording
  - this means the leading branch is still promising, but the next move should be
    careful disagreement-case review rather than another broad retrieval rewrite

## Intervention-Only Measurement Read

- new tooling:
  - `src/bgrag/eval/run_composition.py`
  - `scripts/build_intervention_composite_run.py`
- purpose:
  - isolate real branch effect by reusing control-run cases for all
    `baseline_keep` paths and candidate-run cases only for real interventions
- rebuilt-39 dev composite:
  - `datasets/runs/baseline_20260324_192403_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_194042_intervention_only_20260324_204604_626823.md`
  - selected cases:
    - `HR_030`
    - `HR_038`
  - recall:
    - control `0.7719`
    - full candidate `0.7939`
    - intervention-only composite `0.8333`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_192403_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_intervention_only_20260324_204604_20260324_204736.json`
    - candidate wins `2`
    - control wins `0`
    - ties `17`
- rebuilt-39 holdout composite:
  - `datasets/runs/baseline_20260324_195256_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_201649_intervention_only_20260324_204604_464457.md`
  - selected case:
    - `HR_035`
  - recall:
    - control `0.6858`
    - full candidate `0.7333`
    - intervention-only composite `0.6958`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_195256_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_intervention_only_20260324_204604_20260324_204741.json`
    - candidate wins `1`
    - control wins `0`
    - ties `19`
- interpretation:
  - the conditional verifier branch itself is behaving well where it actually
    intervenes
  - the mixed full-run pairwise surface is mostly noise from regenerated
    `baseline_keep` cases, not evidence that the rewrite logic is bad
  - the next bottleneck is likely question-scoped contract extraction or slot
    pruning, not broader activation tuning

## Promotion Branch Status

- active promotion branch:
  - `feat/eval-infra-exactness-promotion`
- intent:
  - land only the proven evaluation hardening and the narrow exactness-only
    missing-detail path
- current blocker status:
  - the accidental dependency on broad slot selection inside the exactness-only
    path has been removed locally
  - targeted validation is passing again
  - a full preserved-baseline `parity19` comparison is running to re-check the
    branch end to end

## Claude Collaboration Status

- shared Claude instructions now live in:
  - `CLAUDE.md`
- local resumable session workflow now lives in:
  - `scripts/consult_claude.ps1`
  - `.claude/session_local/` (gitignored)
- intended usage:
  - Claude Opus 4.6, max effort
  - repeated peer consultations with preserved context
  - repo-local eval evidence still decides promotions

## Eval Integrity Status

- the promotion branch now includes two additional harness hardening changes:
  - judge claim-text alignment is enforced, not just list length
  - conditional compare summaries now report drift on non-selected cases
- current local validation:
  - full suite: `141 passed`
- practical implication:
  - intervention-only conditional results are now safer to interpret because
    summaries no longer hide untouched-case drift
  - long-running conditional comparisons are now observable through timestamped
    progress logs instead of appearing silent until completion

## Exactness Merge Read

- current branch judgment:
  - the eval hardening is ready to merge
  - the narrow missing-detail exactness path is ready to merge as an available
    sub-path
  - the standalone exactness profile is not a trustworthy full-suite promotion
    surface because regenerated non-selected cases still drift from control
- promotion rule:
  - evaluate the exactness path through:
    - intervention-only composites
    - split-safe exactness slices
  - not through raw full-suite standalone profile comparisons

## Exactness Surface Expansion Status

- active branch:
  - `feat/exactness-surface-expansion`
- status:
  - `HR_016` is now tightened on the canonical eval surfaces with an explicit
    adjacent-form forbidden claim
  - a new audited exactness-family diagnostic surface now exists:
    - `datasets/eval/dev/exactness_family_dev.jsonl`
    - `datasets/eval/holdout/exactness_family_holdout.jsonl`
  - the new authored cases are:
    - `EX_001`
      - Controlled Goods Directorate contact-detail abstention case
    - `EX_002`
      - PSD / GCDocs internal-file abstention case
- current local validation:
  - `python scripts/validate_eval_cases.py` on the new exactness-family files:
    - `6 total case(s)`
  - targeted exactness tests:
    - `5 passed`
  - full suite:
    - `144 passed`
- interpretation:
  - the repo now has a stronger exactness-family regression surface than the
    original 4-case parity39-only slice
  - this remains a diagnostic family for exactness and abstention behavior, not
    a full answer-quality promotion surface by itself

## Immediate Next Step

- use the new `exactness_family_dev` and `exactness_family_holdout` surfaces as
  the next regression gate for:
  - the narrow exactness-only sub-path
  - any broader answer-side branch that claims better abstention or exactness
    behavior
- if the family proves stable over a few runs, either:
  - expand it with one more newly audited holdout-quality case, or
  - move back to the broader answer-architecture work with this stronger
    exactness gate in place

## Exactness-family Rerun Read

- branch:
  - `feat/exactness-surface-expansion`
- current trusted read:
  - the exactness-family surface is useful and should be merged
  - the narrow exactness sub-path still helps on the cases it actually rewrites
  - the gate still underfires on the new authored exactness cases
- latest intervention-only summaries:
  - dev:
    - `datasets/runs/conditional_compare_summary_20260325_073131_514889_ba4d.json`
    - selected:
      - `HR_038`
    - recall:
      - `0.7222 -> 0.8889`
  - holdout:
    - `datasets/runs/conditional_compare_summary_20260325_073144_962664_a209.json`
    - selected:
      - `HR_016`
      - `HR_037`
    - recall:
      - `0.7778 -> 0.8889`
    - forbidden violations:
      - `1 -> 0`
    - abstain accuracy:
      - `0.6667 -> 1.0`
- merge recommendation:
  - merge the branch for the eval-surface and harness improvements
  - treat `EX_001` / `EX_002` underfiring as a bounded follow-up branch
    targeted at exactness-gate sensitivity
- open blocker:
  - OpenAI pairwise is currently unavailable again because the live key is
    returning `401 account_deactivated` on `responses`
