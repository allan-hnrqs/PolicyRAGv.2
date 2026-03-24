# Experiment Log

This file records architecture and tuning experiments in a compact
human-readable format so the project history does not live only in chat or raw
run artifacts.

Use one entry per experiment family or promotion decision. Each entry should
capture:

- date
- experiment name
- hypothesis
- code or profile changes
- evaluation surfaces used
- key run artifacts
- result
- promotion decision
- methodology notes or caveats

## 2026-03-23

### Measurement Hardening: Secondary Ragas Lane

- Hypothesis:
  - the repo needs a second, claim-oriented evaluation surface so we do not
    over-trust a single judge implementation
  - Cohere's official RAG evaluation framing suggests grounding, correctness,
    and coverage should be measured separately
- Research inputs:
  - Cohere RAG evaluation deep dive:
    - https://docs.cohere.com/page/rag-evaluation-deep-dive
  - Ragas official docs:
    - `llm_factory`, caching, and metric references
- Implementation changes:
  - added `bgrag eval-ragas`
  - added `src/bgrag/eval/ragas_runner.py`
  - added dedicated result types in `src/bgrag/types.py`
  - added unit coverage in `tests/unit/test_ragas_runner.py`
- Important methodology finding:
  - the installed `ragas.llm_factory(..., provider="cohere")` path is broken
    against `cohere.ClientV2`
  - a repo-native workaround does work:
    - `instructor.from_cohere(...)`
    - wrapped by `ragas.llms.base.InstructorLLM`
    - with `top_p` removed from the default model args
- Metrics selected for the secondary lane:
  - `context_recall`
  - `faithfulness`
  - `correctness_precision`
  - `coverage_recall`
- Validation:
  - unit tests:
    - `tests/unit/test_ragas_runner.py`
    - `tests/unit/test_profiles.py`
  - smoke run:
    - `datasets/runs/baseline_ragas_20260324_003708.json`
- Smoke result:
  - case count: `1`
  - `context_recall_mean = 1.0`
  - `faithfulness_mean = 1.0`
  - `correctness_precision_mean = 0.57`
  - `coverage_recall_mean = 0.67`
  - answer phase: `27.39s`
  - ragas phase: `89.94s`
- Result:
  - accepted as a secondary eval lane
- Promotion decision:
  - keep
  - but do not replace the current primary eval harness
- Caveats:
  - slower than the current judge path
  - still same-family evaluation for now
  - if we later add a strong OpenAI or Anthropic evaluator key, the secondary
    lane should be revisited to reduce same-family bias

### Measurement Hardening: OpenAI Pairwise A/B Lane

- Hypothesis:
  - some architecture decisions are easier to trust when a strong judge directly
    compares control vs candidate rather than scoring each answer in isolation
- Research inputs:
  - Cohere retrieval-eval Pydantic AI example:
    - https://docs.cohere.com/page/retrieval-eval-pydantic-ai
  - official OpenAI SDK / Responses API
- Implementation changes:
  - added `bgrag eval-pairwise`
  - added `src/bgrag/eval/pairwise.py`
  - added pairwise result types
  - added unit coverage in `tests/unit/test_pairwise.py`
- Design choices:
  - compare two existing run artifacts instead of regenerating answers
  - use official OpenAI SDK structured parsing (`responses.parse`)
  - use `gpt-5.4`
  - do not use `mini`
  - use local `diskcache` plus OpenAI prompt-cache fields
  - blind answer order deterministically per case
- Real validation surface:
  - control:
    - `datasets/runs/query_decomposition_20260323_041110.json`
  - candidate:
    - `datasets/runs/mode_aware_planned_answering_20260323_081625.json`
- First real artifact:
  - `datasets/runs/pairwise_query_decomposition_20260323_041110_vs_mode_aware_planned_answering_20260323_081625_20260324_011849.json`
- Result:
  - case count: `19`
  - control wins: `11`
  - candidate wins: `6`
  - ties: `2`
  - candidate non-tie win rate: `0.3529`
- Cached replay artifact:
  - `datasets/runs/pairwise_query_decomposition_20260323_041110_vs_mode_aware_planned_answering_20260323_081625_20260324_011934.json`
  - `cache_hit_count = 19`
- Additional rebuilt-39 artifact:
  - `datasets/runs/pairwise_baseline_20260323_061530_vs_mode_aware_planned_answering_20260323_072358_20260324_012451.json`
  - result:
    - control wins `22`
    - candidate wins `14`
    - ties `3`
- Promotion decision:
  - keep as a secondary promotion check
  - do not replace the primary eval harness
- Methodology notes:
  - initial implementation needed two fixes before it was trustworthy:
    - ensure repo `.env` overrides stale process-level OpenAI env vars
    - add bounded retries with a higher output cap for cases where structured
      JSON was truncated
  - after those fixes, the lane produced stable real results and the cache worked
  - important outcome:
    - pairwise A/B judgment did not endorse the mode-aware branch on either the
      original `19`-case control surface or the rebuilt `39`-case surface
    - this diverges from the rebuilt-39 scalar required-claim recall result and
      should influence future promotion decisions

### Answer-Layer Control Before Mode-Aware Planning

- Control profile: `baseline`
- Control answer strategy: `inline_evidence_chat`
- Control retrieval strategy: query decomposition already promoted
- Main control artifacts:
  - `datasets/runs/query_decomposition_20260323_041110.json`
  - `datasets/runs/baseline_20260323_061530.json`
- Current interpretation:
  - retrieval recall is strong enough to treat retrieval as the control
  - remaining weakness is mostly answer-side precision and abstention behavior
  - weak cases to watch while testing a new answer strategy:
    - `HR_002`
    - `HR_011`
    - `HR_021`
    - `HR_026`
    - `HR_030`
    - `HR_037`
    - `HR_038`
- Next experiment family:
  - build one generic, mode-aware planned answer strategy
  - keep it evidence-grounded and avoid keyword-routed case hacks
  - require it to beat the control on `parity39_dev_draft` and
    `parity39_holdout_draft` before promotion

### Mode-Aware Planned Answering

- Hypothesis:
  - retrieval was already strong enough that the next gain should come from a
    generic answer layer that chooses a better answer frame from the question
    and evidence
  - expected wins were better multi-branch coverage and cleaner abstention
- Code and profile changes:
  - added `mode_aware_planned_inline_evidence_chat`
  - added profile `mode_aware_planned_answering`
  - added prompt/schema tests for the new planning path
- Evaluation surfaces used:
  - weak answer-layer slice:
    - `datasets/eval/generated/answer_layer_weak_cases.jsonl`
  - rebuilt `39`-case dev:
    - `datasets/eval/dev/parity39_dev_draft.jsonl`
  - rebuilt `39`-case holdout:
    - `datasets/eval/holdout/parity39_holdout_draft.jsonl`
  - full rebuilt `39`-case suite:
    - `datasets/eval/parity/parity39_working.jsonl`
  - original `19`-case parity suite:
    - `datasets/eval/parity/parity19.jsonl`
- Key run artifacts:
  - weak slice good candidate:
    - `datasets/runs/mode_aware_planned_answering_20260323_063152.json`
  - rebuilt `39` dev repeats:
    - `datasets/runs/mode_aware_planned_answering_20260323_065108.json`
    - `datasets/runs/mode_aware_planned_answering_20260323_074615.json`
  - rebuilt `39` holdout repeats:
    - `datasets/runs/mode_aware_planned_answering_20260323_065201.json`
    - `datasets/runs/mode_aware_planned_answering_20260323_080021.json`
  - full rebuilt `39` suite:
    - `datasets/runs/mode_aware_planned_answering_20260323_072358.json`
  - original `19`-case parity:
    - `datasets/runs/mode_aware_planned_answering_20260323_081625.json`
- Results:
  - weak-slice mean moved from `0.2857` under control to `0.6667`
  - rebuilt `39` dev repeat mean:
    - control `0.7215`
    - mode-aware `0.8092`
  - rebuilt `39` holdout repeat mean:
    - control `0.7204`
    - mode-aware `0.7304`
  - holdout forbidden violations:
    - control `1` in both repeats
    - mode-aware `0` in both repeats
  - full rebuilt `39` single run:
    - control `0.7415`
    - mode-aware `0.8047`
  - original `19`-case parity full run:
    - control `0.8684`
    - mode-aware `0.8158`
- Promotion decision:
  - not promoted to the default baseline
- Why not promoted:
  - it clearly improves the rebuilt `39`-case surface and abstention behavior
  - but it regresses the original `19`-case full parity surface, which is
    still the formal phase-1 control benchmark
  - it is also materially slower than the current baseline
- Methodology notes:
  - judged scores showed visible run-to-run variance, so promotion decisions
    were based on repeated dev/holdout checks rather than a single full-suite run
  - deterministic retrieval metrics stayed unchanged, confirming the gains and
    regressions were answer-side rather than retrieval-side

### Overnight Autonomy Audit

- Context:
  - an isolated autonomy copy was created at
    `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean-autonomy`
  - the autonomous worker did not leave a clean written milestone in the repo
    docs, likely in part because the machine shut down during the run
- What the autonomy run produced:
  - repeated `parity19_dev` comparisons:
    - `datasets/runs/profile_compare_20260323_103343.md`
  - repeated `parity19_holdout` comparisons:
    - `datasets/runs/profile_compare_20260323_105546.md`
  - an additional full baseline parity run:
    - `datasets/runs/baseline_20260323_110211.json`
- What the repeated validation said:
  - on `parity19_dev`, baseline and `selective_mode_aware_planned_answering`
    tied at mean required-claim recall `0.875`
  - on `parity19_holdout`, selective was modestly better:
    - baseline mean `0.8000`
    - selective mean `0.8333`
  - selective also stayed cleaner on forbidden-claim behavior in those holdout
    repeats
- What was *not* accepted from autonomy:
  - no new answer-side promotion decision
  - no undocumented or partially justified architecture change
  - no direct copy-back of the autonomy repo as a whole
- What was accepted and ported back into control:
  - richer human-readable repeat summaries in `scripts/compare_profiles.py`
  - repo-relative test paths in unit tests
  - a self-contained parity integration test fixture
  - a small wording cleanup in `src/bgrag/config.py`
- Why only these pieces were kept:
  - they are low-risk hygiene/tooling improvements
  - they passed targeted tests cleanly
  - they do not depend on trusting the autonomy worker's incomplete promotion logic

### Control-Side Selective Repeat Validation

- Purpose:
  - re-run the promotion question from the control repo after the autonomy
    audit, using the new repeat-summary tooling and without relying on the
    interrupted overnight run
- Evaluation surfaces:
  - `datasets/eval/dev/parity19_dev.jsonl`
  - `datasets/eval/holdout/parity19_holdout.jsonl`
- Key artifacts:
  - dev comparison:
    - `datasets/runs/profile_compare_20260323_194102.md`
    - `datasets/runs/profile_compare_20260323_194102.json`
  - holdout comparison:
    - `datasets/runs/profile_compare_20260323_201115.md`
    - `datasets/runs/profile_compare_20260323_201115.json`
- Results:
  - canonical `parity19_dev` repeats:
    - baseline mean `0.9028`
    - selective mean `0.8750`
  - canonical `parity19_holdout` repeats:
    - baseline mean `0.8000`
    - selective mean `0.8333`
  - both profiles stayed at `0` answer failures
  - both profiles stayed at `0` forbidden-claim violations in these control-side repeats
- Interpretation:
  - the selective branch remains promising, but the control-side split results
    are mixed rather than cleanly dominant
  - selective is helping on holdout but is not beating baseline on dev
  - this is not yet a strong enough signal to replace the default baseline
- Promotion decision:
  - still not promoted

### Failed Workflow-Route Refinements

- Goal:
  - recover the selective branch's canonical `parity19_dev` regressions without
    losing its holdout gains
- Narrow evaluation slice used:
  - `datasets/eval/generated/selective_regression_mix.jsonl`
  - cases:
    - `HR_001`
    - `HR_002`
    - `HR_006`
    - `HR_008`
    - `HR_010`
    - `HR_012`

#### Attempt 1: Tighter Workflow Prompt

- Change:
  - temporarily tightened the workflow answer prompt to emphasize bottom-line
    answers and suppress adjacent policy/background detail
- Key artifact:
  - `datasets/runs/selective_mode_aware_planned_answering_20260323_202146.json`
- Result:
  - regression-mix mean `0.8333`
  - baseline on same slice: `0.8750`
- Interpretation:
  - this did not recover the known regressions and was not worth keeping
- Action:
  - reverted

#### Attempt 2: Planner-Driven Structured-Workflow Flag

- Change:
  - temporarily let the planner decide whether workflow questions should use a
    structured checklist path or the simpler inline path
- Key artifact:
  - `datasets/runs/selective_mode_aware_planned_answering_20260323_202805.json`
- Result:
  - regression-mix mean `0.6667`
  - strong degradations on cases that had been selective wins, especially
    `HR_002`, `HR_006`, and `HR_010`
- Interpretation:
  - the planner did not make this route distinction reliably enough
  - this added complexity without improving the actual quality signal
- Action:
  - reverted

#### Retrieval Follow-Up: Low-Coverage Slice

- Goal:
  - test whether the remaining weakest cases were primarily retrieval/packing
    problems rather than answer-synthesis problems
- Narrow evaluation slice:
  - `datasets/eval/generated/retrieval_gap_mix.jsonl`
  - cases:
    - `HR_011`
    - `HR_020`
    - `HR_023`
    - `HR_028`
- Baseline artifact:
  - `datasets/runs/baseline_20260323_203327.json`
  - mean required-claim recall `0.5208`
- Attempt A: `diverse_packing`
  - `datasets/runs/diverse_packing_20260323_203258.json`
  - result:
    - same mean answer score as baseline: `0.5208`
    - worse packed expected URL recall
  - action:
    - rejected as a promising next step for this slice
- Attempt B: `unified_source_hybrid`
  - first run:
    - `datasets/runs/unified_source_hybrid_20260323_203524.json`
  - methodology correction:
    - the profile was stale and not baseline-aligned, so it was updated to
      mirror the current baseline except for source topology
  - corrected rerun:
    - `datasets/runs/unified_source_hybrid_20260323_203846.json`
  - result:
    - same mean answer score as baseline: `0.5208`
    - no retrieval-coverage improvement on the slice
  - action:
    - rejected as a promising next step for this slice

#### Retrieval Follow-Up: Hard-Cluster Structural Expansion

- Goal:
  - improve the remaining structurally weak cases by adding controlled page and
    document context rather than more answer-layer prompt logic
- Narrow evaluation slice:
  - `datasets/eval/generated/hard_cluster_probe.jsonl`
  - cases:
    - `HR_011`
    - `HR_021`
    - `HR_022`
    - `HR_028`
- Control artifact:
  - `datasets/runs/baseline_20260323_210002.json`
  - mean required-claim recall `0.2708`
- Attempt A: `hierarchical_context_expansion`
  - `datasets/runs/hierarchical_context_expansion_20260323_210432.json`
  - result:
    - mean required-claim recall `0.3542`
    - same aggregate deterministic retrieval coverage as baseline
    - improved `HR_021`, but did not resolve `HR_011`, `HR_022`, or `HR_028`
  - interpretation:
    - restricted re-retrieval can help a little, but it is still too indirect
      and expensive for the specific missing-evidence pattern in this slice
- Attempt B: `structural_context_expansion`
  - `datasets/runs/structural_context_expansion_20260323_211338.json`
  - result:
    - mean required-claim recall `0.3542`
    - answer failures `0`
    - forbidden-claim violations `0`
    - lower latency than the hierarchical attempt
  - interpretation:
    - deterministic structural companion selection is a cleaner shape than the
      earlier hierarchical pass
    - it surfaced some better supporting chunks, especially on `HR_011`
    - but it still plateaued at the same answer score and did not materially
      fix the page-coverage failures in `HR_021`, `HR_022`, or `HR_028`
- Action:
  - keep `structural_context_expansion` as an experimental profile
  - do not promote it
  - next architectural move should be page-first or document-first retrieval,
    because the current failure mode is increasingly about selecting the right
    page context before chunk packing

#### Retrieval Follow-Up: Page-Seed Retrieval

- Goal:
  - move earlier in the retrieval funnel by selecting likely pages from intro
    material first, then re-scoring chunks inside those seeded pages
- Narrow evaluation slice:
  - `datasets/eval/generated/hard_cluster_probe.jsonl`
- Artifact:
  - `datasets/runs/page_seed_retrieval_20260323_212235.json`
- Result:
  - mean required-claim recall `0.3542`
  - answer failures `0`
  - forbidden-claim violations `0`
  - notably faster than both the hierarchical and structural context runs
- Interpretation:
  - the page-first idea is directionally sound, but this first intro-chunk
    implementation did not beat the existing structural experiment
  - it still failed to resolve the core page-coverage misses in `HR_011`,
    `HR_022`, and `HR_028`
  - it therefore does not justify keeping additional control-path complexity in
    its current form
- Action:
  - reject this first implementation
  - keep moving toward earlier page selection, but the next attempt should use
    a stronger page-ranking mechanism than intro-chunk aggregation alone

#### Retrieval Follow-Up: Document-Reranked Page Seeding

- Goal:
  - strengthen page-first retrieval by using Cohere rerank over page summaries
    instead of intro-chunk aggregation alone
- Attempt A: global document rerank seeding
  - hard-cluster artifact:
    - `datasets/runs/document_rerank_seed_retrieval_20260323_212856.json`
  - hard-cluster result:
    - mean required-claim recall `0.7708`
    - strong gains on `HR_021`, `HR_022`, and `HR_028`
    - but it catastrophically displaced the right evidence on `HR_011`
  - interpretation:
    - global page reranking is powerful
    - but unconstrained page jumps are too unsafe for broad use

- Attempt B: localized document rerank seeding
  - hard-cluster artifact:
    - `datasets/runs/localized_document_rerank_seed_retrieval_20260323_213426.json`
  - hard-cluster result:
    - mean required-claim recall `0.7708`
    - restored strong deterministic retrieval metrics
    - improved `HR_011` to `0.75`
  - broad validation:
    - `datasets/runs/localized_document_rerank_seed_retrieval_20260323_214054.json`
      - canonical `parity19_dev`: `0.8333`
    - `datasets/runs/localized_document_rerank_seed_retrieval_20260323_214821.json`
      - canonical `parity19_holdout`: `0.6167`
  - interpretation:
    - local graph scoping fixed the worst global misrouting
    - but broad deployment still regressed badly, especially on reciprocal
      procurement and other support-linked detail questions

- Attempt C: lineage-only document rerank seeding
  - hard-cluster artifact:
    - `datasets/runs/lineage_document_rerank_seed_retrieval_20260323_215406.json`
  - hard-cluster result:
    - mean required-claim recall `0.7292`
    - weaker than localized graph scope, but still much better than baseline
  - broad validation:
    - `datasets/runs/lineage_document_rerank_seed_retrieval_20260323_215922.json`
      - canonical `parity19_dev`: `0.8611`
    - `datasets/runs/lineage_document_rerank_seed_retrieval_20260323_220846.json`
      - canonical `parity19_holdout`: `0.6167`
  - interpretation:
    - lineage-only scoping reduced some page-drift problems
    - but the page-rerank family still fails broad promotion because holdout
      retrieval quality falls too much

- Overall conclusion:
  - page-rerank seeding is a real targeted capability, not noise
  - it solves the specific page-coverage cluster far better than the previous
    chunk-only experiments
  - but every broad variant tested so far is non-promotable on the formal
    control surface
- Action:
  - keep these profiles experimental only
  - do not replace the baseline retriever with any page-rerank variant
  - next move should be either:
    - derive a principled selective activation gate for page-rerank seeding, or
    - shift back to answer-side improvements while preserving baseline retrieval

#### Retrieval Follow-Up: Selective Page-Rerank Activation

- Goal:
  - keep the targeted page-rerank gains while avoiding the broad regressions on
    the canonical `parity19` control
- Implementation:
  - added a retrieval-mode selector that inspects the question plus the
    baseline evidence summary and decides between:
    - `baseline`
    - `page_family_expansion`
  - profile:
    - `profiles/selective_localized_document_rerank_seed_retrieval.yaml`

- Attempt A: initial selective prompt
  - artifact:
    - `datasets/runs/selective_localized_document_rerank_seed_retrieval_20260323_222052.json`
  - result:
    - hard-cluster mean `0.3542`
  - interpretation:
    - the selector was too conservative and mostly stayed on baseline

- Attempt B: stronger workflow-and-exception prompt
  - hard-cluster artifact:
    - `datasets/runs/selective_localized_document_rerank_seed_retrieval_20260323_222454.json`
  - hard-cluster result:
    - mean required-claim recall `0.6875`
    - selected page expansion for:
      - `HR_021`
      - `HR_022`
      - `HR_028`
    - kept baseline for:
      - `HR_011`
  - broad validation:
    - `datasets/runs/selective_localized_document_rerank_seed_retrieval_20260323_223026.json`
      - canonical `parity19_dev`: `0.8611`
    - `datasets/runs/selective_localized_document_rerank_seed_retrieval_20260323_223602.json`
      - canonical `parity19_holdout`: `0.7417`
  - interpretation:
    - this is meaningfully better than the broad retrieval swap
    - but it still underperforms the promoted baseline on the formal control
      surface
    - the remaining bad activations include:
      - `HR_004`
      - `HR_010`
      - `HR_019`

- Overall conclusion:
  - selective page-rerank activation is a real improvement over broad
    activation, not noise
  - but it is still not promotable as a default retrieval path
- Action:
  - keep it as an experimental profile
  - stop short of promotion
  - the clean next step is to return to answer-side work while preserving the
    baseline retriever, unless a more principled retrieval gate emerges

#### Answer Follow-Up: Structured Answer Contract

- Goal:
  - replace the current free-form mode-aware coverage checklist with a
    mode-specific structured answer contract so the final answer can fill
    explicit slots instead of relying on a loose checklist
- Implementation:
  - added experimental profile:
    - `profiles/structured_contract_mode_aware_answering.yaml`
  - added a new structured planner and final-answer route family in:
    - `src/bgrag/answering/strategies.py`

- Attempt A: first structured contract schema
  - weak-slice control:
    - `datasets/runs/baseline_20260323_231030.json`
    - `0.3333`
    - `1` forbidden violation
  - current selective branch on the same slice:
    - `datasets/runs/selective_mode_aware_planned_answering_20260323_231520.json`
    - `0.5595`
    - `0` forbidden violations
  - first structured-contract run:
    - `datasets/runs/structured_contract_mode_aware_answering_20260323_231929.json`
    - `0.6310`
    - `0` forbidden violations
  - interpretation:
    - the structured contract is a real improvement on the targeted
      answer-layer weak slice
    - it clearly helps branch-heavy cases like `HR_002`
    - but its slot schema is too coarse for some broader workflow questions

- Attempt B: formal `parity19` control validation
  - canonical dev:
    - `datasets/runs/structured_contract_mode_aware_answering_20260323_232655.json`
    - `0.8611`
  - canonical holdout:
    - `datasets/runs/structured_contract_mode_aware_answering_20260323_232724.json`
    - `0.7917`
  - comparison against current control signals:
    - baseline mean dev `0.9028`
    - baseline mean holdout `0.8000`
    - selective mean dev `0.8750`
    - selective mean holdout `0.8333`
  - interpretation:
    - the first structured-contract branch does not clear the canonical
      `parity19` gate
    - it is better than the raw baseline on the targeted weak slice, but not
      strong enough on the formal control surface
    - main regressions show the schema is missing:
      - prerequisites or scope checks
      - required forms or inputs
      - follow-on obligations such as notifications or publication steps

- Attempt C: second schema iteration
  - added workflow slots for:
    - `prerequisite_or_scope`
    - `required_document_or_input`
    - `follow_on_requirement`
  - rerun on the weak slice:
    - `datasets/runs/structured_contract_mode_aware_answering_20260323_233451.json`
    - `0.6310`
  - interpretation:
    - no measured gain over the first structured-contract run
    - this lane has not earned broader promotion yet

- Overall conclusion:
  - structured contracts are not noise; they improve the specific omission-heavy
    weak slice
  - but the current form is still not broad-promotable on canonical `parity19`
  - a second schema iteration did not improve the target slice
- Action:
  - keep `structured_contract_mode_aware_answering` experimental only
  - do not promote it
  - the next sound answer-side experiment should preserve the stronger current
    selective branch and add a narrow repair/completeness pass rather than
    replacing the planner with a heavier schema

#### Measurement Follow-Up: Pairwise vs Scalar Disagreement Analysis

- Goal:
  - explain why scalar required-claim recall keeps liking some answer-side
    branches that pairwise A/B judging still rejects
- Implementation:
  - added:
    - `scripts/analyze_pairwise_vs_scalar.py`
  - fixed the script's output naming to include the pairwise-run stem so
    parallel runs do not overwrite each other
  - generated analyses for:
    - `datasets/runs/pairwise_query_decomposition_20260323_041110_vs_mode_aware_planned_answering_20260323_081625_20260324_011849.json`
    - `datasets/runs/pairwise_baseline_20260323_061530_vs_mode_aware_planned_answering_20260323_072358_20260324_012451.json`

- Results:
  - original `19`-case control:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_query_decomposition_20260323_041110_vs_mode_aware_planned_answering_20260323_081625_20260324_011849_20260324_013105_610347.md`
    - candidate mean length ratio vs control: `1.596`
    - candidate mean length ratio on control-win cases: `1.870`
    - candidate truncation-risk count: `12`
  - rebuilt `39`-case surface:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260323_061530_vs_mode_aware_planned_answering_20260323_072358_20260324_012451_20260324_013105_619678.md`
    - candidate mean length ratio vs control: `1.638`
    - candidate mean length ratio on candidate-win cases: `1.931`
    - candidate truncation-risk count: `27`
    - only one case (`HR_021`) showed a control win despite positive scalar
      recall delta

- Interpretation:
  - the main disagreement is not retrieval
  - the answer-side candidates often win scalar recall by getting longer and
    covering more points, but pairwise judges then penalize them for:
    - unnecessary detail
    - reduced faithfulness
    - truncation risk
    - weaker directness
  - most control wins are equal-recall cases where the candidate answer is
    simply too long or too embellished

- Action:
  - treat pairwise A/B as a real promotion gate, not a decorative extra
  - future answer-side experiments should target concise completeness rather
    than raw coverage expansion

#### Answer Follow-Up: Narrow Repair Pass

- Goal:
  - preserve the stronger selective answer branch while repairing only missing
    supported points or unsupported claims after the draft answer is produced
- Implementation:
  - existing experimental profile:
    - `profiles/selective_mode_aware_answer_repair.yaml`

- Targeted weak-slice signal:
  - earlier slice:
    - `datasets/runs/selective_mode_aware_answer_repair_20260323_234645.json`
    - `0.7619`
  - repeat on `datasets/eval/generated/answer_precision_slice.jsonl`:
    - `datasets/runs/selective_mode_aware_answer_repair_20260324_022519.json`
    - `0.7000`
    - `0` forbidden violations

- Canonical control validation:
  - dev:
    - `datasets/runs/profile_compare_20260324_015047.md`
    - baseline `0.8889`
    - selective planned `0.9167`
    - repair `0.8889`
  - holdout:
    - `datasets/runs/profile_compare_20260324_015238.md`
    - baseline `0.8500`
    - selective planned `0.8750`
    - repair `0.9000`
  - full original `19`:
    - `datasets/runs/profile_compare_20260324_021300.md`
    - baseline `0.8553`
    - repair `0.8684`

- Pairwise cross-check:
  - `datasets/runs/pairwise_baseline_20260324_020025_vs_selective_mode_aware_answer_repair_20260324_021259_20260324_021545.json`
  - control wins `13`
  - candidate wins `5`
  - ties `1`
  - disagreement analysis:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_020025_vs_selective_mode_aware_answer_repair_20260324_021259_20260324_021545_20260324_021606_633997.md`
    - candidate mean length ratio on control-win cases: `1.571`
    - many losing cases were equal-recall but more verbose than control

- Interpretation:
  - the repair pass is real signal, not noise
  - it improves scalar recall on the formal control surface
  - but it still fails pairwise A/B because it does not revise many long
    answers that remain technically covered but too verbose or embellished

- Action:
  - keep `selective_mode_aware_answer_repair` experimental only
  - do not promote it as the default answer path

#### Answer Follow-Up: Refinement Pass Rejected

- Goal:
  - extend the repair pass so it can also remove unnecessary but supported
    detail, not just missing or unsupported content
- Implementation:
  - briefly added an experimental refinement branch on top of the repair path
  - validated it on the omission-heavy answer-precision slice

- Result:
  - `datasets/runs/profile_compare_20260324_022912.md`
  - repair `0.7000`
  - refinement `0.5500`
  - same retrieval metrics, so the drop came from answer-side degradation

- Interpretation:
  - the refinement audit over-compressed the answer and misclassified some
    needed workflow points as unnecessary
  - this is the wrong mechanism

- Action:
  - reject the refinement branch
  - revert its code from the control repo

#### Answer Follow-Up: Compact Mode-Aware Answering

- Goal:
  - keep the stronger selective mode-aware answer behavior
  - remove the verbosity / truncation problem that pairwise A/B kept rejecting
  - enforce concise completeness at generation time instead of with a post-hoc
    compression pass
- Implementation:
  - added profile:
    - `profiles/selective_mode_aware_compact_answering.yaml`
  - added compact planner and mode-specific compact prompts in:
    - `src/bgrag/answering/strategies.py`
  - added prompt/profile coverage in:
    - `tests/unit/test_answering_prompts.py`
    - `tests/unit/test_profiles.py`
  - added a verbosity-focused probe slice:
    - `datasets/eval/generated/verbosity_focus_slice.jsonl`

- Initial targeted signal:
  - `datasets/runs/profile_compare_20260324_025547.md`
  - on the verbosity-focused slice:
    - baseline `0.9167`
    - selective planned `1.0000`
    - compact `1.0000`
  - compact was materially faster than the planned selective branch on that
    slice

- Canonical control validation:
  - canonical dev first pass:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_031431.json`
    - `0.7778`
  - canonical dev repeat:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_032106.json`
    - `0.9167`
  - canonical holdout:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_031555.json`
    - `0.9250`
    - `0` forbidden violations

- Full-suite signal:
  - original `19` first full run:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_032955.json`
    - `0.9079`
    - `0` answer failures
    - `0` forbidden violations
  - rebuilt `39` full run:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_034912.json`
    - `0.8218`
    - `0` answer failures
    - `0` forbidden violations

- Pairwise cross-check:
  - original `19`:
    - `datasets/runs/pairwise_baseline_20260324_020025_vs_selective_mode_aware_compact_answering_20260324_032955_20260324_033148.json`
    - control wins `10`
    - candidate wins `9`
    - ties `0`
  - rebuilt `39`:
    - `datasets/runs/pairwise_baseline_20260323_061530_vs_selective_mode_aware_compact_answering_20260324_034912_20260324_035222.json`
    - control wins `18`
    - candidate wins `19`
    - ties `2`
  - disagreement analyses:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_020025_vs_selective_mode_aware_compact_answering_20260324_032955_20260324_033148_20260324_035234_825529.md`
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260323_061530_vs_selective_mode_aware_compact_answering_20260324_034912_20260324_035222_20260324_035234_468225.md`
  - important outcome:
    - unlike earlier answer-side branches, compact was no longer losing because
      it was longer and more truncation-prone than baseline

- Stability audit:
  - second original `19` full run:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_040035.json`
    - `0.8553`
  - later canonical dev repeat:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_040651.json`
    - `0.8333`
  - main regressions were planner-driven and not retrieval-driven:
    - `HR_002`
    - `HR_015`
    - `HR_017`
  - the compact planner sometimes dropped indispensable workflow or
    missing-detail points while staying short

- Follow-up planner tightening:
  - increased compact plan target from `1 to 5` to `2 to 6`
  - added explicit instructions to preserve operative timing / expiry branches
    and to prefer general locator guidance over long example lists
  - this was directionally sensible but did not remove the repeat instability

- Interpretation:
  - compact mode-aware answering is the strongest new answer-side branch since
    the original selective planner
  - it hits the practical rebuilt-39 quality target and materially improves the
    verbosity / truncation failure mode
  - however, it is not yet stable enough on the original `19`-case control to
    promote as the default answer path

- Action:
  - keep `selective_mode_aware_compact_answering` experimental only
  - do not promote it
  - shift the next iteration from prompt tweaking toward measurement hardening
    on the pairwise-disagreement cases

#### Measurement Hardening: Canonical Answer Pairwise Precision Slice

- Goal:
  - create a stable, human-readable promotion gate for answer-side branches on
    the exact cases where scalar recall and pairwise judgment tend to disagree
- Implementation:
  - added builder:
    - `scripts/build_answer_pairwise_precision_slice.py`
  - added machine-readable manifest:
    - `datasets/eval/manifests/answer_pairwise_precision_slice_manifest.json`
  - generated canonical slice:
    - `datasets/eval/generated/answer_pairwise_precision_slice.jsonl`

- Selection basis:
  - original full `19`-case pairwise control-win analyses for:
    - `selective_mode_aware_answer_repair`
    - `selective_mode_aware_compact_answering`
  - selected cases:
    - `HR_001`
    - `HR_003`
    - `HR_004`
    - `HR_005`
    - `HR_007`
    - `HR_008`
    - `HR_009`
    - `HR_011`
    - `HR_013`
    - `HR_015`
    - `HR_017`
    - `HR_018`
    - `HR_019`

- Validation:
  - `python scripts/build_answer_pairwise_precision_slice.py`
  - `python scripts/validate_eval_cases.py datasets/eval/generated/answer_pairwise_precision_slice.jsonl`
  - result:
    - `13` valid cases

- Interpretation:
  - this slice is precision-heavy by design
  - it should be used as a targeted answer-side promotion gate alongside:
    - canonical `parity19` dev/holdout
    - rebuilt `parity39` dev/holdout
  - it does not replace the broader parity surfaces

- Action:
  - use this slice before running another answer-side promotion attempt
  - especially for branches that improve scalar recall by changing answer
    length, structure, or planner behavior

#### Measurement Hardening: Ragas Timeout Robustness

- Goal:
  - make the secondary Ragas lane usable on disagreement-heavy slices instead
    of aborting the whole run when one metric times out
- Problem found:
  - serial Ragas runs on the new pairwise-precision slice initially failed with
    metric timeouts
  - root cause in repo code:
    - `src/bgrag/eval/ragas_runner.py` was calling `evaluate(...)` with
      `raise_exceptions=True`
    - no explicit `RunConfig` was being passed
    - one timed-out metric therefore aborted the entire run
- Implementation:
  - added settings:
    - `ragas_timeout_seconds`
    - `ragas_max_workers`
  - updated `src/bgrag/eval/ragas_runner.py` to:
    - pass explicit `RunConfig`
    - use `raise_exceptions=False`
    - normalize `NaN` metric values to `None`
    - mark fully failed rows with `skip_reason=ragas_metric_timeout_or_error`
  - documented new env knobs in:
    - `.env.example`
- Validation:
  - `pytest -q tests/unit/test_ragas_runner.py tests/unit/test_pairwise.py tests/unit/test_profiles.py`
    - `23 passed`
  - `python -m py_compile` on:
    - `src/bgrag/eval/ragas_runner.py`
    - `src/bgrag/config.py`
    - `scripts/build_answer_pairwise_precision_slice.py`

#### Targeted Validation: Compact Branch on the Pairwise-Precision Slice

- Goal:
  - test the leading compact branch on the new canonical precision gate before
    allowing another broader answer-side promotion discussion
- Evaluation surface:
  - `datasets/eval/generated/answer_pairwise_precision_slice.jsonl`
- Scalar runs:
  - baseline:
    - `datasets/runs/baseline_20260324_041922.json`
    - `0.8500`
  - baseline repeat:
    - `datasets/runs/baseline_20260324_042616.json`
    - `0.8500`
  - compact:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_044115.json`
    - `0.8500`
  - all three had:
    - `0` forbidden violations
    - packed claim-evidence recall annotated `0.98`
- Pairwise A/B:
  - `datasets/runs/pairwise_baseline_20260324_042616_vs_selective_mode_aware_compact_answering_20260324_044115_20260324_044745.json`
  - result:
    - control wins `6`
    - candidate wins `7`
    - ties `0`
  - disagreement analysis:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_042616_vs_selective_mode_aware_compact_answering_20260324_044115_20260324_044745_20260324_052605_625658.json`
    - candidate truncation-risk count `0`
    - control truncation-risk count `3`
    - mean candidate length ratio `0.763`
- Secondary Ragas cross-check:
  - baseline:
    - `datasets/runs/baseline_ragas_20260324_051602.json`
    - `context_recall_mean = 0.9615`
    - `faithfulness_mean = 0.9026`
    - `correctness_precision_mean = 0.6331`
    - `coverage_recall_mean = 0.7138`
  - compact:
    - `datasets/runs/selective_mode_aware_compact_answering_ragas_20260324_052552.json`
    - `context_recall_mean = 0.9615`
    - `faithfulness_mean = 0.9722`
    - `correctness_precision_mean = 0.5408`
    - `coverage_recall_mean = 0.6038`
- Interpretation:
  - on the canonical precision slice, compact no longer loses the qualitative
    judgment battle
  - it ties baseline on scalar recall and narrowly wins pairwise
  - it is also shorter and cleaner than baseline on average
  - however, the Ragas cross-check is mixed:
    - compact improves faithfulness
    - compact regresses correctness precision and coverage recall
  - this means the compact branch is directionally right on verbosity /
    directness, but still drops or distorts too many supported points
- Action:
  - keep compact as the leading answer-side branch
  - do not promote it yet
  - next answer-side work should focus on the specific compact losses on:
    - `HR_005`
    - `HR_009`
    - `HR_011`
    - `HR_013`
    - `HR_015`
    - `HR_017`

#### Narrow Compact-Branch Prompt Tightening

- Goal:
  - tighten the compact planner and compact answer prompts without adding a new
    route family
  - reduce extra process detail while preserving follow-on documentation steps
    and better closest-supported-context behavior on missing-detail cases
- Code changes:
  - `src/bgrag/answering/strategies.py`
    - tightened `_build_compact_mode_aware_answer_plan_prompt`
    - tightened `_build_compact_workflow_answer_prompt`
    - tightened `_build_compact_missing_detail_answer_prompt`
  - `tests/unit/test_answering_prompts.py`
    - added prompt assertions for:
      - checklist binding
      - follow-on documentation requirement
      - locator-over-contact guidance in missing-detail mode
- Validation:
  - `pytest -q tests/unit/test_answering_prompts.py tests/unit/test_profiles.py tests/unit/test_pairwise.py tests/unit/test_ragas_runner.py`
    - `47 passed`
  - `python -m py_compile src/bgrag/answering/strategies.py src/bgrag/config.py src/bgrag/eval/ragas_runner.py`
    - passed
- Precision-slice scalar check:
  - `datasets/runs/selective_mode_aware_compact_answering_20260324_054426.json`
  - result:
    - `required_claim_recall_mean = 0.8462`
    - `answer_failure_count = 0`
    - `forbidden_claim_violation_count = 0`
    - `mean_case_seconds = 34.47`
  - interpretation:
    - scalar recall stayed effectively flat vs the earlier compact run
    - latency improved materially vs the earlier compact slice run
- Precision-slice pairwise:
  - `datasets/runs/pairwise_baseline_20260324_042616_vs_selective_mode_aware_compact_answering_20260324_054426_20260324_054554.json`
  - result:
    - baseline wins `5`
    - compact wins `8`
    - ties `0`
  - pairwise-scalar analysis:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_042616_vs_selective_mode_aware_compact_answering_20260324_054426_20260324_054554_20260324_054601_836929.json`
  - interpretation:
    - this is a real improvement over the previous compact slice result
    - the compact branch is now winning the canonical precision gate more
      clearly
- Canonical `parity19` dev:
  - comparison artifact:
    - `datasets/runs/profile_compare_20260324_055800.md`
  - baseline:
    - `datasets/runs/baseline_20260324_055303.json`
    - `0.9167`
  - compact:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_055800.json`
    - `0.8889`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_055303_vs_selective_mode_aware_compact_answering_20260324_055800_20260324_060121.json`
    - baseline wins `4`
    - compact wins `4`
    - ties `1`
- Canonical `parity19` holdout:
  - comparison artifact:
    - `datasets/runs/profile_compare_20260324_055940.md`
  - baseline:
    - `datasets/runs/baseline_20260324_055452.json`
    - `0.7417`
  - compact:
    - `datasets/runs/selective_mode_aware_compact_answering_20260324_055940.json`
    - `0.8750`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_055452_vs_selective_mode_aware_compact_answering_20260324_055940_20260324_060120.json`
    - baseline wins `5`
    - compact wins `5`
    - ties `0`
- Net read:
  - the compact branch is materially healthier than before
  - it now:
    - wins the canonical precision slice more clearly
    - ties pairwise on both canonical `parity19` dev and holdout
    - substantially improves scalar holdout quality
    - stays faster than baseline on both canonical surfaces
  - but it still:
    - trails baseline scalar quality on canonical `parity19` dev
    - has not earned a clean default promotion yet

#### Rejected Follow-Up: Governing-Rule Prompt Tightening and Contextual Missing-Detail Reuse

- Goal:
  - see whether one more generic prompt tightening could recover the remaining
    canonical `parity19_dev` gap without creating a new route family
- Change:
  - tightened the compact workflow prompt further
  - temporarily switched the compact branch to reuse the contextual
    missing-detail route
- Gate:
  - `datasets/eval/dev/parity19_dev.jsonl`
- Result:
  - `datasets/runs/selective_mode_aware_compact_answering_20260324_062250.json`
  - scalar `0.8333`
  - regression from the prior compact dev run `0.8889`
- Diagnosis:
  - the change did not improve the `HR_001` omission problem
  - it regressed `HR_017` from `1.0` to `0.5`
  - the contextual missing-detail reuse was the wrong move for the compact
    branch
- Action:
  - reverted the experiment completely
  - kept the repo at the previous known-good compact state

#### Rejected Follow-Up: Compact Answer Repair Branch

- Goal:
  - test whether a repair pass could recover missing compact checklist points
    without broadening the branch further
- Temporary implementation:
  - added a short-lived experimental profile:
    - `selective_mode_aware_compact_answer_repair`
  - used the compact planner and compact route for the draft answer, then the
    existing repair planner and revision prompt
- Gate:
  - `datasets/eval/dev/parity19_dev.jsonl`
- Result:
  - `datasets/runs/selective_mode_aware_compact_answer_repair_20260324_063223.json`
  - scalar `0.8333`
- Diagnosis:
  - the branch still failed the dev gate
  - worse, the repair planner proved methodologically unsound on `HR_001`
  - it tried to add unrelated nearby supported detail such as:
    - equipment calibration
    - offers reception
    - secure handling
  - that means the current repair planner is too eager to treat adjacent
    supported detail as missing answer content
- Action:
  - removed the experimental profile and code from the repo
  - do not pursue the compact-repair lane further without a materially better
    repair-planning method

#### Structured Contract Deterministic Rendering

- Goal:
  - test a larger answer-architecture change for omission-heavy workflow and
    missing-detail cases
  - extract a cited structured contract from evidence, then deterministically
    render the final answer instead of asking for another mostly free-form
    answer
- Research inputs:
  - Cohere structured JSON docs:
    - https://docs.cohere.com/v2/docs/parameter-types-in-json
  - Instructor Cohere integration:
    - https://python.useinstructor.com/integrations/cohere/
  - Self-RAG for the adaptive-method framing:
    - https://arxiv.org/abs/2310.11511
- Implementation changes:
  - `src/bgrag/answering/strategies.py`
    - added typed cited-contract payload models
    - switched the deterministic branch to `instructor.from_cohere(...)`
    - added deterministic rendering from cited slots
    - tightened slot instructions to prefer concise, question-relevant slots
  - `profiles/structured_contract_deterministic_answering.yaml`
  - `tests/unit/test_answering_prompts.py`
  - `tests/unit/test_profiles.py`
  - `pyproject.toml`
    - added explicit `instructor` dependency
- Validation:
  - `pytest -q tests/unit/test_answering_prompts.py tests/unit/test_profiles.py`
    - `43 passed`
  - `python -m py_compile src/bgrag/answering/strategies.py`
    - passed
- Weak-slice gate:
  - first clean artifact:
    - `datasets/runs/structured_contract_deterministic_answering_20260324_072311.json`
  - baseline comparison:
    - `datasets/runs/profile_compare_20260324_072311.md`
  - scalar result on `datasets/eval/generated/answer_layer_weak_cases.jsonl`:
    - baseline `0.3214`
    - deterministic `0.6667`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_071848_vs_structured_contract_deterministic_answering_20260324_072311_20260324_072430.json`
    - baseline wins `1`
    - deterministic wins `6`
- Canonical `parity19_dev` gate:
  - comparison:
    - `datasets/runs/profile_compare_20260324_074230.md`
  - scalar result:
    - baseline `0.9167`
    - deterministic `0.8056`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_073811_vs_structured_contract_deterministic_answering_20260324_074230_20260324_074422.json`
    - baseline wins `5`
    - deterministic wins `4`
  - disagreement analysis:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_073811_vs_structured_contract_deterministic_answering_20260324_074230_20260324_074422_20260324_074502_635260.md`
- Interpretation:
  - the method is real for omission-heavy cases
  - but broad application over-explains cases that baseline already handles
    well, especially equal-recall cases like `HR_009` and `HR_013`
  - it is therefore not promotable as a general default
- Decision:
  - keep the typed deterministic branch as a useful experimental capability
  - do not promote it broadly
  - the next step should be selective activation, not broader rollout

#### Selective Workflow Contract Answering

- Goal:
  - keep baseline behavior on cases already served well by `inline_evidence_chat`
  - activate the structured-contract renderer only where the problem shape
    looks more like workflow or missing-detail reasoning
- Implementation changes:
  - `src/bgrag/answering/strategies.py`
    - added `selective_workflow_contract_inline_evidence_chat`
    - reused the compact mode-aware planner only as a route selector
    - routed to deterministic contract rendering for workflow / missing-detail
      plans and baseline inline evidence elsewhere
  - `profiles/selective_workflow_contract_answering.yaml`
  - `tests/unit/test_profiles.py`
- Validation:
  - `pytest -q tests/unit/test_answering_prompts.py tests/unit/test_profiles.py`
    - `44 passed`
  - `python -m py_compile src/bgrag/answering/strategies.py`
    - passed
- Weak-slice gate:
  - comparison:
    - `datasets/runs/profile_compare_20260324_075104.md`
  - scalar result:
    - baseline `0.2857`
    - selective workflow contract `0.6548`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_074845_vs_selective_workflow_contract_answering_20260324_075104_20260324_075300.json`
    - baseline wins `3`
    - candidate wins `4`
- Canonical `parity19_dev` gate:
  - comparison:
    - `datasets/runs/profile_compare_20260324_075938.md`
  - scalar result:
    - baseline `0.8889`
    - selective workflow contract `0.8889`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_075551_vs_selective_workflow_contract_answering_20260324_075938_20260324_080045.json`
    - baseline wins `6`
    - candidate wins `2`
    - ties `1`
- Important methodology finding:
  - on canonical `parity19_dev`, the route selector still chose the structured
    contract path for all `9` cases
  - that means the current planner-based gate is not selective enough to justify
    the extra route family
- Decision:
  - keep the branch as a recorded experiment, not a promoted profile
  - do not carry its current selector logic forward as-is
  - if we revisit selective structured contracts, the selector needs a stronger
    justification than the current compact mode-aware planner alone

#### Verifier-Gated Structured Contract Answering

- Goal:
  - stop predicting the answer route before generation
  - instead, generate the normal baseline draft first, then run a bounded
    verifier over the question, evidence, and draft answer
  - only escalate to structured-contract rewriting when the verifier sees a
    clear omission or exact-detail abstention failure
- Research inputs:
  - Self-RAG:
    - https://arxiv.org/abs/2310.11511
  - Chain-of-Verification:
    - https://arxiv.org/abs/2309.11495
  - Instructor Cohere integration:
    - https://python.useinstructor.com/integrations/cohere/
- Implementation changes:
  - `src/bgrag/answering/strategies.py`
    - added a typed post-draft verifier verdict schema
    - added a conservative verifier prompt with only two actions:
      - `keep`
      - `rewrite_structured_contract`
    - added `verifier_gated_structured_contract_inline_evidence_chat`
  - `profiles/verifier_gated_structured_contract_answering.yaml`
  - `tests/unit/test_answering_prompts.py`
  - `tests/unit/test_profiles.py`
- Validation:
  - `pytest -q tests/unit/test_answering_prompts.py tests/unit/test_profiles.py`
    - `47 passed`
  - `python -m py_compile src/bgrag/answering/strategies.py`
    - passed
- Weak-slice gate:
  - comparison:
    - `datasets/runs/profile_compare_20260324_165622.md`
  - scalar result on `datasets/eval/generated/answer_layer_weak_cases.jsonl`:
    - baseline `0.3571`
    - verifier-gated `0.3810`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_165249_vs_verifier_gated_structured_contract_answering_20260324_165622_20260324_165755.json`
    - baseline wins `3`
    - candidate wins `1`
    - ties `3`
  - route counts:
    - baseline keep `6`
    - rewrite structured-contract `1`
- Canonical `parity19_dev` gate:
  - comparison:
    - `datasets/runs/profile_compare_20260324_170545.md`
  - scalar result:
    - baseline `0.9444`
    - verifier-gated `0.8889`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_170202_vs_verifier_gated_structured_contract_answering_20260324_170544_20260324_170723.json`
    - baseline wins `2`
    - candidate wins `2`
    - ties `5`
  - route counts:
    - baseline keep `9`
    - rewrite structured-contract `0`
- Interpretation:
  - the post-draft verification family is methodologically stronger than the
    earlier pre-answer route selectors because it checks the actual draft
    against the actual evidence
  - the first implementation is too conservative and underfires
  - it is therefore safe but not useful enough yet
- Decision:
  - keep the branch as a recorded experiment
  - do not promote it
  - if revisited, the next step should be a bounded sensitivity improvement to
    the verifier, not a return to pre-answer route prediction

#### Contract-Aware Verifier-Gated Structured Contract Answering

- Goal:
  - keep the post-draft verifier approach, but stop asking the verifier to
    infer omissions directly from question plus evidence alone
  - instead, give the verifier an independently extracted structured contract
    and ask it whether the baseline draft materially covers that contract
- Implementation changes:
  - `src/bgrag/answering/strategies.py`
    - added a contract-aware verifier prompt and extraction helper
    - added `contract_aware_verifier_gated_structured_contract_inline_evidence_chat`
  - `profiles/contract_aware_verifier_gated_structured_contract_answering.yaml`
  - `tests/unit/test_answering_prompts.py`
  - `tests/unit/test_profiles.py`
- Validation:
  - `pytest -q tests/unit/test_answering_prompts.py tests/unit/test_profiles.py`
    - `49 passed`
  - `python -m py_compile src/bgrag/answering/strategies.py`
    - passed
- Weak-slice gate:
  - comparison:
    - `datasets/runs/profile_compare_20260324_172448.md`
  - scalar result:
    - baseline `0.3333`
    - contract-aware verifier `0.3810`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_172045_vs_contract_aware_verifier_gated_structured_contract_answering_20260324_172448_20260324_172904.json`
    - baseline wins `1`
    - candidate wins `3`
    - ties `3`
- Canonical `parity19_dev` gate:
  - comparison:
    - `datasets/runs/profile_compare_20260324_172753.md`
  - scalar result:
    - baseline `0.9167`
    - contract-aware verifier `0.9167`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_172153_vs_contract_aware_verifier_gated_structured_contract_answering_20260324_172753_20260324_172918.json`
    - baseline wins `2`
    - candidate wins `3`
    - ties `4`
- Interpretation:
  - this is better than the first verifier version because it no longer loses
    the canonical dev scalar gate
  - but it still underfires badly:
    - weak slice route counts were still `baseline_keep = 6`,
      `rewrite_structured_contract = 1`
    - canonical dev route counts were `baseline_keep = 9`
  - so the method family improved, but the actual trigger remained too weak
- Decision:
  - keep as a recorded experiment
  - do not promote
  - the next bounded refinement should force the verifier to reason at the
    contract-slot level rather than emit a single opaque keep or rewrite verdict

#### Contract-Slot Coverage Verifier-Gated Structured Contract Answering

- Goal:
  - replace the opaque keep or rewrite verifier with a slot-coverage verifier
    that explicitly marks which populated structured-contract slots are missing
    or weakened in the baseline draft
  - route deterministically from that slot verdict instead of trusting a single
    model-chosen action
- Research and method rationale:
  - closer to checklist-style verification than free-form repair
  - still bounded:
    - baseline draft first
    - typed structured contract
    - typed slot-coverage verdict
    - deterministic rewrite if critical populated slots are missing
- Implementation changes:
  - `src/bgrag/answering/strategies.py`
    - added `ContractSlotCoverageVerdict`
    - added a slot-coverage prompt and extraction helper
    - added `contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat`
    - later refined the deterministic renderer to drop awkward slot labels such
      as `Branch if all:` and render cleaner natural bullets instead
  - `profiles/contract_slot_coverage_verifier_gated_structured_contract_answering.yaml`
  - `tests/unit/test_answering_prompts.py`
  - `tests/unit/test_profiles.py`
- Validation:
  - `pytest -q tests/unit/test_answering_prompts.py tests/unit/test_profiles.py`
    - `52 passed`
  - `python -m py_compile src/bgrag/answering/strategies.py`
    - passed
- First slot-coverage runs before renderer cleanup:
  - weak slice:
    - `datasets/runs/profile_compare_20260324_174322.md`
    - baseline `0.3810`
    - slot-coverage verifier `0.6310`
    - route counts:
      - `rewrite_structured_contract = 4`
      - `baseline_keep = 3`
  - canonical `parity19_dev`:
    - `datasets/runs/profile_compare_20260324_174711.md`
    - baseline `0.8333`
    - slot-coverage verifier `0.8611`
    - route counts:
      - `rewrite_structured_contract = 8`
      - `baseline_keep = 1`
  - pairwise on canonical dev:
    - `datasets/runs/pairwise_baseline_20260324_173850_vs_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_174711_20260324_174818.json`
    - baseline wins `5`
    - candidate wins `4`
- Follow-up renderer cleanup:
  - goal:
    - remove presentation penalties from the deterministic answerer
  - changed:
    - cited structured-contract answers now use natural bullets without slot
      labels like `Required document or input:` or `Branch if some:`
- Post-cleanup runs:
  - weak slice:
    - `datasets/runs/profile_compare_20260324_180252.md`
    - baseline `0.2619`
    - slot-coverage verifier `0.5000`
    - route counts:
      - `rewrite_structured_contract = 5`
      - `baseline_keep = 2`
    - forbidden violations improved from `1` to `0`
    - pairwise:
      - `datasets/runs/pairwise_baseline_20260324_175348_vs_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_180252_20260324_180954.json`
      - baseline wins `0`
      - candidate wins `6`
      - ties `1`
  - canonical `parity19_dev`:
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
- Interpretation:
  - this is the first verifier-gated family that really solved the underfiring
    problem
  - it produces a real weak-slice gain and removes a forbidden-claim failure
  - but it is still too broad on canonical dev and rewrites far too many cases
  - the remaining problem is no longer omission detection; it is selective
    activation and final answer style
- Decision:
  - keep this as the strongest current verifier-gated branch
  - do not promote it yet
  - the next move should be narrower critical-slot gating or better mode-aware
    activation, not more generic prompt loosening

#### Narrow Contract-Slot Coverage Verifier-Gated Structured Contract Answering

- Goal:
  - keep the stronger slot-coverage verifier, but stop rewriting on softer
    slots that were causing broad over-activation on canonical dev
  - allow rewrites only for:
    - clearly multi-branch workflow gaps
    - missing-detail cases where the baseline did not already abstain cleanly
    - obviously corrupted baseline answers
- Implementation changes:
  - `src/bgrag/answering/strategies.py`
    - added:
      - `_looks_like_missing_detail_abstention`
      - `_looks_corrupted`
      - `narrow_contract_slot_coverage_verifier_gated_structured_contract_inline_evidence_chat`
    - increased structured-contract extraction output cap from `900` to `1400`
      tokens after rebuilt-39 runs exposed valid-contract truncation
  - `profiles/narrow_contract_slot_coverage_verifier_gated_structured_contract_answering.yaml`
  - `tests/unit/test_answering_prompts.py`
  - `tests/unit/test_profiles.py`
- Validation:
  - `pytest -q tests/unit/test_answering_prompts.py tests/unit/test_profiles.py`
    - `55 passed`
  - `python -m py_compile src/bgrag/answering/strategies.py`
    - passed
- Canonical weak slice:
  - scalar:
    - `datasets/runs/profile_compare_20260324_183103.md`
    - baseline `0.2857`
    - narrow verifier `0.5357`
  - route counts:
    - `rewrite_structured_contract = 1`
    - `baseline_keep = 6`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_182254_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_183103_20260324_183609.json`
    - baseline wins `0`
    - candidate wins `3`
    - ties `4`
- Canonical `parity19_dev`:
  - scalar:
    - `datasets/runs/profile_compare_20260324_183449.md`
    - baseline `0.8889`
    - narrow verifier `0.8611`
  - route counts:
    - `baseline_keep = 9`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_182557_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_183449_20260324_183634.json`
    - baseline wins `2`
    - candidate wins `1`
    - ties `6`
- Canonical `parity19_holdout`:
  - scalar:
    - `datasets/runs/profile_compare_20260324_185113.md`
    - baseline `0.8000`
    - narrow verifier `0.8000`
  - route counts:
    - `baseline_keep = 10`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_184212_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_185113_20260324_185243.json`
    - baseline wins `6`
    - candidate wins `1`
    - ties `3`
- Rebuilt `39` dev:
  - scalar:
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
- Rebuilt `39` holdout:
  - scalar:
    - `datasets/runs/profile_compare_20260324_201649.md`
    - baseline `0.6858`
    - narrow verifier `0.7333`
    - forbidden violations improved from `2` to `1`
  - route counts:
    - `baseline_keep = 19`
    - `rewrite_structured_contract = 1`
  - pairwise:
    - `datasets/runs/pairwise_baseline_20260324_195256_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_201649_20260324_202401.json`
    - baseline wins `8`
    - candidate wins `6`
    - ties `6`
- Interpretation:
  - this is the healthiest verifier-gated branch so far
  - it no longer over-activates on the formal `19`-case control surface
  - it produces real scalar gains on the rebuilt `39` dev and holdout surfaces
    while rewriting only a very small number of cases
  - but pairwise on the broader surfaces is still mixed rather than clearly
    favorable
- Decision:
  - keep this as the leading current verifier-gated candidate
  - do not promote it as the default yet
  - the next move should be case-level analysis of the rebuilt-39 pairwise
    disagreement cases before any promotion decision

#### Parallel Worker Expansion and Rebuilt-39 Disagreement Audit

- Goal:
  - maximize safe background experimentation while keeping shared Elasticsearch
    state conflict-free
  - make the next answer-side move depend on actual rebuilt-39 disagreement
    evidence rather than intuition
- Parallel worker setup:
  - added `docs/parallel_worker_protocol.md`
  - launched three isolated worker lanes:
    - disagreement lane
    - research lane
    - infra lane
  - all worker lanes are required to:
    - work only in disposable repo copies
    - treat the shared baseline namespace as read-only
    - create unique namespaces for any new build/index mutation
    - keep their own human-readable docs updated
- Fresh disagreement artifacts:
  - dev rebuilt-39:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_192403_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_194042_20260324_202555_20260324_203929_290278.md`
  - holdout rebuilt-39:
    - `datasets/runs/pairwise_scalar_analysis_pairwise_baseline_20260324_195256_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_201649_20260324_202401_20260324_203928_965117.md`
- Findings:
  - the new verifier branch's broad-surface value still comes from a small number
    of rewritten omission-heavy cases
    - dev:
      - `HR_030`
      - `HR_038`
    - holdout:
      - `HR_006`
      - `HR_016`
      - `HR_035`
  - there were no cases where pairwise preferred baseline despite candidate
    having higher scalar recall
  - most pairwise control wins came from equal-recall cases where baseline was
    judged slightly more faithful or less embellished
  - that makes the current bottleneck narrower:
    - less about retrieval
    - less about activation
    - more about whether rewritten answers can stay as tight and faithful as the
      baseline on the cases they do not need to change
- Decision:
  - keep the narrow verifier branch as the leading experimental answer path
  - do not promote it yet
  - use the background workers to parallelize:
    - disagreement-case diagnosis
    - research on bounded post-draft verification methods
    - infra for namespace-safe detached experimentation

#### Intervention-Only Composite Evaluation and CLI Root Fix

- Goal:
  - separate real branch effect from unchanged-case generation variance on
    conditional answer strategies
  - fix a worker-safety issue where the installed CLI could resolve artifacts
    into the wrong repo copy
- Implementation:
  - extracted reusable overall-metric aggregation into:
    - `src/bgrag/eval/run_composition.py`
  - added:
    - `scripts/build_intervention_composite_run.py`
  - updated:
    - `src/bgrag/eval/runner.py`
    - `src/bgrag/config.py`
    - `src/bgrag/cli.py`
  - added tests:
    - `tests/unit/test_run_composition.py`
    - `tests/unit/test_cli_settings.py`
- Validation:
  - `pytest -q tests/unit/test_run_composition.py tests/unit/test_answering_prompts.py tests/unit/test_profiles.py`
    - `57 passed`
  - `pytest -q tests/unit/test_cli_settings.py tests/unit/test_run_composition.py tests/unit/test_profiles.py`
    - `24 passed`
  - `py_compile` clean on the touched modules and script
- Worker safety result:
  - CLI project-root detection now prefers the current working tree and supports
    `BGRAG_PROJECT_ROOT`
  - this fixes the worker-lane issue where installed `bgrag` could otherwise
    write artifacts back into the control repo
- New measurement result:
  - rebuilt-39 dev intervention-only composite:
    - selected cases:
      - `HR_030`
      - `HR_038`
    - recall:
      - control `0.7719`
      - full candidate `0.7939`
      - intervention-only composite `0.8333`
    - pairwise:
      - candidate wins `2`
      - control wins `0`
      - ties `17`
  - rebuilt-39 holdout intervention-only composite:
    - selected case:
      - `HR_035`
    - recall:
      - control `0.6858`
      - full candidate `0.7333`
      - intervention-only composite `0.6958`
    - pairwise:
      - candidate wins `1`
      - control wins `0`
      - ties `19`
- Interpretation:
  - this is the cleanest evidence so far that the narrow verifier branch is
    good at the cases it actually changes
  - the remaining ambiguity is largely in extraction quality and slot scope, not
    in whether the intervention logic should exist at all

#### Question-Scoped Slot Pruning Follow-Up

- Goal:
  - tighten the narrow verifier branch by pruning rewritten structured-contract
    slots to the smallest sufficient set for the actual question
- Implementation:
  - added a typed contract-slot selection step in:
    - `src/bgrag/answering/strategies.py`
  - kept the existing narrow verifier gate and applied pruning only on the
    rewrite path
  - added deterministic helpers for:
    - core-slot preservation
    - contract pruning
    - quantitative branch preservation
  - added focused tests in:
    - `tests/unit/test_answering_prompts.py`
  - added a tiny focused eval slice:
    - `datasets/eval/generated/contract_pruning_focus.jsonl`
- Validation:
  - `pytest -q tests/unit/test_answering_prompts.py tests/unit/test_profiles.py`
    - `60 passed`
  - `python -m py_compile src/bgrag/answering/strategies.py`
    - passed
- New measurement result:
  - first full rebuilt-39 rerun after pruning:
    - dev:
      - `datasets/runs/narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_214054_529055_1878.json`
      - recall `0.8070`
      - forbidden violations `0`
    - holdout:
      - `datasets/runs/narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_214037_629995_a577.json`
      - recall `0.7483`
      - forbidden violations `1`
  - later broad reruns were noisy and exposed one real pruning miss:
    - `HR_030` lost the limited-tendering branch when the selector kept only
      `bottom_line` and the urgent branch
    - that regression showed up in:
      - `datasets/runs/narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_221020_555462_26e4.json`
  - the focused 3-case slice gave the cleanest read:
    - baseline:
      - `datasets/runs/baseline_20260324_221515_942877_586a.json`
      - recall `0.5444`
    - pruning branch before quantitative safeguard:
      - `datasets/runs/narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_221620_419549_a150.json`
      - recall `0.8333`
      - pairwise:
        - `datasets/runs/pairwise_baseline_20260324_221515_942877_586a_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_221620_419549_a150_20260324_221703_627913_98e7.json`
        - candidate wins `2`
        - control wins `1`
    - pruning branch after quantitative safeguard:
      - `datasets/runs/narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_222513_040835_b02c.json`
      - recall `0.7667`
- Interpretation:
  - the slot-pruning method family is real and worth keeping as an experimental
    branch
  - the quantitative branch safeguard fixes the main `HR_030` pruning miss
  - `HR_038` is now better understood as a missing-detail
    verification/extraction problem, not a pure slot-pruning problem
  - full-suite reruns remain too stochastic to treat this branch as promoted

#### Broader Missing-Detail Activation Check

- Goal:
  - test whether the narrow verifier branch should rewrite missing-detail cases
    even when the baseline already abstains, as long as the contract says the
    closest supported context is still missing
- Validation surface:
  - `datasets/eval/generated/missing_detail_focus.jsonl`
  - cases:
    - `HR_016`
    - `HR_017`
    - `HR_037`
    - `HR_038`
- Result:
  - baseline:
    - `datasets/runs/baseline_20260324_230325_074860_d4bb.json`
    - recall `0.7083`
    - forbidden violations `1`
  - broadened candidate:
    - `datasets/runs/narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_230412_461598_1f95.json`
    - recall `0.6667`
    - forbidden violations `0`
    - pairwise:
      - `datasets/runs/pairwise_baseline_20260324_230325_074860_d4bb_vs_narrow_contract_slot_coverage_verifier_gated_structured_contract_answering_20260324_230412_461598_1f95_20260324_230528_229622_6b70.json`
      - control wins `2`
      - candidate wins `2`
- Per-case read:
  - `HR_038` improved and rewrote correctly
  - `HR_037` improved somewhat
  - `HR_017` over-fired and regressed from `1.0` to `0.5`
  - `HR_016` still did not become a clean fix
- Decision:
  - reject the broader missing-detail activation rule
  - keep the more conservative gate from the feature-branch base
  - treat missing-detail follow-up as a separate method problem, not something
    to patch by broadly rewriting all abstention-style cases
