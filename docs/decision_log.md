# Decision Log

This file records durable architectural decisions so long conversations do not
become the only place where rationale lives.

## 2026-03-22

- The new repo is a clean-room rebuild based on `feat-retrieval-expensive-methods-eval`
  as a benchmark, not a code donor.
- The `feat` repo's stored scraped corpus is not canonical source input for
  this rebuild. Source truth for the new corpus should come from fresh
  collection against the official sites; `feat` artifacts are benchmark and
  evaluation references only.
- The system is Buyer’s Guide-first. Buy Canadian policy and the TBS directive
  are supporting sources by default, not equal peers.
- Phase 1 is backend-only with a Python package and CLI.
- The architecture uses a thin custom core with strong library-first bias.
- Collection, normalization, chunking, retrieval, answering, and evaluation are
  separate subsystems.
- `inline_evidence_chat` is the default answer strategy for phase 1.
- `documents_chat` remains as a benchmark profile, not the default path.
- Index builds and eval runs should be snapshot-addressable:
  - Elasticsearch indices are namespaced
  - each index build writes a manifest
  - eval runs record a run manifest with code/profile/corpus/index fingerprints
  - detached/background evals should target a specific snapshot, not mutable global state
- KG/Neo4j is explicitly out of phase-1 parity scope.
- The trustworthy phase-1 benchmark surface is the 19-case suite plus its
  dev/holdout split. A clean 39-case suite must be rebuilt here later.
- The baseline runtime should fail hard when core dependencies are missing or
  incomplete. No local keyword-search fallback, no query-time operation without
  Elasticsearch, and no query/eval runs against a partial or missing embedding
  store.
- Parameter tuning is allowed and expected when an architectural decision
  depends on it. Cohere token cost is not the limiting factor, but tuning must
  stay methodologically clean:
  - tune on dev or explicit experiment slices
  - validate on frozen holdout before promotion
  - record the tuned parameters and rationale in repo docs
- The old `dev_feat` split is not a canonical tuning surface for the structured
  judge because many of its cases are missing `required_claims`. Canonical dev
  and holdout files should instead be derived directly from the fully annotated
  `datasets/eval/parity/parity19.jsonl`.
- LLM query decomposition is the current leading retrieval architecture for
  phase 1. On the canonical `parity19_dev` and `parity19_holdout` splits it
  improved both judged answer quality and deterministic retrieval recall
  without adding brittle question-family routing logic.
- The current answer-side baseline remains `inline_evidence_chat`.
  Two generic answer-prompt experiments were evaluated on top of the promoted
  query-decomposition retriever:
  - `structured_inline_evidence_chat`
  - `planned_inline_evidence_chat`
  Neither is currently promoted. Both produced strong local or holdout wins on
  some workflow cases, but neither beat the baseline cleanly on the canonical
  dev/holdout pair without introducing new regressions.
- The 39-case rebuild and final blind acceptance set should be managed as
  explicit authored artifacts, not as ad hoc additions. The current planning
  artifacts for that work are:
  - `docs/eval_authoring.md`
  - `datasets/eval/manifests/parity39_blueprint.json`
  - `datasets/eval/manifests/final_acceptance_blueprint.json`
- The first authored parity39 batch now exists as source data in:
  - `datasets/eval/manifests/parity39_additions_batch1.json`
  and is built into:
  - `datasets/eval/parity/parity39_working.jsonl`
  - `datasets/eval/dev/parity39_dev_draft.jsonl`
  - `datasets/eval/holdout/parity39_holdout_draft.jsonl`
  The batch currently adds 7 cases across approval-authority, solicitation,
  security, debriefing, and close-out workflows.
- The rebuilt `39`-case suite now exists as a complete working surface:
  - `datasets/eval/parity/parity39_working.jsonl`
  - `datasets/eval/dev/parity39_dev_draft.jsonl`
  - `datasets/eval/holdout/parity39_holdout_draft.jsonl`
  It is now the main expanded architecture-validation surface, but the
  original `19`-case parity suite still remains the formal phase-1 control.
- A generic mode-aware answer-layer experiment was validated against the
  rebuilt `39`-case suite and the original `19`-case suite.
  Result:
  - it improves the rebuilt `39`-case surface materially
  - it removes the known forbidden-claim failure on rebuilt holdout
  - it regresses the original `19`-case full parity benchmark
  - it increases per-case latency substantially
  Decision:
  - keep `mode_aware_planned_answering` as a serious experimental profile
  - do not replace the default baseline answer strategy with it yet
- An isolated overnight autonomy pass was audited rather than trusted by
  default. The repo copy produced useful repeat-validation artifacts and a few
  low-risk tooling/test improvements, but it did not leave a clean enough
  documented milestone to justify direct promotion of any new answer strategy.
  Only the following autonomy-derived changes were accepted into control:
  - richer human-readable repeat summaries in `scripts/compare_profiles.py`
  - repo-relative unit test paths
  - a self-contained parity integration test fixture
  - a small generic wording cleanup in `src/bgrag/config.py`
- Clean control-side repeat validation of `selective_mode_aware_planned_answering`
  now exists on canonical `parity19_dev` and `parity19_holdout`.
  Result:
  - selective is modestly better on holdout
  - baseline is stronger on dev
  - therefore the signal is still mixed on the formal control benchmark
  Decision:
  - keep `selective_mode_aware_planned_answering` as the leading candidate
  - do not promote it as the default baseline yet
- Two follow-up attempts to rescue the selective branch's workflow regressions
  were tried and rejected:
  - a tighter workflow prompt
  - a planner-driven structured-workflow flag
  Both were reverted after they failed a targeted mixed regression slice.
- Retrieval-side follow-up on a low-coverage slice also failed to produce a
  better next direction:
  - `diverse_packing` showed no answer gain and worse URL coverage
  - a corrected baseline-aligned `unified_source_hybrid` run also showed no gain
- Two controlled retrieval-context experiments were then tested on a focused
  hard cluster (`HR_011`, `HR_021`, `HR_022`, `HR_028`):
  - `hierarchical_context_expansion`
  - `structural_context_expansion`
  Result:
  - both improved the slice over the plain baseline
  - both plateaued at the same answer score
  - `structural_context_expansion` achieved that plateau with lower latency and
    a cleaner deterministic mechanism
  Decision:
  - keep `structural_context_expansion` as an experimental profile
  - do not promote either context-expansion profile
  - next retrieval architecture should move earlier in the funnel, toward
    page-first or document-first selection, because the remaining failures are
    increasingly about page coverage rather than missing generic retrieval power
- A first page-first retrieval attempt (`page_seed_retrieval`) was then tested
  on the same hard cluster.
  Result:
  - it tied the structural experiment on answer quality
  - it was faster
  - it still did not resolve the key page-selection failures
  Decision:
  - reject that first implementation
  - continue toward page-first or document-first retrieval, but with a stronger
    page-ranking mechanism than intro-chunk aggregation alone
- Stronger document-reranked page seeding was then tested in three forms:
  - global document rerank seeding
  - localized graph-scoped document rerank seeding
  - lineage-only document rerank seeding
  Result:
  - all three materially beat the baseline on the specific hard cluster
  - none survived broad promotion on the canonical `parity19` control,
    especially holdout
  - the family is strongest on page-coverage workflow/navigation problems and
    weakest on reciprocal-procurement and support-linked detail questions
  Decision:
  - keep the page-rerank family as an experimental targeted capability only
  - do not promote any of those variants as the default retriever
  - if revisited, the next step should be a principled selective activation
    gate rather than another broad retrieval swap
- A principled selective activation gate was then tested using an LLM-based
  retrieval-mode selector over the question plus the baseline evidence summary.
  Result:
  - it materially improved over broad page-rerank activation
  - it preserved the main hard-cluster gains on `HR_021`, `HR_022`, and `HR_028`
  - but it still underperformed the promoted baseline on the canonical
    `parity19` control, especially holdout
  Decision:
  - keep `selective_localized_document_rerank_seed_retrieval` as an
    experimental profile only
  - do not promote selective page-rerank activation as the default retriever
  - the next sound move is to return to answer-side improvements while keeping
    the baseline retriever as control
- A new answer-side structured-contract experiment was then tested to replace
  free-form coverage points with explicit mode-specific slots.
  Result:
  - it improved the targeted answer-layer weak slice materially
  - it did not beat the canonical `parity19` control surface
  - a second schema iteration with extra workflow slots produced no additional
    gain on the weak slice
  Decision:
  - keep `structured_contract_mode_aware_answering` as an experimental profile
    only
  - do not promote it
  - the strongest current answer-side candidate remains
    `selective_mode_aware_planned_answering`
  - the next clean experiment should preserve that candidate and add a narrow
    repair/completeness pass rather than replacing it with a heavier planner
- A secondary Ragas-based measurement lane has now been added as an explicit
  cross-check, not a replacement for the main eval harness.
  Decision details:
  - keep the current deterministic retrieval metrics and structured Cohere judge
    as the primary promotion surface
  - add `bgrag eval-ragas` as a secondary claim-oriented measurement surface
  - use claim-oriented metrics aligned with Cohere's RAG-eval framing:
    - `context_recall`
    - `faithfulness`
    - `correctness_precision`
    - `coverage_recall`
  - do not use the stock `ragas.llm_factory(..., provider=\"cohere\")` path in
    the current environment because it is incompatible with `cohere.ClientV2`
  - instead, use a repo-native wrapper built on:
    - `instructor.from_cohere`
    - `ragas.llms.base.InstructorLLM`
  - keep this lane explicitly secondary because:
    - it is slower than the existing judge path
  - it still uses the same model family as generation
  - Cohere's own RAG evaluation guidance recommends an independent strong
      evaluator when possible
- A second secondary measurement lane has now been added for blinded pairwise
  A/B comparison of two existing run artifacts.
  Decision details:
  - use the official OpenAI SDK rather than a bespoke HTTP client
  - use `responses.parse(...)` with a Pydantic verdict model
  - use `gpt-5.4`, not a mini model, for this lane
  - compare two already-generated run artifacts instead of regenerating answers
    just to judge them
  - cache pairwise verdicts locally with `diskcache`
  - also set OpenAI prompt-cache fields on the request
  - keep pairwise judging as a promotion cross-check, not the sole metric
  First real result:
  - baseline vs `mode_aware_planned_answering` on the full original `19`-case
    control suite yielded:
    - control wins `11`
    - candidate wins `6`
    - ties `2`
  - baseline vs `mode_aware_planned_answering` on the rebuilt full `39`-case
    suite yielded:
    - control wins `22`
    - candidate wins `14`
    - ties `3`
  Interpretation:
  - pairwise judging currently supports the existing decision not to promote the
    mode-aware branch over baseline on the formal control surface
  - pairwise judging is also not currently endorsing the mode-aware branch on
    the rebuilt `39`-case suite, despite its stronger required-claim recall
  - therefore future promotion decisions should treat pairwise A/B judgment as
    a meaningful counterweight to single-run scalar recall improvements
- The new pairwise-vs-scalar analysis confirms the practical failure mode:
  - answer-side candidates are often getting scalar recall gains through
    longer, more detailed answers
  - pairwise judging then rejects those answers for unnecessary detail,
    truncation risk, or weaker faithfulness
  Decision:
  - treat concise completeness, not raw coverage expansion, as the answer-side
    target
  - keep pairwise A/B as a real promotion gate, not a secondary vanity metric
- A narrow post-draft repair pass (`selective_mode_aware_answer_repair`) has
  now been validated.
  Result:
  - it improves scalar recall on canonical holdout and on one full original
    `19`-case run
  - but it still loses pairwise A/B against the current baseline:
    - control wins `13`
    - candidate wins `5`
    - ties `1`
  Decision:
  - keep the repair pass as an experimental branch only
  - do not promote it
- A further refinement branch that tried to remove unnecessary but supported
  detail was tested on the answer-precision slice and rejected immediately.
  Result:
  - repair `0.7000`
  - refinement `0.5500`
  Decision:
  - revert the refinement code from the control repo
  - do not carry rejected prompt families forward in the main codebase
- A compact selective answer branch has now been validated.
  Result:
  - best original `19` full run:
    - `0.9079`
  - best rebuilt `39` full run:
    - `0.8218`
  - pairwise cross-checks are much healthier than earlier answer-side branches:
    - original `19`: control `10`, candidate `9`
    - rebuilt `39`: control `18`, candidate `19`, ties `2`
  - however, repeated canonical dev and full-`19` runs regressed materially,
    showing planner instability rather than a cleanly stable improvement
  Decision:
  - keep `selective_mode_aware_compact_answering` as the leading experimental
    answer-side branch
  - do not promote it as the default baseline yet
  - do not keep trying to rescue it by piling on more prompt-only tweaks
- A canonical answer-side precision gate should now exist as a first-class
  measurement surface.
  Decision details:
  - add `datasets/eval/generated/answer_pairwise_precision_slice.jsonl`
  - generate it reproducibly from `datasets/eval/parity/parity19.jsonl`
    using:
    - `scripts/build_answer_pairwise_precision_slice.py`
  - keep its rationale in:
    - `datasets/eval/manifests/answer_pairwise_precision_slice_manifest.json`
  - use this slice for answer-side promotion checks on branches that alter
    answer structure, compactness, or planning
  - do not let this slice replace the canonical `parity19` dev/holdout or the
    rebuilt `parity39` surfaces; it is a targeted complement
- Internal analysis now supports a methodology change:
  - before running another answer-side branch, first check whether it can at
    least tie or beat baseline on the canonical pairwise-precision slice
  - answer-side promotion should now be judged on all of:
    - scalar recall on canonical `parity19`
    - scalar recall on rebuilt `parity39`
    - pairwise A/B
    - the pairwise-precision slice
  - Ragas `faithfulness` and `correctness_precision` should be used as
    secondary spot checks on disagreement-heavy slices rather than only on
    smoke cases
- The Ragas lane has now been hardened for disagreement-heavy slices.
  Decision details:
  - do not force `raise_exceptions=True` in the repo wrapper
  - pass explicit `RunConfig` from repo settings
  - treat metric-level timeouts as missing values, not as a reason to abort the
    full evaluation
  - current repo defaults:
    - `ragas_timeout_seconds = 480`
    - `ragas_max_workers = 4`
- The new canonical pairwise-precision slice has now been validated against the
  leading compact branch.
  Result:
  - scalar recall:
    - baseline `0.8500`
    - compact `0.8500`
  - pairwise:
    - baseline wins `6`
    - compact wins `7`
  - Ragas:
    - compact improves `faithfulness`
    - compact regresses `correctness_precision` and `coverage_recall`
  Decision:
  - compact remains the leading answer-side branch
  - compact is still not promotable
  - the next answer-side work should target the compact branch's remaining
    precision/coverage losses on the canonical precision slice, not launch a
    new broad branch immediately
- A narrow compact-branch tightening pass has now been validated.
  Result:
  - canonical precision slice:
    - scalar `0.8462`
    - pairwise wins `8` vs baseline `5`
  - canonical `parity19_dev`:
    - baseline `0.9167`
    - compact `0.8889`
    - pairwise tie `4-4-1`
  - canonical `parity19_holdout`:
    - baseline `0.7417`
    - compact `0.8750`
    - pairwise tie `5-5`
  Decision:
  - keep `selective_mode_aware_compact_answering` as the leading experimental
    answer-side branch
  - do not promote it to default yet, because canonical `parity19_dev` still
    favors baseline on scalar quality
  - treat the branch as materially stronger than before:
    - it now clears the canonical precision gate more convincingly
    - it no longer loses the canonical pairwise checks outright
    - it substantially improves canonical holdout scalar quality
  - next answer-side work should be case-targeted and generic:
    - preserve the new checklist-binding discipline
    - improve the remaining compact dev regressions without adding another
      broad route family
- A follow-up compact prompt tweak that reused the contextual missing-detail
  route and pushed harder on governing-rule phrasing was rejected.
  Result:
  - canonical `parity19_dev` dropped to `0.8333`
  - it did not improve the remaining `HR_001` omission
  - it regressed `HR_017`
  Decision:
  - revert the tweak completely
  - do not reuse the contextual missing-detail route inside the compact branch
    for now
- A compact-plus-repair branch was also rejected.
  Result:
  - canonical `parity19_dev` also came in at `0.8333`
  - the repair planner tried to add unrelated nearby supported detail on
    `HR_001`, which is methodologically unsound
  Decision:
  - remove the compact-repair code and profile from the repo
  - do not pursue repair-based promotion work until the repair planner can
    distinguish required answer content from adjacent supported context much
    better
- A typed structured-contract renderer was then validated as a larger
  answer-architecture experiment.
  Result:
  - on the omission-heavy weak slice:
    - baseline `0.3214`
    - deterministic structured-contract `0.6667`
    - pairwise wins `6` vs baseline `1`
  - on canonical `parity19_dev`:
    - baseline `0.9167`
    - deterministic structured-contract `0.8056`
    - pairwise baseline wins `5` vs candidate `4`
  Diagnosis:
  - the method is real for omission-heavy workflow and missing-detail cases
  - but broad application over-explains direct-rule cases that the baseline
    already handles well
  Decision:
  - keep the deterministic structured-contract path only as an experimental
    capability
  - do not promote it as a general default
  - the method implication is to try selective activation, not broader rollout
- A selective workflow-contract branch was then tested.
  Result:
  - weak slice:
    - baseline `0.2857`
    - selective workflow-contract `0.6548`
    - pairwise candidate wins `4` vs baseline `3`
  - canonical `parity19_dev`:
    - baseline `0.8889`
    - selective workflow-contract `0.8889`
    - pairwise baseline wins `6`, candidate `2`, ties `1`
  Diagnosis:
  - the selector still routed all canonical dev cases through the structured
    contract path, so it was not truly selective
  Decision:
  - do not promote the selective workflow-contract branch
  - do not keep the current planner-only selector logic
  - if selective structured contracts are revisited, require a stronger route
    selector than the current compact mode-aware plan alone
- A post-draft verifier-gated structured-contract branch has now been tested.
  Result:
  - weak omission slice:
    - baseline `0.3571`
    - verifier-gated `0.3810`
    - pairwise baseline wins `3`, candidate `1`, ties `3`
  - canonical `parity19_dev`:
    - baseline `0.9444`
    - verifier-gated `0.8889`
    - pairwise `2-2-5`
  Diagnosis:
  - the method family is more principled than pre-answer route selection
    because it verifies a real baseline draft against real evidence
  - the first implementation is too conservative and underfires
  - broad promotion would therefore add complexity without enough quality gain
  Decision:
  - keep verifier-gated specialization as a live method family
  - do not promote the current implementation
  - if revisited, change verifier sensitivity in a bounded way rather than
    returning to pre-answer routing
- Two stronger post-draft verifier refinements have now been tested:
  - `contract_aware_verifier_gated_structured_contract_answering`
  - `contract_slot_coverage_verifier_gated_structured_contract_answering`
  Result:
  - the contract-aware verifier is safer than the first verifier version and no
    longer loses canonical dev scalar quality, but it still underfires
  - the slot-coverage verifier solves the underfiring problem and materially
    improves the weak omission-heavy slice, but it over-activates on canonical
    dev
  - renderer cleanup helped answer quality on the rewritten cases, but did not
    solve the over-activation problem
  Decision:
  - keep post-draft verification as the right answer-method family
  - keep the slot-coverage verifier as the strongest current verifier-gated
    branch
  - do not promote it yet
  - the next refinement should narrow activation using critical-slot or
    mode-aware thresholds, not loosen prompts further or return to pre-answer
    routing
- A narrower slot-coverage verifier gate has now been validated:
  - `narrow_contract_slot_coverage_verifier_gated_structured_contract_answering`
  Result:
  - canonical `19` dev and holdout are now close to baseline behavior because
    the branch rewrites little or nothing there
  - rebuilt `39` dev improves from `0.7719` to `0.7939`
  - rebuilt `39` holdout improves from `0.6858` to `0.7333`
  - rebuilt `39` holdout forbidden violations improve from `2` to `1`
  - pairwise on rebuilt `39` dev and holdout is still mixed rather than
    clearly favorable
  Decision:
  - keep the narrow slot-coverage verifier as the leading verifier-gated
    candidate
  - do not promote it to default yet
  - the next step should be disagreement-case analysis on rebuilt `39`, not
    another broad prompt rewrite
- Intervention-only composition is now the preferred measurement for conditional
  answer branches whose unchanged `baseline_keep` cases would otherwise add
  non-method generation variance.
- The narrow verifier branch remains the leading conditional answer path.
  Current best read:
  - the intervention logic itself is good
  - the remaining problem is structured-contract extraction or slot scope, not
    broader activation tuning
- A staged promotion branch is now the preferred integration method for
  extracting proven ideas from the broader verifier branch.
  Decision:
  - promote only:
    - evaluation hardening
    - conditional/intervention-only comparison tooling
    - split-safe exactness slices
    - the narrow missing-detail exactness sub-path
  - do not drag the broader verifier family into the promotion branch
- Claude collaboration should use a durable/local split.
  Decision:
  - keep repo-shared Claude guidance in committed `CLAUDE.md`
  - keep session IDs and transcripts in `.claude/session_local/` and out of git
  - use Claude Opus 4.6 with max effort through a resumable wrapper
  - treat Claude as a peer reviewer and advisor, not as the promotion authority
- Conditional comparison summaries must explicitly surface preserved-baseline
  drift on non-selected cases.
  Decision:
  - promotion summaries now record:
    - `non_selected_changed_case_count`
    - `non_selected_changed_case_ids`
    - `non_selected_preserved_baseline`
  - intervention-only composites remain useful, but they are not allowed to
    silently imply that untouched candidate cases matched control
- Judge normalization must validate claim identity, not just list length.
  Decision:
  - reject judge payloads whose `claim` text does not align with the authored
    required/forbidden claims at the same index
  - this prevents reordered or paraphrased claim lists from being scored against
    the wrong target claims
- Conditional-compare tooling should emit explicit progress for background runs.
  Decision:
  - keep progress reporting in the runner and CLI wrapper
  - prefer rerunning long comparisons on current branch code over trusting stale
    background jobs that were started before harness changes
- The narrow exactness path is mergeable as code and as an intervention-only
  measured method, but not as a standalone full-suite promotion surface.
  Decision:
  - merge the exactness sub-path and supporting eval hardening
  - do not read raw full-suite `baseline` vs
    `missing_detail_exactness_verifier_gated_structured_contract_answering`
    runs as meaningful direct A/B evidence
  - use intervention-only composites and exactness-family slices when deciding
    whether the exactness path is helping
- `HR_016` should no longer rely on a single narrow forbidden claim.
  Decision:
  - explicitly block adjacent PSPC or `PWGSC-TPSGC` form-number hallucinations
    in all canonical `HR_016` surfaces
  - keep the required claims unchanged
  Reason:
  - the case concept was sound, but judged runs could still surface plausible
    unrelated form numbers unless the failure contract named that loophole
- The exactness family should be expanded with newly authored cases rather than
  by reclassifying adjacent workflow cases.
  Decision:
  - add an audited `exactness_family` dev/holdout surface
  - include:
    - the existing split-safe parity39 exactness cases
    - one new dev case from the Controlled Goods Directorate page
    - one new holdout case from the PSD audit / GCDocs page
  - do not add the trade-agreement-unit contact-detail idea as the current dev
    expansion case
  Reason:
  - the controlled-goods page is the cleaner negative exactness source because
    it explicitly says to contact the directorate via email or telephone while
    still withholding the actual details
- Conditional-compare tooling must reject selector misuse when a candidate run
  clearly intervened.
  Decision:
  - if the requested `intervention_paths` do not match any observed
    non-baseline `selected_path` values, fail fast with an explicit error
  - keep allowing zero selected cases when the candidate genuinely stayed on
    `baseline_keep`
  Reason:
  - silently accepting a profile name instead of a raw `selected_path` label
    produces a false zero-intervention read and contaminates promotion logic
- `feat/exactness-surface-expansion` is mergeable as eval-surface work even
  though the narrow exactness gate still underfires on `EX_001` and `EX_002`.
  Decision:
  - merge the branch for:
    - tightened `HR_016`
    - the new audited `exactness_family` regression surface
    - the conditional-compare selector guard
  - do not mix gate-sensitivity tuning into this merge
  Reason:
  - the branch contract is a stronger, more honest exactness regression gate
  - the remaining underfiring is a follow-up answer-method problem, not a
    reason to block the eval-surface improvement itself
- Repo-local `.env` values must override ambient shell environment in script
  entrypoints, not just in the main CLI.
  Decision:
  - `compare_conditional_profile.py` and `compare_profiles.py` now use the same
    repo-env precedence helper as `bgrag.cli`
  Reason:
  - stale machine-level API keys were able to override the repo `.env` and
    silently break pairwise evaluation despite the repo being configured
    correctly
- The narrow exactness sub-path has now cleared the holdout exactness-family
  pairwise gate on the cases it actually rewrites.
  Decision:
  - treat the rewrite quality question as answered for `HR_016` and `HR_037`
  - move the next branch to exactness-gate sensitivity for `EX_001` and
    `EX_002`, not another rewrite-style redesign
  Reason:
  - holdout intervention-only pairwise is:
    - candidate wins `2`
    - control wins `0`
    - ties `1`
