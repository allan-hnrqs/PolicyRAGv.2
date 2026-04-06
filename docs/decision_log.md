# Decision Log

This file records durable architectural decisions so long conversations do not
become the only place where rationale lives.

## 2026-04-04

- On the wider `acceptance49` surface, the current serious control
  (`baseline_vector_rerank_shortlist`) is now showing a clearer authority-page
  failure family:
  - cases such as `HR_043` and `HR_048` already surfaced the decisive TBS
    directive chunks in the raw shortlist
  - the current Buyer’s-Guide-first packing then discarded most of that
    authority material before answer generation
  Decision:
  - treat authority/support-page suppression as a real retrieval/packing issue,
    not just an answer-prompt issue
  - use the wider `acceptance49` surface as the place to validate that family,
    while keeping `parity19` frozen as control
- A first global authority-support reserve profile was then tested:
  - profile: `baseline_vector_rerank_shortlist_authority_reserve`
  Result:
  - it rescued authority retrieval on some new cases, especially `HR_043`
  - but it regressed unrelated dev and holdout cases
  Decision:
  - reject a global support-slot reserve
  - do not distort the whole fast lane just to rescue one support-page family
- A more selective authority-support reserve profile was then tested:
  - profile: `baseline_vector_rerank_shortlist_selective_authority_reserve`
  Result:
  - retrieval-only improved cleanly on `acceptance49`
  - matched holdout judged quality improved from `0.6580` to `0.7113`
  - matched dev judged quality regressed from `0.7118` to `0.6667`
  - the decisive authority cases still often failed because only one TBS chunk
    was preserved, which improved doc-level metrics but not full claim support
  Decision:
  - keep the selective reserve as useful diagnosis
  - do not promote it as a general replacement
- A narrower authority-cluster preservation profile was then tested:
  - profile: `baseline_vector_rerank_shortlist_selective_authority_cluster`
  Result:
  - on holdout retrieval-only, chunk-support recall improved materially:
    - overall chunk-support mean: `0.4800 -> 0.8533`
    - `HR_043`: `0.0 -> 1.0`
    - `HR_048`: `0.0 -> 0.6667`
  - matched holdout judged quality improved modestly:
    - `0.7260 -> 0.7460`
  - matched dev judged quality regressed slightly:
    - `0.6840 -> 0.6771`
  - the holdout gain came mainly from `HR_043`, while the dev regression was
    spread across a handful of answer-side swings
  Decision:
  - classify the authority-cluster intervention as `mixed`
  - keep it as a targeted failure-family candidate, not as the new default
  - the next step should be another small cycle:
    - either expand this authority/governance case family further
    - or isolate it as an intervention-only capability rather than a global
      replacement
- A paraphrase-robustness audit was then added for wording-triggered runtime
  heuristics:
  - manifest:
    `datasets/eval/manifests/heuristic_trigger_robustness_v1.json`
  - generated slice:
    `datasets/eval/generated/heuristic_trigger_robustness_v1.jsonl`
  - audit runner:
    `scripts/run_heuristic_trigger_audit.py`
  Result:
  - the audit found wording-trigger changes in runtime heuristics even when the
    question meaning was preserved:
    - authority-support trigger changed on `2/7` paraphrase pairs
    - question-risk level changed on `2/7`
    - exactness-sensitive changed on `2/7`
  - the strongest current heuristic-heavy candidate
    (`baseline_vector_rerank_shortlist_selective_authority_cluster`) did show
    a real win on the original source cases:
    - source-case mean required-claim recall:
      `0.4952 -> 0.6619`
  - but on the paraphrased versions of those same cases, that advantage
    disappeared:
    - control: `0.3476`
    - cluster: `0.3476`
    - retrieval metrics also regressed for the cluster profile on the
      paraphrase slice
  Decision:
  - do not treat wording-triggered heuristic gains as generalized until they
    survive paraphrase robustness checks
  - the current authority-cluster heuristic fails that proof
  - keep it as a diagnostic/experimental intervention only, not as a
    production-facing serving behavior

- The first selective indexed-retry candidate has now been completed as a full
  large optimization-loop cycle:
  - cycle id: `loop03_large`
  - hypothesis: `hybrid_retry_trigger`
  - profile:
    `baseline_vector_rerank_shortlist_hybrid_retry`
  Result:
  - canonical dev judged quality improved slightly from the current serious
    control's `0.8611` to `0.8889`, but that change came entirely from one case
    (`HR_015`: `0.5 -> 0.75`)
  - canonical holdout judged quality regressed from `0.7750` to `0.7500`
  - the holdout regression also came down to one case moving the wrong way:
    - `HR_002`: `0.5 -> 0.25`
  - the persistent failure surfaces also regressed:
    - dev: `0.5833 -> 0.4167`
    - holdout: `0.4375 -> 0.3750`
  - product serving mean request time increased from the current repeated
    control mean of about `17.7s` to `28.6s`
  - retrieval-only holdout did not improve over control
  Decision:
  - reject the first hybrid indexed-retry trigger as currently designed
  - do not add retrieval assessment plus retry overhead to the fast lane
    without a clearer holdout-quality win
  - the next large hypothesis should be narrower again:
    - either better trigger calibration on the failure slice
    - or a more selective browse-escalation path that only fires where the
      indexed lane demonstrably cannot recover
- A deterministic bundle-risk audit layer has now been added as a small-cycle
  diagnostic:
  - scorer: `bgrag.serving.bundle_risk.assess_bundle_risk(...)`
  - audit runner: `scripts/run_bundle_risk_audit.py`
  Result on the frozen serious-control eval artifacts:
  - holdout:
    - low-recall cases at threshold `< 0.75`: `4`
    - flagged low-recall cases: `3`
    - flag recall on low-recall cases: `0.75`
    - flag precision on low-recall cases: `0.375`
  - dev:
    - low-recall cases at threshold `< 0.75`: `2`
    - flagged low-recall cases: `1`
    - flag recall on low-recall cases: `0.5`
    - flag precision on low-recall cases: `0.1429`
  Diagnosis:
  - bundle-only weakness signals do catch some genuine retrieval/packing misses
  - but they also overflag many good cases
  - they miss some answer-side undercoverage cases where the packed bundle looks
    healthy enough structurally
  Decision:
  - do not use bundle-only risk as the sole retry or escalation trigger
  - keep it as an auxiliary deterministic signal
  - the next serving trigger should combine:
    - bundle-risk features
    - structured retrieval assessment
    - a narrower question-risk / exactness-risk layer
- A single-call answer-side confidence sidecar was tested as an isolated
  replay-only experiment:
  - strategy: `inline_evidence_chat_with_confidence_sidecar`
  - method: same answer call, but constrained to return JSON containing the
    answer plus confidence fields
  Result:
  - canonical dev answer replay regressed from `0.8611` to `0.7222`
  - canonical holdout answer replay regressed from `0.7500` to `0.6667`
  - one holdout case (`HR_012`) failed outright because the model did not
    return valid JSON
  - the sidecar confidences were badly miscalibrated on several misses:
    - `HR_001`, `HR_005`, `HR_018` all reported high confidence despite poor
      judged recall
  Decision:
  - do not promote a JSON answer-plus-confidence sidecar as a serving answer
    format
  - if confidence sidecars are revisited, keep them internal and treat them as
    auxiliary signals only, not as a replacement for natural answer output
- The first large optimization-loop cycle on the new persistent protocol has
  now been completed:
  - cycle id: `loop02_large`
  - hypothesis: `ranked_diverse_packing`
  - profile:
    `baseline_vector_rerank_shortlist_ranked_diverse`
  Result:
  - canonical dev judged quality regressed from the current serious control's
    `0.8611` to `0.8333`
  - canonical holdout judged quality regressed from `0.7750` to `0.7500`
  - the persistent failure holdout slice improved from `0.4375` to `0.5625`,
    but that gain came mainly from `HR_002`
  - `HR_011` retrieval coverage improved, but the final answer still did not
    improve there
  - `HR_018` regressed sharply from `1.0` to `0.0` despite packed evidence
    remaining fully present
  - product serving mean request time on the candidate run was `32.9s`, well
    above the current repeated control mean of `17.7s`
  Decision:
  - reject broad ranked-passthrough plus ranked-diversity packing as the next
    promotion direction
  - do not generalize a cross-source packing reorder across the whole fast lane
  - the next large hypothesis should be selective and targeted:
    - either retrieval retry / escalation on genuinely weak evidence
    - or a narrower intervention on the specific failure families
- A repo-native persistent optimization-loop scaffolding layer now exists.
  Decision:
  - keep `baseline_vector_rerank_shortlist` as the explicit control profile
  - keep canonical surfaces and failure surfaces as named manifests rather than
    ad hoc shell history
  - record each large or small cycle as a first-class artifact
  - use the new cycle runner and protocol docs instead of letting loop state
    live only in conversations
- A strict tiered agentic candidate architecture is now the active design
  direction.
  Decision:
  - indexed RAG remains the default serving lane
  - official-site browsing is an explicit escalation lane
  - the candidate implementation uses structured retrieval assessment plus one
    widened indexed retry before browse escalation
  - silent infrastructure fallbacks are not allowed in this path
- Native Elasticsearch RRF is now the target hybrid fusion method for the
  strict agentic candidate.
  Repo reality:
  - the current local Elasticsearch cluster rejects native RRF because of a
    license error
  Decision:
  - keep the strict candidate profiles on native RRF anyway
  - fail hard if the cluster cannot provide it
  - do not silently fall back to repo-side hybrid fusion inside those profiles
- PydanticAI is now the chosen orchestration library for the bounded
  official-site browse escalation path.
  Decision:
  - keep the escalation loop bounded by explicit tool budgets
  - do not mainline a broader Haystack or LangGraph rewrite unless the current
    PydanticAI-based spike proves inadequate
- A bounded retrieval-budget sweep was run over the serious rerank-shortlist
  lane on the full `19`-case parity surface.
  Result:
  - shortlist/rerank budget does matter
  - the best bounded point in that sweep was:
    - `top_k=16`
    - `candidate_k=64`
    - `rerank_top_n=64`
    - `per_query_candidate_k=32`
  - it improved judged quality over the current serious default on that surface
  - but it still did not reach `0.9+`
  Decision:
  - keep that budget point as the leading retrieval-budget candidate
  - do not claim that larger retrieval budgets alone solve the repo's product target
- The leading bounded retrieval-budget point was then validated on the clean
  dev/holdout split.
  Result:
  - judged end-to-end quality improved on both splits
  - but holdout retrieval-only coverage regressed materially
  Decision:
  - do not promote that tuned budget point as the new default serious profile
  - keep it as mixed evidence, not a clean win
- A richer bounded tool-agent comparator was then tested with an additional
  indexed retrieval tool.
  Result:
  - it still underperformed the serious indexed RAG lane
  - it remained browse-heavy and used the indexed retrieval tool sparingly
  Decision:
  - do not treat “agentic” branding by itself as evidence of a better
    architecture
  - if agentic orchestration is revisited, judge it by tool trace quality,
    latency, and answer quality, not by the presence of a planner loop
- Retrieval-backend alternatives should now be evaluated as bounded spikes, not
  as speculative rewrites.
  Decision:
  - keep the current serious Elasticsearch/Cohere stack as the control
  - run OpenSearch first if a backend alternative is explored
  - run Qdrant second only if OpenSearch is unconvincing
  - do not broaden the spike to Weaviate or Vespa unless the earlier options
    fail to produce a compelling path
- The bounded OpenSearch backend spike has now been completed.
  Result:
  - retrieval-only behavior was competitive with the current control
  - dev end-to-end quality improved slightly
  - holdout end-to-end quality did not improve
  - latency was materially worse on both splits
  Decision:
  - do not promote OpenSearch as the new primary backend
  - treat the bounded OpenSearch spike as unconvincing
  - if backend alternatives are revisited, move to the next planned spike
    rather than relitigating the same like-for-like OpenSearch swap

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

## 2026-04-03

- A controlled follow-up on grounded Cohere `documents` answering was completed
  after the original Phase 4 experiment.
  The follow-up removed the main confounds by:
  - fixing citation parsing for Cohere v2 `sources`
  - aligning the inline and grounded-doc answer contracts
  - replaying answer generation on frozen Phase 1 evidence bundles so retrieval
    and planning stayed fixed
  - testing both per-chunk and single-blob grounded-document packaging
  - testing both free-form and structured-contract grounded-document answer
    paths
  Result:
  - grounded `documents` still underperformed inline on the repo's
    synthesis-heavy tasks
  - the failure was not explained by trimming, citation parsing, prompt drift,
    or per-chunk packaging alone
  Decision:
  - keep `inline_evidence_chat` as the mainline answer path
  - keep grounded `documents` only as a documented experimental branch, not as
    the promoted serving direction

## 2026-04-04

- The repo now has a broader diagnostic acceptance surface built from
  low-context-authored cases rather than from iterative repo-local tuning.
  Decision:
  - keep `parity19` frozen as the optimization control surface
  - keep `parity39` as the broader historical diagnostic bank
  - add `acceptance49` as the next wider acceptance/debugging surface:
    - `datasets/eval/parity/acceptance49_working.jsonl`
    - `datasets/eval/dev/acceptance49_dev_draft.jsonl`
    - `datasets/eval/holdout/acceptance49_holdout_draft.jsonl`
  - source it from:
    - `datasets/eval/manifests/acceptance49_blueprint.json`
    - `datasets/eval/manifests/acceptance49_additions_batch1.json`
    - `scripts/build_acceptance49_working.py`
  Reason:
  - the current failure pool is too small and too repetitive to drive the next
    architectural decisions with confidence
  - the new batch adds workflow, authority, exactness, source-boundary, and
    cross-page synthesis coverage without mutating the frozen control
