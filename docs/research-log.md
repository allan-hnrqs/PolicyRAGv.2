# Research Log

This file records external research that influences architecture decisions in
this repo. Prefer official docs, primary sources, and reproducible technical
references over secondary summaries.

## Active themes

- Cohere-specific RAG patterns
- source-aware retrieval and routing
- hierarchical retrieval
- chunking strategies
- provenance and citation handling
- evaluation governance and anti-overfitting practice
- graph-ready RAG and GraphRAG

## Notes

- Use this file as a running log, not a polished report.
- Link research findings to ADRs once they materially affect architecture.
- If a useful source requires signup or manual access, record the exact manual
  task needed rather than silently replacing it with a weaker source.

## 2026-03-22

### Cohere / architecture findings

- The installed Cohere SDK supports `ClientV2.chat(messages=..., documents=...)`
  and `ClientV2.rerank(...)`, so the phase-1 runtime can be built directly on
  the v2 SDK rather than through compatibility wrappers.
- Live integration check: the Cohere embed API rejected a 100-text request with
  `invalid request: total number of texts must be at most 96`, so the embedder
  should batch requests explicitly instead of assuming arbitrarily large inputs.
- The cleanest phase-1 answer boundary is still:
  - retrieve broadly
  - build a typed evidence bundle
  - answer with `inline_evidence_chat` by default
  - keep `documents_chat` as a benchmark profile only
- KG remains an experiment lane only; graph-ready IDs and lineage belong in
  phase 1, but a Neo4j runtime dependency does not.

### Parity findings

- The local `19`-case human-realistic suite is the trustworthy phase-1 parity
  surface.
- The current `39`-case direction is not cleanly canonical in the local repo
  and must be rebuilt honestly here instead of inherited from orphaned lab
  artifacts.

### Source-domain findings

- The new repo should not inherit `feat`'s stored corpus package as a trusted
  source of truth. Fresh collection from the official Buyer’s Guide / policy /
  directive sites is the intended canonical input path.
- The current reference corpus is overwhelmingly Buyer’s Guide-heavy:
  - `228` Buyer’s Guide documents
  - `4` Buy Canadian policy pages
  - `1` TBS directive page
- The largest Buyer’s Guide branches in the reference manifest are:
  - `plan`
  - `manage`
  - `approve`
  - `create-solicitation`
  - `receive-and-evaluate`
- This strongly supports a Buyer’s Guide-primary retrieval topology with
  supporting-source fallback instead of treating all sources as equal by
  default.

## 2026-03-23

### Evaluation / measurement findings

- Cohere's official RAG evaluation deep dive still lines up with the direction
  we need here:
  - retrieval should be measured with deterministic set-based metrics such as
    precision, recall, and rank-sensitive measures
  - generation should be evaluated at the claim level, separating grounding
    from answer-vs-reference overlap
  - the three most useful claim-level generation concepts remain:
    - faithfulness
    - correctness
    - coverage
  Source:
  - https://docs.cohere.com/page/rag-evaluation-deep-dive
- That same Cohere article explicitly recommends a strong evaluator and notes
  that using the same model family as both generator and evaluator can bias
  results. This strengthens the case for an eventually independent secondary
  judge provider, even though the repo can keep moving with Cohere-backed eval
  for now.
- Ragas is a good fit for a secondary measurement lane, but the installed
  version has an integration bug with Cohere:
  - `ragas.llms.llm_factory(..., provider=\"cohere\")` assumes a client shape
    compatible with `client.messages.create`
  - `cohere.ClientV2` does not expose that shape
  - the stock factory therefore fails for Cohere in the current environment
- A clean workaround does exist:
  - patch `cohere.ClientV2` directly with `instructor.from_cohere(...)`
  - wrap that patched client in `ragas.llms.base.InstructorLLM`
  - remove `top_p` from the default model args because Cohere V2 chat rejects it
- Live validation:
  - the patched wrapper produces real Ragas scores on both a toy sample and a
    repo-native smoke case
  - the stock Cohere `llm_factory` path does not
- Ragas should remain a secondary lane, not the primary source of truth:
  - keep deterministic retrieval metrics
  - keep the current structured judge
  - use Ragas as a cross-check on grounding, correctness, and coverage
- Cohere's retrieval-eval example using pairwise judging is a useful next
  measurement pattern for this repo:
  - compare control answer vs candidate answer directly
  - blind the order
  - let the judge pick A, B, or tie
  Source:
  - https://docs.cohere.com/page/retrieval-eval-pydantic-ai
- The clean implementation path here is not another custom judge stack.
  The repo can use:
  - the official OpenAI Python SDK
  - `responses.parse(...)` with a Pydantic verdict model
  - OpenAI prompt-cache fields on the request
  - local `diskcache` reuse for exact comparison replays
- Live validation with the repo's new pairwise lane:
  - baseline vs `mode_aware_planned_answering` on the full original `19`-case
    control surface produced:
    - control wins: `11`
    - candidate wins: `6`
    - ties: `2`
  - baseline vs `mode_aware_planned_answering` on the rebuilt full `39`-case
    surface produced:
    - control wins: `22`
    - candidate wins: `14`
    - ties: `3`
  - a second identical run hit the local cache on all `19` cases
  Interpretation:
  - pairwise judging currently agrees with the existing promotion decision that
    baseline should remain the formal control on the original `19`-case suite
  - more surprisingly, pairwise judging is also still favoring baseline on the
    rebuilt `39`-case surface, even where scalar required-claim recall favored
    the mode-aware branch
  - that means pairwise eval is measuring something meaningfully different from
    the current required-claim recall aggregate and should be treated as a
    serious cross-check, not a decorative extra
- The first explicit pairwise-vs-scalar disagreement analyses reinforce a
  consistent pattern:
  - scalar recall likes longer answers that mention more supported points
  - pairwise comparison often prefers shorter answers with the same recall
    because they are more direct and less embellished
  - answer-side experiments that try to compress after the fact can easily cut
    real workflow coverage if the audit model does not distinguish
    indispensable branches from optional background
  Practical implication:
  - answer-side work should focus on concise completeness at generation time,
    not only on post-hoc compression
  - pairwise A/B should remain part of the promotion contract because it is
    catching a real failure mode that scalar recall alone misses
- A compact mode-aware branch partially validated that interpretation:
  - its best full original-`19` run reached `0.9079`
  - its best rebuilt-`39` run reached `0.8218`
  - pairwise no longer clearly punished it for verbosity
  - pairwise-vs-scalar analysis showed:
    - candidate truncation-risk count `0`
    - control truncation-risk count `5` on the original `19`
    - candidate truncation-risk count `0`
    - control truncation-risk count `12` on the rebuilt `39`
  Interpretation:
  - concise-complete prompting is the right direction
  - but a compact planner can still fail by dropping indispensable workflow or
    locator points, so shorter is not enough by itself
- The next measurement lesson is that disagreement cases should themselves be a
  first-class eval surface.
  Practical implication:
  - maintain a canonical pairwise-precision slice built from the original
    control cases where baseline still wins despite equal or superficially
    improved scalar recall
  - use that slice to check:
    - faithfulness
    - correctness precision
    - directness / unnecessary detail
  - before trusting another answer-side branch on full-suite numbers
- The first real validation of that disagreement-heavy slice produced a useful
  mixed signal:
  - compact answer generation tied baseline on scalar recall
  - compact narrowly won pairwise A/B
  - Ragas on the same slice showed:
    - higher `faithfulness`
    - lower `correctness_precision`
    - lower `coverage_recall`
  Practical implication:
  - answer brevity/directness is no longer the only blocker
  - the next answer-side improvement should be about concise retention of
    indispensable claims, not just making answers shorter
- The repo-native Ragas integration also needed an operational correction:
  - the installed Ragas API defaults to `raise_exceptions=False`, but the repo
    wrapper had been forcing `True`
  - disagreement-heavy slices exposed that single metric timeouts can be common
    enough to matter
  Practical implication:
  - Ragas should run with explicit `RunConfig`
  - metric timeouts should be recorded as partial missing data rather than
    treated as full-run invalidation

## 2026-03-24

### Method research: answer-side omission and verbosity control

- For the current failure mode, the most relevant method family is not more
  post-hoc compression or generic repair. It is:
  - evidence-backed structured extraction
  - followed by deterministic or low-freedom rendering
  Sources:
  - Cohere structured JSON / parameter typing:
    - https://docs.cohere.com/v2/docs/parameter-types-in-json
  - Instructor Cohere integration:
    - https://python.useinstructor.com/integrations/cohere/
- Self-RAG is relevant here because it frames retrieval and generation as an
  adaptive process rather than one fixed path for every question.
  Source:
  - https://arxiv.org/abs/2310.11511
  Practical inference for this repo:
  - specialized answer methods should be activated selectively when the failure
    mode warrants them
  - broad application of a more constrained answer contract can over-structure
    cases that baseline already handles well
- The current repo evidence also supports a method-level distinction:
  - omission-heavy workflow and missing-detail questions benefit from explicit
    slot coverage
  - direct-rule and some navigation questions are more vulnerable to
    over-explanation when the same slot contract is applied broadly
  This is an inference from the repo's experiments, not a direct claim from the
  external sources above.
- The clean engineering consequence is:
  - typed extraction should use existing structured-output tools
  - routing should be justified by problem shape, not by piling on ad hoc
    prompt tweaks
  - if we add verification later, it should verify missing or unsupported
    slots/claims, not try to "improve" answers by adding adjacent supported
    detail
- The next stronger method family after failed pre-answer selectors is
  post-draft verification. Two sources are especially relevant:
  - Self-RAG:
    - https://arxiv.org/abs/2310.11511
  - Chain-of-Verification:
    - https://arxiv.org/abs/2309.11495
  Practical inference for this repo:
  - the verifier should inspect the actual baseline draft plus evidence
  - it should only decide whether to keep the draft or escalate to a stricter
    rewrite path
  - it should not generate a free-form list of "missing supported points,"
    because that is exactly where the earlier repair branch drifted into nearby
    supported-but-unneeded detail
- The first post-draft verifier results in this repo add one more method
  lesson:
  - a single opaque keep/rewrite verdict still underfires, even when it sees an
    independently extracted contract
  - a slot-coverage verifier is stronger because it asks the model to judge
    concrete obligations rather than invent a global route decision
  This is an inference from repo experiments, not a direct claim from the
  external papers above.
- Practical implication for the next answer-side step:
  - keep the typed structured-contract extraction
  - keep post-draft verification
  - prefer checklist or slot-level verification over another binary route
    selector
  - but constrain rewrite activation more narrowly than the current
    slot-coverage branch, because broad rewriting still harms the canonical dev
    surface

## 2026-03-25

### Claude Code collaboration setup

- Official Claude Code docs support a clean split between durable project
  context and machine-local runtime state:
  - project-scoped files such as `CLAUDE.md` and `.claude/settings.json` are
    meant to be shared with the repository
  - local scope is meant for personal or machine-specific state
  Source:
  - https://code.claude.com/docs/en/settings
- Official memory docs distinguish between:
  - `CLAUDE.md` for instructions loaded every session
  - auto memory for machine-local learned context
  Source:
  - https://code.claude.com/docs/en/memory
- Official subagent docs confirm:
  - subagents run in separate contexts
  - background subagents are appropriate for independent work
  - resumed sessions preserve prior conversation context
  Source:
  - https://code.claude.com/docs/en/sub-agents
- Practical implication for this repo:
  - keep shared Claude collaboration rules in committed `CLAUDE.md`
  - keep live Claude session IDs and transcripts in gitignored local storage
  - use a wrapper that resumes the same Claude Opus 4.6 session for repeated
    Codex <-> Claude consultations
  - keep Claude advisory by default and let repo-local eval evidence decide
    promotions

### Eval-method audit results

- A focused audit of the current eval harness identified a concrete methodological
  risk in structured LLM judging:
  - validating only the length of returned claim lists is insufficient
  - the harness should also verify that each returned claim identity still
    matches the authored claim it is being scored against
  Practical implication:
  - strict claim-text alignment should be enforced in normalization before
    recall or safety metrics are computed
- The same audit also reinforced a methodological limit of intervention-only
  composition:
  - it is only safe as a promotion surface if untouched candidate cases are
    visibly tracked for drift against control
  Practical implication:
  - conditional summaries should always expose whether non-selected candidate
    cases stayed baseline-equivalent or not
