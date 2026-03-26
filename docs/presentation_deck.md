# Presentation Deck

This is a slide-ready companion to `docs/presentation_pipeline_outline.md`.
Use it when building the actual deck. Each slide includes:

- what to put on the slide
- what to say out loud
- what evidence or visual to show

## Slide 1: Title and Thesis

**Title**
- PolicyRAGv.2
- Building a high-exactness procurement-policy RAG system

**On slide**
- Buyer’s Guide-first hybrid RAG
- Query decomposition in the baseline path
- Narrow exactness correction path on `main`
- Strong evaluation and promotion discipline

**Speaker notes**
- This system is designed for procurement-policy question answering, where
  exactness matters more than fluent paraphrase.
- The goal is not just to retrieve relevant pages. The goal is to answer
  correctly without inventing missing identifiers, forms, contacts, or workflow
  steps.
- The most important message from the project is that evaluation discipline and
  selective architectural changes mattered more than just adding complexity.

**Suggested visual**
- one pipeline diagram with the stages:
  - collect
  - normalize
  - chunk
  - retrieve
  - answer
  - evaluate

## Slide 2: Why This Problem Is Hard

**Title**
- Why procurement-policy QA is difficult

**On slide**
- Exactness matters
- Coverage and faithfulness both matter
- Missing-detail behavior matters

**Speaker notes**
- A good answer here has to do two things at once.
- First, it must cover the required rule, branch, exception, or deadline.
- Second, it must stay faithful to the evidence and avoid inventing exact
  details that are not present.
- This means the system can fail in two different ways:
  - low coverage: it leaves out important parts of the answer
  - low faithfulness: it adds unsupported detail
- That is why the evaluation stack tracks more than one metric.

**Suggested visual**
- two-column table:
  - coverage = did we include the needed points?
  - faithfulness = did we stay supported by the evidence?

## Slide 3: End-to-End Architecture

**Title**
- End-to-end system architecture

**On slide**
- Source collection and normalization
- Section-aware chunking
- Buyer’s Guide-first hybrid retrieval
- Inline evidence answer generation
- Exactness-specific post-draft correction
- Multi-lane evaluation

**Speaker notes**
- The backend is split into separate subsystems for collection,
  normalization, chunking, retrieval, answering, and evaluation.
- That separation made it much easier to run controlled experiments and reject
  plausible but weak ideas.
- The current canonical system on `main` is not just plain vector search.
- It is a Buyer’s Guide-first hybrid system with query decomposition and a
  narrow exactness-specific correction path.

**Suggested visual**
- architecture block diagram with arrows between the six stages

## Slide 4: Key Terms

**Title**
- Key terms used in the talk

**On slide**
- `canonical parity`
  - frozen benchmark family for tuning and promotion
- `parity19_dev`
  - 9-case development split used for tuning
- `parity19_holdout`
  - 10-case protected split used for promotion checks
- `p95` / `p99`
  - percentile statistics
- `sliding-window chunking`
  - fixed-size windows, usually with overlap
- `intervention-only evaluation`
  - candidate answer used only where the method actually fires

**Speaker notes**
- Define these once so the rest of the talk stays understandable.
- `canonical parity` mainly refers to `parity19_dev` and `parity19_holdout`.
- `parity19_dev` is the 9-case development split used while tuning methods.
- `parity19_holdout` is the 10-case protected split used when deciding what
  deserves promotion.
- `p95` means 95 percent of values are at or below that number.
- `p99` means 99 percent are at or below that number.
- `sliding-window chunking` is the classic fixed-window approach; our current
  system instead uses structure-aware section chunking.
- `intervention-only` matters for conditional methods because it removes noise
  from untouched baseline cases.

**Suggested visual**
- small glossary table with term and one-line meaning

## Slide 5: Source Design and Corpus Strategy

**Title**
- Source design: make the right source primary

**On slide**
- Buyer’s Guide is the primary retrieval surface
- Buy Canadian policy and TBS directive are supporting sources
- Corpus composition:
  - `228` Buyer’s Guide documents
  - `1` Buy Canadian policy page
  - `1` TBS directive page

**Speaker notes**
- This was an important early design decision.
- The corpus was overwhelmingly Buyer’s Guide-heavy, and most benchmarked
  questions depended first on Buyer’s Guide operational pages.
- So the default retrieval worldview became Buyer’s Guide-first instead of flat
  equal-source retrieval.
- We did try broader retrieval alternatives, but none earned promotion as a new
  default.

**Suggested visual**
- small bar chart with the three source groups

## Slide 6: Chunking

**Title**
- Chunking: preserve structure, but watch the long tail

**On slide**
- Canonical chunker: `section_chunker`
- Keeps heading path and structural metadata
- Chunk statistics:
  - median `129` chars
  - p95 `553`
  - p99 `3578`
  - max `34499`
  - largest packed chunk seen in broad runs `31630`
- Broad runs typically feed `16` packed chunks to the answer model

**Speaker notes**
- The current default is section-aware chunking, not naive fixed windows.
- That was chosen because policy questions often depend on section structure and
  heading lineage.
- Most chunks are small, but there is a real long tail of very large sections.
- The largest chunk in the corpus is `34499` characters, and the largest packed
  chunk we actually saw in broad runs was about `31630`.
- We tested sliding-window chunking as an upstream alternative, but it did not
  help on the canonical dev surface.
- In fact, on `parity19_dev`, scalar recall regressed from `0.8611` to
  `0.8056`, and pairwise was only `4-4-1`.
- So chunking clearly matters, but broad chunker replacement did not earn
  promotion.

**Suggested visual**
- histogram of chunk sizes with the long-tail called out

## Slide 7: What One Chunk Looks Like

**Title**
- What one chunk looks like

**On slide**
- Representative Buyer’s Guide chunk, showing:
  - `chunk_id`
  - `doc_id`
  - `source_family`
  - `chunker_name`
  - `chunk_type`
  - `heading`
  - `heading_path`
  - `section_id`
  - `order`
  - `token_estimate`
  - one metadata field such as `scope_tags`
  - truncated `text` with `...`

**Speaker notes**
- This is useful because “16 chunks” is abstract unless the audience sees the
  structure of one of them.
- The important point is that a chunk is not just free text.
- It carries retrieval and interpretation metadata:
  - what document it came from
  - where it sat in the page structure
  - what heading lineage it belongs to
  - what its local order was
- The answer model typically sees a packed bundle of 16 chunks like this, not
  whole pages.
- The text itself can be shortened with `...`; the structure is the main point.

**Suggested visual**
- simplified pseudo-record or table showing the fields and one truncated text
  excerpt

## Slide 8: Retrieval

**Title**
- Retrieval: hybrid plus query decomposition

**On slide**
- Cohere dense embeddings
- Elasticsearch lexical retrieval
- Hybrid blending
- Buyer’s Guide-first topology
- Query decomposition in the baseline path

**Speaker notes**
- Retrieval on `main` combines dense and lexical search rather than choosing
  one.
- The strongest promoted retrieval improvement was query decomposition.
- On canonical `parity19_dev`, required-claim recall moved from `0.8333` to
  `0.8889`.
- On canonical `parity19_holdout`, it moved from `0.6000` to `0.8250`.
- Deterministic packed claim-evidence recall also improved on both surfaces:
  `0.9722 -> 1.0` on dev and `0.8250 -> 0.9750` on holdout.
- The broad retrieval takeaway is that the system often does retrieve the right
  evidence.

**Suggested visual**
- before/after bar chart for baseline vs query decomposition on dev and holdout

## Slide 9: Evidence Presentation

**Title**
- Evidence presentation: a real issue, but not solved by every prompt change

**On slide**
- Canonical answer path: `inline_evidence_chat`
- Concern: maybe the model is drowned by large evidence bundles
- Tested:
  - `query_guided_answering`
  - `structured_answering`
  - `selective_structured_answering`

**Speaker notes**
- We suspected that even with good retrieval, the answer model might struggle if
  the evidence bundle was too noisy.
- So we ran same-retrieval presentation experiments.
- Some of them improved scalar recall, but that was not enough.
- `query_guided_answering` is a good example:
  - dev scalar improved from `0.8611` to `0.8889`
  - holdout scalar improved from `0.7583` to `0.8750`
  - but pairwise still preferred the control on both surfaces:
    - dev `5-4-0`
    - holdout `6-4-0`
- So evidence presentation is a real subproblem, but the tested broad
  structured-presentation methods were not promotable.

**Suggested visual**
- one chart for scalar, one chart for pairwise, to show why scalar alone was
  not enough

## Slide 10: Answer Generation

**Title**
- Answer generation: some gains were real, but not broad enough

**On slide**
- Baseline answer path remained the most stable broad option
- Tried:
  - planned answering
  - mode-aware answering
  - compact variants
  - verifier-gated structured branches

**Speaker notes**
- We did find answer-side changes that helped.
- The most important example is the broad mode-aware branch.
- On the rebuilt `39`, required-claim recall improved from `0.7415` to
  `0.8047`, and forbidden violations improved from `1` to `0`.
- But on the original `19` full benchmark, the same branch regressed from
  `0.8684` to `0.8158`.
- Pairwise also stayed against broad promotion:
  - original `19`: control `11`, candidate `6`, ties `2`
  - rebuilt `39`: control `22`, candidate `14`, ties `3`
- That is why the broad default remains conservative.

**Suggested visual**
- two-panel chart:
  - rebuilt `39` improvement
  - original `19` regression

## Slide 11: Exactness / Abstention Subpath

**Title**
- The narrow exactness path: the clearest method win on `main`

**On slide**
- Post-draft correction for missing-detail cases
- Goal: do not invent exact identifiers, forms, or contacts
- Narrow and selectively activated

**Speaker notes**
- This is the main non-baseline method on `main`.
- It targets a very specific failure family: cases where the system knows the
  nearby process, but the exact requested identifier or contact is not in the
  evidence.
- On the exactness-family holdout intervention-only surface, this path improved
  recall from `0.7778` to `0.8889`, reduced forbidden violations from `1` to
  `0`, and improved abstain accuracy from `0.6667` to `1.0`.
- Pairwise on the same surface was also clean:
  - candidate `2`
  - control `0`
  - ties `1`
- This is the kind of change we were willing to merge: narrow, measurable, and
  validated on the right surface.

**Suggested visual**
- small table with before/after for recall, forbidden, abstain accuracy, pairwise

## Slide 12: Evaluation and Promotion Discipline

**Title**
- Evaluation: why we trust some results and reject others

**On slide**
- Primary lane:
  - deterministic retrieval metrics
  - structured Cohere judge
  - required-claim recall
  - forbidden-claim violations
  - abstention accuracy
- Secondary lanes:
  - Ragas
  - OpenAI pairwise A/B
- Conditional methods:
  - intervention-only composite evaluation

**Speaker notes**
- One of the strongest parts of the project is the evaluation discipline.
- A single scalar score was not enough, because a branch could improve coverage
  while hurting faithfulness, or look good only because of conditional-method
  noise.
- So the project added:
  - dev/holdout split discipline
  - pairwise A/B
  - exactness-family diagnostics
  - intervention-only evaluation for conditional methods
- The exactness path is a good example of why that matters: it looks strong on
  the intervention-only surface, which is the right place to judge it.

**Suggested visual**
- evaluation stack diagram showing primary and secondary lanes

## Slide 13: What Failed and What We Learned

**Title**
- Failures that changed the architecture

**On slide**
- Broad mode-aware branch:
  - rebuilt `39`: `0.7415 -> 0.8047`
  - original `19`: `0.8684 -> 0.8158`
- Page-rerank / document-seed retrieval:
  - hard cluster `0.2708 -> 0.7708`
  - canonical holdout down to `0.6167`
- Sliding-window chunking:
  - `0.8611 -> 0.8056`

**Speaker notes**
- These failures were useful.
- They showed that local wins are not broad architecture wins.
- The broad mode-aware branch looked very promising on one surface, but it did
  not generalize well enough.
- Page-rerank retrieval helped a hard cluster dramatically, but did not survive
  canonical promotion.
- Sliding-window chunking was a good reminder that not every plausible chunking
  change is testing the intended hypothesis.
- These failures are why the current `main` is narrower and more disciplined
  than the most ambitious experimental branches.

**Suggested visual**
- one table with experiment, local gain, broad outcome, final decision

## Slide 14: Current Status and Next Steps

**Title**
- Current status

**On slide**
- Best original `19` full: `0.8684`
- Best original `19` dev: `0.8889`
- Best original `19` holdout: `0.8250`
- Rebuilt `39` baseline: `0.7415`
- Best rebuilt `39` broad experiment: `0.8047`, not promoted
- Test suite: `154 passed`
- Current readiness:
  - supervised internal assistant: yes
  - fully trusted standalone authority: no

**Speaker notes**
- The system is already usable with human oversight.
- It is not yet ready to be treated as a fully trusted standalone authority.
- The next steps are not broad retrieval rewrites.
- The next steps are:
  - keep baseline retrieval fixed
  - continue on the strongest answer-side lane
  - keep using intervention-only evaluation for conditional methods
  - author and protect a final blind set

**Suggested visual**
- status slide with three sections:
  - what is on `main`
  - what is still experimental
  - what comes next

## Optional Slide 15: Demo Plan

**Title**
- Demo plan

**On slide**
- 1 successful broad question
- 1 exactness / abstention case
- 1 honest limitation

**Speaker notes**
- If you demo the system, do not only show a successful easy case.
- Show one broad workflow question, one exactness-sensitive case, and one case
  that illustrates a current limitation.
- That makes the presentation more credible and better aligned with the actual
  state of the system.

## Optional Appendix: Likely Questions

**Question**
- Why not just use a bigger model?

**Answer**
- Bigger models help, but this project repeatedly found that retrieval
  discipline, evidence presentation, and evaluation methodology mattered more
  than just scaling the generator.

**Question**
- Why did you keep the exactness path but reject broader answer branches?

**Answer**
- Because the exactness path won on the correct targeted surface and stayed
  narrow, while broader branches did not generalize cleanly enough.

**Question**
- What is the main unsolved problem?

**Answer**
- Answer precision and selective activation, not obviously raw retrieval power.

## Optional Appendix Slide: Query Decomposition Example

**Title**
- Worked example: query decomposition improved retrieval

**On slide**
- Case: `HR_008`
- Prompt:
  - "What is the actual contractual difference between a standing offer and a supply arrangement? When does Canada become legally bound under each one?"
- Baseline retrieval:
  - packed mostly standing-offer evidence
  - score `0.3333`
- With query decomposition:
  - added targeted retrieval queries
  - pulled a comparative chunk covering supply arrangement obligations
  - score `1.0`

**Speaker notes**
- This is a good retrieval example because the issue was not generic relevance.
- The baseline already had standing-offer material, but it did not have the
  comparative chunk needed to answer the supply-arrangement part cleanly.
- Query decomposition added targeted subqueries, and one of the newly packed
  chunks explicitly stated:
  - there is no contractual obligation at the supply arrangement stage
  - each contract awarded under the supply arrangement is a separate binding contract

## Optional Appendix Slide: Exactness Example

**Title**
- Worked example: exactness path prevented an unsupported answer

**On slide**
- Case: `HR_037`
- Prompt:
  - "What exact form number or file name do I need for the specimen signature card template in the GC network folder when approvals go beyond delegated authority?"
- Evidence says:
  - the Schedule 3 specimen signature card template exists
  - it is in the Delegation of Authorities folder
  - but it does not provide an exact file name or form number
- Baseline:
  - answered with a specific exact name
  - recall `0.3333`, forbidden `1`
- Exactness path:
  - explicitly said the exact file name is not provided
  - recall `0.6667`, forbidden `0`

**Speaker notes**
- This is the clearest slide for why the narrow exactness path exists.
- The evidence supports the nearby process and the folder location.
- It does not support the exact requested identifier.
- The baseline overcommitted. The exactness path did not.

## Optional Appendix Slide: Broad Answer Branch Rejection Example

**Title**
- Worked example: why broad mode-aware answering was rejected

**On slide**
- Case: `HR_001`
- Prompt:
  - "We received a bid after the deadline and the supplier says they sent it on time. What proof counts, and what are we supposed to do with the offer?"
- Query-decomposition baseline:
  - recall `1.0`
- Mode-aware branch:
  - recall `0.5`
- Judge finding:
  - mode-aware answer dropped support for offeror responsibility and return handling

**Speaker notes**
- This is the simplest case-level explanation for why a broad answer branch that
  helped rebuilt-39 was still rejected.
- It still looked strong at a high level, but on important canonical cases it
  lost required handling detail.
- That is exactly the kind of regression we did not allow onto `main`.
