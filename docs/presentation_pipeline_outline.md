# Presentation Outline: Stage-by-Stage RAG Pipeline

This outline is designed for a presentation. It explains the system by walking
through the RAG pipeline from source collection to evaluation, and it includes
the major experiments that failed and why they were rejected.

Use this with:

- `docs/presentation_snapshot.md`
- `docs/presentation_deck.md`
- `docs/presentation_beamer.tex`

## Suggested talk structure

1. Problem and constraints
2. Pipeline overview
3. Stage-by-stage architecture
4. What we tried and what failed
5. Current quality and readiness
6. Next steps

## Slide 1: Problem

Main message:
- procurement and policy QA is a high-exactness RAG problem
- the system must not just retrieve relevant pages
- it must avoid inventing missing identifiers, forms, contacts, or workflow steps

Useful points:
- exactness matters more than fluent paraphrase
- partial retrieval is not enough if the answer overstates the rule
- evaluation must check both coverage and faithfulness

## Slide 2: Pipeline Overview

Show the pipeline as:

1. collect source pages
2. normalize and enrich documents
3. chunk them
4. build hybrid retrieval index
5. retrieve and pack evidence
6. generate the answer
7. evaluate with multiple measurement lanes

Current one-line summary:
- Buyer’s Guide-first hybrid RAG with query decomposition, strict evaluation,
  and a narrow exactness-specific post-draft correction path

## Slide 2.5: Key Terms

Terms to define explicitly:
- `canonical parity`
  - the frozen benchmark family used for tuning and promotion checks
  - in practice this mainly means `parity19_dev` and `parity19_holdout`
- `parity19_dev`
  - the 9-case development split used during tuning
- `parity19_holdout`
  - the 10-case protected split used during promotion checks
- `p95` / `p99`
  - percentile statistics
  - example: `p99` chunk size means `99%` of chunks are at or below that length
- `sliding-window chunking`
  - fixed-size text windows, usually with overlap
  - contrasted with our current structure-aware section chunking
- `intervention-only composite`
  - for conditional methods, baseline stays untouched unless the method actually fires

## Slide 3: Source Collection and Corpus Design

What we did:
- built a clean, modular backend with separate collection, normalization,
  chunking, retrieval, answering, and evaluation subsystems
- made the Buyer’s Guide the primary retrieval surface
- treated Buy Canadian policy and the TBS directive as supporting sources
- kept collection, normalization, chunking, retrieval, answering, and
  evaluation as separate subsystems

Why:
- the reference corpus was overwhelmingly Buyer’s Guide-heavy
- most benchmarked questions depended first on Buyer’s Guide operational pages
- flat equal-source retrieval was not the best prior

Evidence to cite:
- reference corpus composition:
  - `228` Buyer’s Guide documents
  - `1` Buy Canadian policy page
  - `1` TBS directive page

What failed or was rejected:
- flat source treatment as the default worldview
- broad unified-source retrieval as a new default

Why it failed:
- it did not produce a clean broad gain
- it diluted the stronger Buyer’s Guide-first prior

Evidence to cite:
- the topology decision was based on corpus composition first, then later broad
  retrieval tests failed to show a trustworthy enough gain to justify replacing
  the Buyer’s Guide-first default

## Slide 4: Chunking

Current canonical choice:
- `section_chunker`
- variable-size, section-aware chunks
- keeps heading path and structural metadata

Why this was chosen:
- policy questions often depend on section structure
- heading path and lineage are useful retrieval signals
- it preserves page-local semantics better than naive fixed windows

Important real stats to mention:
- corpus median chunk size: about `129` chars
- p95: about `553`
- p99: about `3578`
- max: about `34499`
- largest packed chunk seen in broad runs: about `31630`
- broad runs typically feed `16` packed chunks to the answer model

What we tested:
- sliding-window chunking as an upstream alternative
- broad chunk-size diagnostics

What failed:
- the current `sliding_window_chunker` did not help
- on this corpus it was actually coarser than the section baseline
- canonical `parity19_dev` regressed:
  - baseline `0.8611`
  - sliding-window `0.8056`
- pairwise on the same dev surface was only:
  - control `4`
  - candidate `4`
  - ties `1`

Why it failed:
- it was not really testing the intended “smaller chunk” hypothesis
- broad chunk-size correlations did not explain the answer gap cleanly

Good lesson:
- chunking is important, but broad chunking rewrites were not the highest-ROI
  move at the current stage

## Slide 4.5: What One Chunk Looks Like

Add one concrete example so the audience knows what “16 chunks” actually means.

Use a simplified pseudo-record based on a real Buyer’s Guide chunk:
- `chunk_id: 9e57865c383b__section__185`
- `doc_id: 9e57865c383b`
- `source_family: buyers_guide`
- `chunker_name: section_chunker`
- `chunk_type: paragraph`
- `heading: Using a standing offer`
- `heading_path: Choose the method of supply -> Using a standing offer`
- `section_id: block_0185`
- `order: 185`
- `token_estimate: 121`
- `scope_tags: standing_offer`
- `text: A standing offer is from an offeror to Canada ... A standing offer is not a contract ...`

Main message:
- a chunk is not just a text span
- it carries structure and metadata that retrieval and interpretation can use
- in broad runs the answer model usually sees a packed bundle of `16` chunks
  like this, not whole pages

## Slide 5: Indexing and Retrieval

Current canonical retrieval:
- Cohere dense embeddings
- Elasticsearch lexical retrieval
- hybrid blending
- Buyer’s Guide-first source topology
- LLM query decomposition in the baseline retrieval path

Why:
- dense retrieval alone was not enough
- lexical retrieval alone was not enough
- query decomposition improved both judged quality and deterministic retrieval
  recall on canonical surfaces

Evidence to cite:
- canonical `parity19_dev`:
  - baseline `0.8333`
  - query decomposition `0.8889`
  - packed claim-evidence recall `0.9722 -> 1.0`
- canonical `parity19_holdout`:
  - baseline `0.6000`
  - query decomposition `0.8250`
  - packed claim-evidence recall `0.8250 -> 0.9750`

Current promoted retrieval result:
- original `19` best full run:
  - answer recall `0.8684`
  - packed claim-evidence recall `0.9868`

What this means:
- the system often retrieves the needed evidence
- the remaining gap is often in answer use of that evidence

Retrieval experiments that failed or stayed unpromoted:
- `diverse_packing`
- broad unified-source retrieval
- page/document seed retrieval families as broad replacements
- selective localized page-rerank retrieval as a broad replacement

Why they failed:
- some helped specific hard clusters
- but they regressed the canonical `19` dev/holdout surfaces
- they were not trustworthy broad promotions

Good lesson:
- retrieval can be improved locally without becoming the new default
- hard-cluster wins are not enough for promotion

## Slide 6: Evidence Packing and Presentation

Current canonical evidence presentation:
- retrieved chunks are packed and passed inline to the model
- default answer path is `inline_evidence_chat`

Why:
- it worked better than the repo’s current `documents_chat` path under the
  current chunk shapes
- it kept the prompt and citation logic simple

What we worried about:
- maybe the answer model is being drowned by large evidence bundles
- maybe `16` section chunks is too much

What we measured:
- prompt size
- chunk rank
- claim coverage rank
- evidence-presentation signal against answer gaps

What we found:
- broad prompt-size and oversize-chunk metrics did not explain the gap well
- many failure cases already had relevant evidence early in the bundle

What we tried:
- `query_guided_answering`
- `structured_answering`
- `selective_structured_answering`

What failed:
- `query_guided_answering`
- `selective_structured_answering`

Why they failed:
- `query_guided_answering` improved scalar recall but lost pairwise on both
  canonical dev and holdout
- `selective_structured_answering` had a sane final gate, but failed the
  correct holdout intervention-only evaluation surface

Evidence to cite:
- `query_guided_answering`
  - dev scalar:
    - baseline `0.8611`
    - query-guided `0.8889`
  - dev pairwise:
    - control `5`
    - candidate `4`
    - ties `0`
  - holdout scalar:
    - baseline `0.7583`
    - query-guided `0.8750`
  - holdout pairwise:
    - control `6`
    - candidate `4`
    - ties `0`
- `selective_structured_answering`
  - dev intervention-only:
    - `0.7500 -> 0.7778`
    - pairwise `1` control / `2` candidate / `6` ties
  - holdout intervention-only:
    - `0.7583 -> 0.7083`
    - pairwise `2` control / `2` candidate / `6` ties

Important methodology lesson:
- conditional answer branches must be measured with intervention-only
  composites, not only raw profile runs

## Slide 7: Answer Generation

Current canonical answer path:
- `inline_evidence_chat`

Why it remains the default:
- it is the most stable broad baseline
- many heavier answer-side branches improved one surface and regressed another

Broad answer-side experiments tried:
- `structured_inline_evidence_chat`
- `planned_inline_evidence_chat`
- `mode_aware_planned_answering`
- `selective_mode_aware_planned_answering`
- compact variants
- verifier-gated structured-contract variants

What partially worked:
- mode-aware and compact answer families improved some rebuilt-39 results
- verifier-gated and exactness-specific paths helped narrow failure families

Evidence to cite:
- broad mode-aware branch on rebuilt `39`:
  - baseline `0.7415`
  - mode-aware `0.8047`
  - failures `0 -> 0`
  - forbidden violations `1 -> 0`
- but original `19` full regressed:
  - baseline `0.8684`
  - mode-aware `0.8158`
- pairwise also stayed against broad promotion:
  - original `19`: control `11`, candidate `6`, ties `2`
  - rebuilt `39`: control `22`, candidate `14`, ties `3`

Why these are still not fully promoted:
- some improved rebuilt `39` but regressed original `19`
- some looked good on scalar recall but lost pairwise
- some only worked when judged intervention-only

Good lesson:
- answer-side improvements are real
- but broad promotion requires stronger selective activation than we currently
  have

## Slide 8: Exactness / Abstention Subpath

This is the main non-baseline addition on `main`.

What it does:
- handles missing-detail cases where the system must avoid inventing the exact
  requested identifier or contact detail
- runs as a narrow post-draft correction path

Why it matters:
- this is a common failure mode in policy systems
- the model may know the nearby process but not the exact form or contact

What worked:
- exactness-family holdout intervention-only result:
  - recall `0.7778 -> 0.8889`
  - forbidden violations `1 -> 0`
  - abstain accuracy `0.6667 -> 1.0`
- pairwise on the same holdout intervention-only surface:
  - candidate wins `2`
  - control wins `0`
  - ties `1`

Why this is promotable while other answer branches were not:
- it is narrow
- it is measurable on the right surface
- it solves a specific real failure family instead of trying to replace the
  whole answer system

## Slide 9: Evaluation and Promotion Methodology

This is one of the strongest parts of the project.

Primary eval lane:
- deterministic retrieval metrics
- structured Cohere judge
- required-claim recall
- forbidden-claim violations
- abstention accuracy

Secondary lanes:
- Ragas
- OpenAI pairwise A/B

Important promotion rules:
- use canonical `parity19_dev` for tuning
- use canonical `parity19_holdout` for promotion checks
- use rebuilt `39` as a broader architecture surface
- use intervention-only composites for conditional methods

Big methodological improvements over time:
- dev/holdout split discipline
- pairwise A/B added
- exactness-family diagnostic set added
- intervention-only evaluation added

## Slide 10: Main Failures and What They Taught Us

This slide should be explicit. It makes the project look more serious, not
less.

Failure 1:
- broad mode-aware answer branch improved rebuilt `39`
- but regressed original `19`
- lesson:
  - one benchmark win is not enough
  - broad answer routing can overfit one surface
- evidence:
  - rebuilt `39`: `0.7415 -> 0.8047`
  - original `19`: `0.8684 -> 0.8158`

Failure 2:
- page-rerank and document-seed retrieval helped a hard cluster
- but did not survive canonical promotion
- lesson:
  - local retrieval wins are not broad architecture wins
- evidence:
  - hard cluster baseline: `0.2708`
  - document-rerank seed: `0.7708`
  - localized document-rerank seed: `0.7708`
  - but canonical holdout fell to `0.6167`

Failure 3:
- broad structured presentation looked promising
- but failed holdout intervention-only evaluation
- lesson:
  - the selector and the method must be judged separately
  - raw full-profile results are too noisy for conditional branches
- evidence:
  - dev intervention-only: `0.7500 -> 0.7778`
  - holdout intervention-only: `0.7583 -> 0.7083`

Failure 4:
- sliding-window chunking did not help
- lesson:
  - not every chunking change is actually testing the intended hypothesis
- evidence:
  - canonical dev: `0.8611 -> 0.8056`
  - pairwise: `4-4-1`

## Slide 11: Current System Quality

Useful numeric summary:
- best original `19` full:
  - `0.8684`
- best original `19` dev:
  - `0.8889`
- best original `19` holdout:
  - `0.8250`
- rebuilt `39` baseline:
  - `0.7415`
- best rebuilt `39` broad experimental:
  - `0.8047`
  - not promoted
- tests:
  - `154 passed`

Plain-language conclusion:
- usable as a supervised internal assistant
- not yet strong enough to be a fully trusted standalone authority

## Slide 12: What is on `main` vs Experimental

On `main`:
- Buyer’s Guide-first hybrid retrieval
- query decomposition
- strict runtime
- exactness-specific narrow correction path
- stronger evaluation stack

Experimental only:
- broad mode-aware routing
- verifier-gated broad answer branches
- page-rerank retrieval families
- broad structured-presentation branches

## Slide 13: Best Next Steps

- stop chasing broad retrieval rewrites for now
- keep baseline retrieval fixed
- return to the strongest remaining answer-side lane
- use intervention-only evaluation for conditional branches
- author and protect the final blind set

## Optional appendix: likely audience questions

Question:
- Why not just use a bigger model?
Answer:
- bigger models help, but this project found that architecture and evaluation
  discipline matter more than just model size

Question:
- Why did you keep some narrow path but reject broader answer branches?
Answer:
- because the narrow exactness path won on the correct targeted surface, while
  broader branches did not generalize cleanly

Question:
- Why are there so many metrics?
Answer:
- because one scalar score can miss faithfulness, abstention behavior, or
  conditional-method noise

Question:
- What is the main unsolved problem?
Answer:
- answer precision and selective activation, not obviously raw retrieval power
