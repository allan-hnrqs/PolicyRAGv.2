# Clean Buyer’s Guide RAG Rebuild Plan

## Summary

Build a new backend-only repo at `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean` as a clean-room reimplementation inspired by `feat-retrieval-expensive-methods-eval`, not copied from it and not based on `main`.

The purpose of the system is **Buyer’s Guide-first RAG**. Buy Canadian and the TBS directive are supporting sources, not equal peers by default. Phase 1 must match or beat a frozen `feat` parity profile on both the original `19`-case suite and a newly rebuilt `39`-case suite, while keeping the architecture clean, modular, handoff-friendly, and easy to experiment with.

Delete `autonomy_lab_goal_v1` as the first cleanup step and do not reuse its code or eval cases as trusted implementation inputs. Any useful idea from that lane must be recreated cleanly.

## Key Architecture

### 1. Repo shape and public interface

Create a new repo with:
- `src/bgrag/`
- `tests/`
- `profiles/`
- `datasets/`
- `docs/`
- `scripts/`

Ship a Python package plus CLI only in phase 1. No HTTP API yet.

The CLI surface is fixed:
- `bgrag collect`
- `bgrag build-corpus`
- `bgrag build-index`
- `bgrag query`
- `bgrag eval`
- `bgrag freeze-baseline`
- `bgrag inspect`

Use a thin custom core with strong library-first bias. Default libraries:
- `cohere`
- `elasticsearch`
- `httpx`
- `beautifulsoup4`
- `pydantic` + `pydantic-settings`
- `typer`
- `rich`
- `tenacity`
- `pytest`

Rule: prefer a library unless a custom implementation is clearly simpler, clearer, and easier to hand off.

### 2. Core subsystem boundaries

Implement explicit typed subsystems:
- `collect`
- `normalize`
- `chunking`
- `metadata`
- `indexing`
- `retrieval`
- `answering`
- `eval`
- `profiles`

Core public types:
- `SourceDocument`
- `NormalizedDocument`
- `ChunkRecord`
- `RetrievalCandidate`
- `EvidenceBundle`
- `AnswerResult`
- `EvalCase`
- `EvalRunResult`
- `SourceFamily`

The source families are explicit and first-class:
- `buyers_guide`
- `buy_canadian_policy`
- `tbs_directive`

Use registry-based profile selection, not scattered conditionals. Profiles must be able to swap:
- collector strategies
- chunkers
- metadata enrichers
- source-topology retrieval policies
- retrievers/rerankers
- answer strategies
- evaluator profiles

### 3. Buyer’s Guide-first source topology

Do **not** throw all sources into one undifferentiated pool by default.

Phase 1 should index sources separately by family and support two retrieval profiles:
- `bg_primary_support_fallback`
- `unified_source_hybrid`

Default phase-1 retrieval profile is `bg_primary_support_fallback`.

Implementation shape:
- one logical corpus model
- separate per-family index partitions or indices
- a retrieval orchestrator that queries Buyer’s Guide first
- supporting-source retrieval only when triggered by evidence, source links, or explicit policy need

This keeps the system aligned with the actual product purpose while still allowing unified-source experiments.

The default fallback logic should be explicit:
- answer from Buyer’s Guide alone when sufficient
- retrieve supporting policy/directive evidence only when:
  - Buyer’s Guide chunks cite or imply external policy authority
  - the query explicitly asks about policy/directive interpretation
  - Buyer’s Guide evidence is incomplete for the required claim set

### 4. Collector and corpus rebuild

Phase 1 includes a clean rebuilt collector.

Rebuild from scratch around the same seed scope as `feat`:
- Buyer’s Guide
- Buy Canadian page(s)
- TBS directive page(s)

Collector stages must be separate and testable:
- fetch
- canonicalize
- classify source family
- extract structure
- build graph/lineage metadata
- persist normalized corpus package

The normalized corpus must preserve:
- canonical URL
- source family
- authority rank
- breadcrumb path
- parent/child/lineage graph links
- structural blocks/sections
- provenance paths
- content hash

Chunking must be explicitly swappable. Initial chunkers:
- `section_chunker`
- `block_chunker`
- `sliding_window_chunker`

Metadata must also be swappable. Initial enrichers:
- `authority_metadata`
- `lineage_metadata`
- `scope_tag_metadata`
- `source_topology_metadata`

The collector output must support trying different chunking and metadata strategies without recollecting source pages.

### 5. Retrieval and answer baseline

The phase-1 baseline is Cohere-first and mirrors the successful `feat` shape at the system level:
- Cohere generation
- Cohere rerank
- Elasticsearch lexical retrieval
- hybrid retrieval
- explicit packed evidence before answer generation

Do not bring over feat’s ad hoc keyword-routing answer logic as default architecture.

The default answer path should be:
- retrieve
- assemble `EvidenceBundle`
- generate with a pluggable answer strategy

The initial answer strategies are:
- `documents_chat`
- `inline_evidence_chat`

The default phase-1 answer strategy should be `inline_evidence_chat`, because the audited lesson from the previous system is that this is the strongest clean architectural fix for Command A-style failures.

### 6. Knowledge graph lane

KG is an experiment lane only, not a phase-1 baseline dependency.

Design for KG from day one by:
- preserving stable document/chunk IDs
- preserving graph-ready lineage and semantic links
- defining an experimental retriever interface that can later accept a graph backend

Do not make KG part of parity scope.

Phase 2 should add one optional KG prototype, likely via Neo4j’s official GraphRAG tooling, but only after baseline parity is reached.

### 7. Research as a first-class workstream

Add a dedicated research lane to the repo process, not as an afterthought.

Create:
- `docs/research-log.md`
- `docs/adr/`
- `docs/architecture-survey/`

Research work must explicitly survey and reuse strong existing ideas before inventing custom logic. Required research themes:
- source-aware retrieval and routing
- hierarchical/recursive retrieval
- chunking strategies and metadata-rich chunking
- graph-ready RAG / GraphRAG patterns
- citation/provenance strategies
- evaluation governance and anti-overfitting practice
- Cohere-specific RAG patterns

Use primary or official sources where possible. When a useful resource requires a sign-up, manual download, or inaccessible paper/article, stop and surface a concrete manual task for the user instead of silently substituting a weaker source.

## Evaluation and Parity Contract

### 1. Freeze the benchmark honestly

Before rebuilding parity, freeze a `feat` reference profile using the current `feat-retrieval-expensive-methods-eval` repo.

The frozen parity contract includes:
- original annotated `19`-case suite
- newly rebuilt `39`-case suite
- a smaller true unseen holdout
- a final blind acceptance set that remains untouched until we believe phase 1 is complete

Do **not** trust the autonomy-lab-expanded case bank as frozen truth. Rebuild the additional `20` cases independently in the new repo with fresh case authoring and source-grounded claim annotations.

The parity baseline runtime profile is fixed:
- generator: `command-a-03-2025`
- judge: `command-a-03-2025`
- rerank: `rerank-v4.0-fast`
- embeddings: `embed-english-v3.0`
- `top_k=16`
- `chat_temperature=0.0`

Run the frozen `feat` baseline twice per parity suite and record the **lower** judged score as the parity floor.

### 2. Evaluation bank as a major deliverable

Treat evaluation-bank quality as a co-equal phase-1 deliverable.

Organize:
- `datasets/eval/dev/`
- `datasets/eval/holdout/`
- `datasets/eval/final/`
- `datasets/eval/parity/`
- `datasets/eval/generated/`
- `datasets/eval/manifests/`

Each case must contain:
- ID
- source URLs
- source family expectations
- question
- reference answer
- required claims
- forbidden claims where relevant
- claim evidence
- abstain expectation where relevant
- tags for family, difficulty, and failure mode

The rebuilt `39`-case suite must intentionally cover:
- operational workflow questions
- comparison questions
- abstention/detail questions
- navigation questions
- cross-page synthesis
- authority/policy hierarchy
- source-boundary questions
- Buyer’s Guide-primary vs policy-support questions

Holdout policy:
- once marked holdout, never use for targeted tuning
- every accepted architecture change must be tested on holdout before promotion
- add new challenge cases after accepted changes to prevent overfitting

Final-set policy:
- the final set is a blind acceptance gate, not a tuning surface
- do not inspect final-case results during architecture search except at explicit milestone reviews
- do not tune directly on final-set failures
- if the final set exposes weaknesses, add fresh sibling cases to dev/holdout instead of reopening the same final items for iterative tuning

### 3. Phase-1 acceptance criteria

Phase 1 is complete only if the rebuild:
- matches or beats the frozen `feat` parity floor on the `19`-case suite
- matches or beats the frozen `feat` parity floor on the rebuilt `39`-case suite
- has zero answer-generation failures on both suites
- does not increase forbidden-claim violations
- preserves Buyer’s Guide-primary behavior in source-boundary tests
- produces a codebase and docs package that another team can run and understand without tribal knowledge

## Test Plan

Unit tests must cover:
- URL normalization and canonicalization
- source-family classification
- authority ranking
- graph/lineage extraction
- chunker behavior on structured markdown fixtures
- metadata-enricher composition
- source-topology retrieval policies
- hybrid retrieval score composition
- answer-strategy selection
- profile loading and validation
- eval-case schema validation

Integration tests must cover:
- collect → normalize → chunk → index on deterministic fixture pages
- Buyer’s Guide-only retrieval
- Buyer’s Guide + support fallback retrieval
- unified-source retrieval profile
- retrieve → answer on representative cases from each failure family
- parity eval runner on smoke subsets

Regression suites must cover:
- original `19`-case parity suite
- rebuilt `39`-case parity suite
- true unseen holdout
- final blind acceptance set
- explicit source-boundary cases where supporting policy should and should not be pulled in
- chunking-profile A/B checks
- metadata-profile A/B checks

Documentation/handoff tests must cover:
- a fresh engineer can run `collect`, `build-corpus`, `build-index`, `query`, and `eval` from the README only
- each subsystem has a short module README
- ADRs explain the major architectural choices:
  - thin custom core over heavy framework orchestration
  - Buyer’s Guide-primary topology
  - phase-1 inline-evidence answer strategy
  - KG deferred to experiment lane

## Assumptions and Defaults

- Delete `autonomy_lab_goal_v1` entirely and do not preserve code or eval assets from it.
- Use `feat-retrieval-expensive-methods-eval` as the benchmark and behavior reference only; do not copy implementation code from it.
- Build phase 1 as backend-only with Python package + CLI.
- Use a thin custom modular core with strong library-first bias.
- Rebuild collection/normalization/chunking in phase 1.
- Default source topology is Buyer’s Guide-primary with supporting-source fallback.
- Also implement unified-source retrieval as an experimental benchmark profile in phase 1.
- Keep KG out of parity scope and add it later as an experiment lane.
- Surface any required manual sign-up, download, or inaccessible external resource as an explicit user task instead of working around it silently.
