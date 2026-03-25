# Presentation Snapshot

This is the shortest truthful summary of the repo for a presentation or quick
handoff. For full history, use:

- `docs/presentation_pipeline_outline.md`
- `docs/current_state.md`
- `docs/decision_log.md`
- `docs/experiment_log.md`
- `docs/research-log.md`

## System in one sentence

PolicyRAGv.2 is a Buyer’s Guide-first procurement-policy RAG backend with
hybrid retrieval, query decomposition, strict evaluation, and a narrow
exactness-specific correction path.

## Current architecture

- Source topology:
  - Buyer’s Guide is primary
  - Buy Canadian policy and the TBS directive are supporting sources
- Retrieval:
  - Cohere dense embeddings
  - Elasticsearch lexical retrieval
  - hybrid blending
  - LLM query decomposition in the baseline path
- Answering:
  - default answer path is `inline_evidence_chat`
  - narrow missing-detail exactness subpath exists on `main`
  - broader answer-side branches remain experimental
- Runtime:
  - strict runtime
  - no silent fallback retrieval path
  - namespaced index manifests and run manifests

## Measurement

- Primary eval lane:
  - deterministic retrieval metrics
  - structured Cohere judge
  - required-claim recall
  - forbidden-claim violations
  - abstention accuracy
- Secondary eval lanes:
  - Ragas
  - OpenAI pairwise A/B
- For conditional branches, the trustworthy surface is:
  - intervention-only composite evaluation

## Headline numbers

- Best original `19`-case full run:
  - required-claim recall `0.8684`
  - answer failures `0`
  - forbidden-claim violations `0`
- Best original `19` dev run:
  - required-claim recall `0.8889`
- Best original `19` holdout run:
  - required-claim recall `0.8250`
- Rebuilt `39` baseline:
  - required-claim recall `0.7415`
  - answer failures `0`
  - forbidden-claim violations `1`
- Best rebuilt `39` broad experimental run:
  - required-claim recall `0.8047`
  - answer failures `0`
  - forbidden-claim violations `0`
  - not promoted because it regressed the original `19`
- Narrow exactness holdout intervention-only result on `main`:
  - recall `0.7778 -> 0.8889`
  - forbidden violations `1 -> 0`
  - abstain accuracy `0.6667 -> 1.0`

## Current quality read

- Better than the earlier prototype / previous system:
  - yes
- Usable as a supervised internal assistant:
  - yes
- Ready to rely on without human checking:
  - no

The broad bottleneck is no longer obvious retrieval failure. Retrieval is often
good enough that the remaining gap comes from answer precision, selective
activation, and exactness handling.

## What is canonical on `main`

- Buyer’s Guide-first hybrid RAG
- query decomposition in the baseline retriever
- strict runtime and snapshot-addressable evaluation
- narrow exactness-specific post-draft correction path
- stronger evaluation methodology than the original project had

## What is still experimental

- broad mode-aware answer routing
- broader verifier-gated structured answering
- page-rerank / document-seed retrieval families
- broad structured-presentation answer branches

## What the latest side branch showed

The most recent evidence-presentation branch found that prompt presentation does
matter, but the tested `selective_structured_answering` method still failed the
correct holdout intervention-only gate. That branch is useful as a negative
result and a methodology lesson, not a promotion candidate.

## Most honest next steps

- finish the phase-1 story with a clean presentation and demo
- keep baseline retrieval fixed
- return to the remaining compact answer-side regressions
- continue using intervention-only composites for conditional answer branches
- author and protect the final blind acceptance set
