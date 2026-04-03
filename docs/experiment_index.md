# Experiment Index

This file is the shortest accurate map of what has been tried recently and what
the repo currently believes.

It is not a full changelog. It points to the notes and artifacts that already
contain the supporting evidence.

## Current Read

- strongest answer transport tested so far:
  - `inline_evidence_chat`
- strongest reversible infrastructure checkpoint so far:
  - Phase 1 Elasticsearch native vector retrieval
- grounded Cohere `documents` answering:
  - useful as an experiment
  - not promoted as the mainline answer path
- open official-site browsing agent comparator:
  - materially stronger on judged coverage than the indexed RAG checkpoints
  - materially slower

## Mainline Phase Notes

### Phase 1: Elasticsearch native vector retrieval

- note:
  - [phase1_vector_retrieval_note.md](phase1_vector_retrieval_note.md)
- verified read:
  - real latency win
  - real retrieval-speed win
  - keep as the main reversible infrastructure checkpoint

### Phase 2: bounded rerank and parallel query branches

- note:
  - [phase2_rerank_note.md](phase2_rerank_note.md)
- verified read:
  - judged quality regressed
  - latency worsened badly
  - not promoted

### Phase 3: span-aware evidence packing

- note:
  - [phase3_span_packing_note.md](phase3_span_packing_note.md)
- verified read:
  - packed-rank behavior improved in places
  - end-to-end answer quality got worse
  - not promoted

### Phase 4: grounded `documents` answering

- note:
  - [phase4_documents_note.md](phase4_documents_note.md)
- verified read:
  - better chat-shell behavior in some serving cases
  - weaker benchmark quality
  - later replay work did not rescue it
  - not promoted as the mainline answer path

## Comparator Lanes

### Official-site live browse baseline

- purpose:
  - lower-bound comparator that fetches live official pages and answers from
    them
- implementation:
  - [run_official_site_baseline.py](../scripts/run_official_site_baseline.py)
- relevant policy:
  - [product_promotion_contract.md](product_promotion_contract.md)

### Bounded tool-using official-site agent baseline

- purpose:
  - slightly more agentic official-site comparator with a fixed tool budget
- implementation:
  - [run_tool_agent_baseline.py](../scripts/run_tool_agent_baseline.py)
- relevant policy:
  - [product_promotion_contract.md](product_promotion_contract.md)

### Open official-site browsing agent comparator

This is the strongest external comparator tested so far.

- dev slice artifact:
  - [agent_browse_dev_slice_20260403_030313_036.md](../datasets/runs/agent_browse_benchmark/agent_browse_dev_slice_20260403_030313_036.md)
- holdout slice artifact:
  - [agent_browse_holdout_slice_20260403_0315.md](../datasets/runs/agent_browse_benchmark/agent_browse_holdout_slice_20260403_0315.md)
- verified read:
  - dev required-claim recall mean: `0.9444`
  - holdout required-claim recall mean: `0.9500`
  - no forbidden-claim violations in either slice
  - mean wall-clock upper bound stayed around `52s`

This comparator is now explicitly treated as a separate lane in:
- [product_promotion_contract.md](product_promotion_contract.md)

The important distinction is:
- restricted-source RAG and open official-site browsing are not the same task
- extra official pages are allowed in the open-browse lane
- exactness and hallucination standards still apply

## Core Reference Docs

- current repo role and state:
  - [current_state.md](current_state.md)
- durable decisions:
  - [decision_log.md](decision_log.md)
- product acceptance and comparator rules:
  - [product_promotion_contract.md](product_promotion_contract.md)
- serving benchmark purpose:
  - [product_serving_benchmark.md](product_serving_benchmark.md)
- external research notes:
  - [research-log.md](research-log.md)
