# Experiment Index

This file is the shortest accurate map of what has been tried recently and what
the repo currently believes.

It is not a full changelog. It points to the notes and artifacts that already
contain the supporting evidence.

## Public Surfaces

- control profiles:
  - [`../profiles/README.md`](../profiles/README.md)
- canonical benchmark namespace:
  - `bgrag.benchmarks.*`
- current strict tiered candidate ADR:
  - [agentic_architecture_adr.md](agentic_architecture_adr.md)
- persistent optimization-loop protocol:
  - [optimization_loop_protocol.md](optimization_loop_protocol.md)
- bounded retrieval-backend migration plan:
  - [retrieval_backend_spike_plan.md](retrieval_backend_spike_plan.md)
- completed OpenSearch spike result:
  - [opensearch_spike_note.md](opensearch_spike_note.md)

## Current Read

- status map:
  - control / candidate / archived profiles are recorded in
    [../profiles/README.md](../profiles/README.md)
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
- bounded retrieval-budget sweep on the serious rerank-shortlist lane:
  - shortlist/rerank budget matters
  - bigger is not monotonically better
- richer bounded tool-agent with an indexed retrieval tool:
  - still underperformed the serious indexed RAG lane
  - did not justify an "agentic" architectural jump
- strict tiered agentic candidate:
  - now exists as an explicit candidate profile family
  - fails hard if native Elasticsearch RRF is unavailable
  - does not silently fall back to repo-side fusion
- persistent optimization loop:
  - now has repo-native scaffolding
  - uses a canonical control-surface manifest, a persistent failure-surface
    manifest, and a cycle runner that writes loop artifacts
- bounded OpenSearch backend spike:
  - retrieval-only behavior was competitive
  - end-to-end quality did not beat the current control cleanly
  - latency was materially worse
  - not promoted
- single-call JSON answer-plus-confidence sidecar:
  - benchmarked in isolated answer replay on the frozen control runs
  - quality regressed on both canonical dev and holdout
  - one holdout case failed the JSON contract entirely
  - self-reported confidence was not trustworthy on several misses
  - not promoted
- deterministic bundle-risk audit:
  - added as a new small-cycle diagnostic over eval artifacts
  - catches some real holdout misses
  - overflags too many good cases to be used alone
  - useful as an auxiliary retry signal, not as a standalone trigger
- first large optimization-loop packing candidate:
  - `baseline_vector_rerank_shortlist_ranked_diverse`
  - improved one holdout failure case and some retrieval coverage
  - regressed canonical dev and holdout judged quality
  - increased serving latency materially
  - rejected
- first selective hybrid indexed-retry candidate:
  - `baseline_vector_rerank_shortlist_hybrid_retry`
  - combined structured retrieval assessment, deterministic bundle risk, and
    question-risk signals to trigger one widened indexed retry
  - canonical dev improved only on `HR_015`
  - canonical holdout regressed on `HR_002`
  - both persistent failure surfaces regressed
  - product-serving mean request time increased materially
  - rejected
- authority-support page preservation on the wider `acceptance49` surface:
  - `baseline_vector_rerank_shortlist_authority_reserve`
    - global reserve
    - rescued some support-page retrieval
    - regressed unrelated cases
    - rejected
  - `baseline_vector_rerank_shortlist_selective_authority_reserve`
    - selective one-slot reserve
    - useful diagnosis
    - retrieval improved, but broad judged quality stayed mixed
    - not promoted
  - `baseline_vector_rerank_shortlist_selective_authority_cluster`
    - selective same-document support cluster preservation
    - real retrieval win on `HR_043` / `HR_048`
    - modest holdout judged gain, slight dev judged regression
    - classified mixed; keep as a targeted intervention candidate only
- wording-trigger robustness audit:
  - manifest:
    [heuristic_trigger_robustness_v1.json](../datasets/eval/manifests/heuristic_trigger_robustness_v1.json)
  - generated slice:
    [heuristic_trigger_robustness_v1.jsonl](../datasets/eval/generated/heuristic_trigger_robustness_v1.jsonl)
  - audit artifacts:
    under `datasets/runs/heuristic_trigger_audit/`
  - verified read:
    - runtime wording-trigger heuristics are not fully paraphrase-robust
    - the current authority-cluster candidate loses its advantage on the
      paraphrased versions of the same source cases
    - therefore those heuristic gains should not be treated as generalized

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
- indexed-tool note:
  - [tool_agent_indexed_note.md](tool_agent_indexed_note.md)
- relevant policy:
  - [product_promotion_contract.md](product_promotion_contract.md)

## Retrieval Budgeting

### Serious rerank-shortlist budget sweep

- implementation:
  - [run_retrieval_budget_sweep.py](../scripts/run_retrieval_budget_sweep.py)
- note:
  - [retrieval_budget_sweep_note.md](retrieval_budget_sweep_note.md)
- verified read:
  - shortlist/rerank budget tuning does matter
  - the best bounded point so far is better than the current serious default
  - larger retrieval budgets alone still do not get the repo to `0.9+`

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

## Tiered Agentic Candidate

- ADR:
  - [agentic_architecture_adr.md](agentic_architecture_adr.md)
- canonical benchmark namespace:
  - `bgrag.benchmarks.*`
- candidate profiles:
  - [baseline_vector_rerank_shortlist_agentic.yaml](../profiles/baseline_vector_rerank_shortlist_agentic.yaml)
  - [demo_vector_rerank_shortlist_agentic.yaml](../profiles/demo_vector_rerank_shortlist_agentic.yaml)
- verified read:
  - the serving code now supports structured retrieval assessment, one indexed retry, and bounded official-site browsing
  - the strict candidate profiles target native Elasticsearch RRF and intentionally fail hard if the cluster cannot provide it

## Evaluation Surfaces

### Expanded diagnostic acceptance surface

- blueprint:
  - [acceptance49_blueprint.json](../datasets/eval/manifests/acceptance49_blueprint.json)
- authored batch:
  - [acceptance49_additions_batch1.json](../datasets/eval/manifests/acceptance49_additions_batch1.json)
- build script:
  - [build_acceptance49_working.py](../scripts/build_acceptance49_working.py)
- generated surfaces:
  - [acceptance49_working.jsonl](../datasets/eval/parity/acceptance49_working.jsonl)
  - [acceptance49_dev_draft.jsonl](../datasets/eval/dev/acceptance49_dev_draft.jsonl)
  - [acceptance49_holdout_draft.jsonl](../datasets/eval/holdout/acceptance49_holdout_draft.jsonl)
- verified read:
  - `acceptance49` is a broader diagnostic and acceptance surface, not a new frozen control
- the new batch was authored from the corpus snapshot plus live official-site structure awareness with low-context subagent help
  - it materially broadens workflow, authority, exactness, and cross-page synthesis coverage
  - it exposed a concrete new failure family:
    - authority/governance questions where TBS directive evidence appears in the
      raw shortlist but gets discarded by Buyer’s-Guide-first packing

## Core Reference Docs

- current repo role and state:
  - [current_state.md](current_state.md)
- optimization loop protocol:
  - [optimization_loop_protocol.md](optimization_loop_protocol.md)
- durable decisions:
  - [decision_log.md](decision_log.md)
- backend-alternative spike plan:
  - [retrieval_backend_spike_plan.md](retrieval_backend_spike_plan.md)
- completed OpenSearch spike:
  - [opensearch_spike_note.md](opensearch_spike_note.md)
- product acceptance and comparator rules:
  - [product_promotion_contract.md](product_promotion_contract.md)
- serving benchmark purpose:
  - [product_serving_benchmark.md](product_serving_benchmark.md)
- external research notes:
  - [research-log.md](research-log.md)
