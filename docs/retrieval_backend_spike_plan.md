# Retrieval Backend Spike Plan

This note defines a bounded, evidence-driven way to test whether a different
retrieval backend is worth adopting.

It is not permission to rewrite the repo.

## Current Control

The current serious indexed control remains:

- profile:
  - [`../profiles/baseline_vector_rerank_shortlist.yaml`](../profiles/baseline_vector_rerank_shortlist.yaml)
- current shape:
  - Elasticsearch BM25
  - Elasticsearch kNN
  - repo-side hybrid fusion
  - Cohere shortlist rerank
  - inline answer generation

This control is the comparison target for any backend spike.

## Why This Exists

The repo now has two separate truths:

- native Elasticsearch RRF is attractive on paper
- the local self-managed cluster currently rejects it on the active `basic`
  license

That does **not** justify an immediate platform migration.

It justifies a bounded spike plan with honest success criteria.

## Primary Question

Can another retrieval backend materially improve the repo's retrieval and
end-to-end answer quality without forcing a worse operational tradeoff than the
current control?

For this repo, "materially improve" means:

- better judged answer quality on the canonical dev/holdout split
- no hidden dependence on browse-heavy behavior
- acceptable latency relative to the current serious control
- no collapse in observability or debuggability

## Candidate Order

### 1. OpenSearch

Why it is first:

- closest operationally to the current Elasticsearch-based stack
- supports native hybrid query execution
- supports shard/coordinator-side normalization and combination through search
  pipelines

Official references:

- OpenSearch hybrid query:
  - https://docs.opensearch.org/latest/query-dsl/compound/hybrid/
- Repo-relevant details from the docs:
  - hybrid combines multiple queries into one score
  - subquery scores are combined through a search pipeline
  - rescoring is supported for hybrid queries
  - there are real query-shape limitations that would need to be respected

Repo inference:

- this is the lowest-risk backend spike if the goal is "keep the same general
  shape, but stop depending on Elastic's RRF licensing path"

### 2. Qdrant

Why it is second:

- stronger native support for hybrid and multi-stage query orchestration than
  the current repo core
- explicit support for `rrf` and `dbsf`
- cleaner built-in support for staged retrieval experiments

Official references:

- Qdrant hybrid queries:
  - https://qdrant.tech/documentation/search/hybrid-queries/
- Repo-relevant details from the docs:
  - hybrid queries support `prefetch`
  - fusion supports `rrf`
  - fusion also supports `dbsf`
  - the same API supports multi-stage query patterns

Repo inference:

- if the repo eventually wants cleaner native support for multi-stage retrieval
  without so much hand-rolled orchestration, Qdrant is the strongest next spike
  after OpenSearch

### 3. Weaviate

Why it is not first:

- it has strong hybrid and rerank support
- but it implies a more opinionated platform shift than OpenSearch
- and it still does not obviously solve the repo's core serving problem by
  itself

Official references:

- Weaviate hybrid search:
  - https://docs.weaviate.io/weaviate/search/hybrid
- Weaviate rerank:
  - https://docs.weaviate.io/weaviate/search/rerank

Repo-relevant details from the docs:

- hybrid combines vector search and BM25F
- fusion method and weights are configurable
- reranking is a first-class query feature
- explain-score metadata exists

Repo inference:

- this is a credible platform, but it is a less disciplined first spike for
  this repo than OpenSearch or Qdrant

### 4. Vespa

Why it is last:

- technically the most powerful option in this list
- operationally the biggest jump by far

Official references:

- Vespa query API:
  - https://docs.vespa.ai/en/querying/query-api.html
- Vespa phased ranking:
  - https://docs.vespa.ai/en/ranking/phased-ranking.html

Repo-relevant details from the docs:

- queries can combine `nearestNeighbor(...)` with text operators like
  `userQuery()`
- `second-phase` ranking exists on content nodes
- `global-phase` reranking exists after gathering top hits
- `global-phase` supports cross-hit normalization and reciprocal-rank-fusion-like
  composition

Repo inference:

- Vespa is only worth spiking if ranking control becomes the main product moat
- right now it is too large a platform shift for a first alternative

## What A Spike Must Keep Fixed

To keep the comparison honest, each backend spike must preserve:

- the same frozen corpus contents
- the same chunking policy
- the same answer strategy
- the same judge surface
- the same dev/holdout split

The backend spike is allowed to change:

- lexical/vector retrieval engine
- fusion mechanism
- shortlist generation
- native reranking or staged retrieval support

The spike is **not** allowed to change:

- answer transport
- evaluation contract
- benchmark definitions
- product success criteria

## Required Measurements

Every spike must record:

- retrieval-only metrics:
  - raw shortlist recall
  - selected recall
  - packed recall
  - chunk/claim support recall
  - URL/page recall
- end-to-end metrics:
  - required-claim recall
  - exactness failures
  - forbidden-claim violations
- latency:
  - retrieval time
  - rerank time
  - answer time
  - total wall-clock time
- observability:
  - enough trace data to explain why a document/chunk ranked where it did

## Success Criteria

A backend spike should only be considered promotion-worthy if it does all of
these:

1. Beats the current control on judged dev quality.
2. Beats or at least matches the current control on judged holdout quality.
3. Does not hide a regression behind coarser retrieval metrics.
4. Does not require browse-style escalation just to look good.
5. Does not create a worse operational dependency than the problem it solves.

For this repo, a backend spike that is "interesting" but still below the
current quality target is still just research, not a promotion candidate.

## Recommended Spike Order

### Phase A: OpenSearch viability spike

Goal:

- check whether native hybrid + search-pipeline scoring can simplify the
  current control path without hurting quality

Minimum scope:

- one ingestion/index build
- one retrieval-only benchmark run
- one dev judged run
- one holdout judged run

Promotion rule:

- only continue past OpenSearch if it is competitive with the current control

### Phase B: Qdrant retrieval-shape spike

Goal:

- test whether native `prefetch` + fusion + multi-stage query support can
  outperform the current hand-rolled shortlist logic

Minimum scope:

- dense + sparse indexing plan
- hybrid query with `rrf`
- one retrieval-only benchmark run
- one dev judged run
- one holdout judged run

Promotion rule:

- only consider broader adoption if it beats the OpenSearch spike or clearly
  beats the current control

### Phase C: Stop

Do **not** spike Weaviate or Vespa unless Phase A and Phase B both fail to
produce a compelling path.

That is not conservatism. It is discipline.

## What Not To Do

- do not rewrite the repo before the first spike result exists
- do not mix backend changes with answer-prompt changes
- do not claim a backend win from a browse-heavy comparator
- do not confuse "more features" with "better fit for this repo"
- do not trust vendor positioning over benchmark evidence

## Recommendation

The next backend research move should be:

1. keep the current control as-is
2. run a bounded OpenSearch spike first
3. run a bounded Qdrant spike only if OpenSearch is unconvincing
4. avoid deeper platform migration until one of those spikes earns it

That is the shortest path to real evidence without turning the repo into a
multi-quarter infrastructure detour.
