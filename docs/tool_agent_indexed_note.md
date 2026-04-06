# Indexed Tool-Agent Baseline

This note records the richer bounded tool-agent comparator where the agent can
use:

- `search_inventory`
- `fetch_page`
- `rerank_fetched_chunks`
- `retrieve_indexed_chunks`
- `answer`

## Scope

- eval surface:
  - `datasets/eval/parity/parity19.jsonl`
- answer profile:
  - `baseline_vector_rerank_shortlist`
- indexed retrieval tool profile:
  - `baseline_vector_rerank_shortlist`
- artifact:
  - [tool_agent_baseline_vector_rerank_shortlist_20260404_055012_225902_eb41.md](../datasets/runs/tool_agent_baseline/tool_agent_baseline_vector_rerank_shortlist_20260404_055012_225902_eb41.md)

## Results

- required-claim recall mean: `0.7149`
- mean case seconds: `61.10`
- packed expected-URL recall mean: `0.6842`
- packed claim-evidence recall mean: `0.8158`
- forbidden-claim violations: `0`

## Trace Read

- The richer agent did not become meaningfully smarter just because it had one
  more tool.
- It stayed browse-heavy:
  - `search_inventory`: `19`
  - `fetch_page`: `24`
  - `rerank_fetched_chunks`: `19`
  - `retrieve_indexed_chunks`: `3`
- The indexed retrieval tool was used in only `3` of `19` cases.

## Decision

- Do not treat this bounded tool-agent baseline as proof that “agentic RAG” is
  already the right product architecture here.
- The current bounded agent runner underperforms the serious indexed RAG lane
  on both quality and latency.
- If the repo revisits agentic orchestration, it should learn from tool traces
  and action utility, not from vague “agentic” branding.
