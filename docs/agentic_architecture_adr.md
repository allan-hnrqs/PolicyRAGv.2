# Agentic Architecture ADR

## Status

Accepted as the current candidate direction on 2026-04-04.

## Decisions

- Use a **tiered hybrid runtime**:
  - indexed RAG is the default path
  - official-site browsing is an explicit escalation path
- Use **Elasticsearch native RRF** as the target hybrid fusion method.
- Keep **Cohere Rerank** as the shortlist and page/passage reranker.
- Use **PydanticAI** for the bounded browse escalation loop and MCP-friendly
  tool orchestration.
- Do **not** mainline a Haystack or LangGraph rewrite unless the PydanticAI
  spike fails materially on the benchmark.
- Do **not** learn from agent “thoughts”.
  Distill only from:
  - tool traces
  - timings
  - visited URLs
  - benchmark outcomes

## Non-Decision

- This ADR does not claim that the current local environment can run the target
  architecture end to end.

Repo reality:

- the local Elasticsearch cluster currently rejects native RRF with a license
  error
- the strict agentic candidate profiles therefore fail hard locally rather than
  silently falling back to repo-side fusion

That is intentional. Silent fallback would make the benchmark lie about what is
actually running.

## Why

- Static indexed RAG alone is still below the quality target.
- Open official-site browsing can reach materially higher judged coverage, but
  it is too slow to be the default path.
- A tiered system is the cleanest way to preserve fast answers for routine
  cases while still allowing higher-effort recovery on hard questions.
