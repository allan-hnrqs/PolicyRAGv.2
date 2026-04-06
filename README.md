# PolicyRAGv.2

This repository is a clean-room, backend-only rebuild of the Buyer’s Guide RAG
system. It is inspired by `DepartmentDefence-Winter2026-feat-retrieval-expensive-methods-eval`
as a benchmark and behavior reference, but it does not copy implementation code
from that branch.

## Relation To `DepartmentDefence-Winter2026`

This repo currently sits beside
[DepartmentDefence-Winter2026](../DepartmentDefence-Winter2026/README.md).

The practical split is:

- `DepartmentDefence-Winter2026`
  - integrated app repo
  - Flask backend, React frontend, auth, chat UI, moderator dashboard
- `PolicyRAGv.2`
  - cleaner backend experimentation and evaluation repo
  - retrieval, chunking, answering, evaluation, and product-comparator work
  - used to test backend ideas more rigorously before deciding whether they are
    worth carrying into the app repo

So this repo is not the product shell. It is the benchmark-heavy backend lab
and decision surface for the procurement-policy RAG work.

## Repo Status

This Git repository is now the primary home for ongoing development.

- public GitHub repo:
  - `https://github.com/allan-hnrqs/PolicyRAGv.2`

- primary tracked repo:
  - `c:\Users\14164\Documents\CohereThing\PolicyRAGv.2`
- legacy local fallback:
  - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean`

New work, commits, and branches should happen here unless there is a specific
reason to use an isolated disposable copy for experimentation.

## Purpose

The system is Buyer’s Guide-first:
- the Buyer’s Guide is the primary retrieval surface
- Buy Canadian policy and the TBS directive are supporting sources
- supporting sources should be pulled in intentionally rather than treated as
  equal peers by default

## Current Status

This repo is now beyond bare bootstrap. It is the active backend experimentation
repo for:

- corpus collection and normalization
- retrieval and answer-strategy experiments
- judged eval and retrieval-only benchmarking
- product-serving benchmarking
- comparisons against live official-site and agent-style comparators

The repo still aims to provide a clean, documented, testable backend that can:
- collect and normalize the procurement sources
- build a reusable corpus package
- index the corpus with pluggable retrieval profiles
- answer questions with pluggable answer strategies
- evaluate against frozen parity suites derived from the current `feat` branch

The current strong takeaways are:
- inline evidence remains the strongest answer transport tested so far
- the best reversible infrastructure checkpoint so far is the Phase 1 native
  Elasticsearch vector retrieval path
- the current serious indexed-RAG control is:
  - `baseline_vector_rerank_shortlist`
- the current live tiered candidate is:
  - `baseline_vector_rerank_shortlist_agentic`
- grounded Cohere `documents` answering was tested seriously and is currently
  documented as an unsuccessful mainline path for this repo's synthesis-heavy
  task
- bounded shortlist/rerank budget tuning matters, but larger retrieval budgets
  alone have still not produced a `0.9+` judged-quality lane
- a richer bounded tool-agent baseline with an indexed retrieval tool still did
  not outperform the serious indexed RAG lane

The current promoted baseline family uses:
- Buyer’s Guide-first source topology
- Cohere hybrid retrieval with Elasticsearch lexical search
- LLM query decomposition before retrieval
- inline evidence answer generation with Command A

## Where To Read Next

If you need the shortest trustworthy map of the repo, start here:

- [Current State](docs/current_state.md)
- [Decision Log](docs/decision_log.md)
- [Experiment Index](docs/experiment_index.md)
- [Profile Surface](profiles/README.md)
- [Product Promotion Contract](docs/product_promotion_contract.md)
- [Product Serving Benchmark](docs/product_serving_benchmark.md)
- [Research Log](docs/research-log.md)

The main recent phase notes are:

- [Phase 1: native vector retrieval](docs/phase1_vector_retrieval_note.md)
- [Phase 2: rerank experiment](docs/phase2_rerank_note.md)
- [Phase 3: span-aware packing](docs/phase3_span_packing_note.md)
- [Phase 4: grounded `documents` answering](docs/phase4_documents_note.md)
- [Agentic architecture ADR](docs/agentic_architecture_adr.md)
- [Retrieval backend spike plan](docs/retrieval_backend_spike_plan.md)
- [Retrieval budget sweep](docs/retrieval_budget_sweep_note.md)
- [Indexed tool-agent baseline](docs/tool_agent_indexed_note.md)

## CLI

The intended CLI surface is:
- `bgrag collect`
- `bgrag build-corpus`
- `bgrag build-index`
- `bgrag query`
- `bgrag eval`
- `bgrag eval-ragas`
- `bgrag eval-pairwise`
- `bgrag freeze-baseline`
- `bgrag inspect`

## Layout

- `src/bgrag/`: package source
- `src/bgrag/benchmarks/`: canonical benchmark/comparator import namespace
- `profiles/`: named runtime profiles
- `datasets/`: evaluation suites, manifests, and seed inputs
- `docs/`: architecture, ADRs, and research log
- `tests/`: unit and integration tests
- `scripts/`: helper entrypoints for repeatable local workflows

Useful benchmark runners:
- [`scripts/run_retrieval_budget_sweep.py`](scripts/run_retrieval_budget_sweep.py)
- [`scripts/run_tool_agent_baseline.py`](scripts/run_tool_agent_baseline.py)
- [`scripts/run_retrieval_benchmark.py`](scripts/run_retrieval_benchmark.py)
- [`scripts/run_product_benchmark.py`](scripts/run_product_benchmark.py)

Canonical benchmark module namespace:
- `bgrag.benchmarks.retrieval`
- `bgrag.benchmarks.product`
- `bgrag.benchmarks.answer_replay`
- `bgrag.benchmarks.official_site`
- `bgrag.benchmarks.tool_agent`
- `bgrag.benchmarks.retrieval_budget_sweep`

## Claude Collaboration

Claude-specific project guidance lives in [CLAUDE.md](CLAUDE.md). A resumable
consultation wrapper for Claude Opus 4.6 lives in
[`scripts/consult_claude.ps1`](scripts/consult_claude.ps1), with local session
state kept out of git under `.claude/session_local/`.

## First steps

1. Install the package in editable mode:

```powershell
pip install -e .[dev]
```

2. Copy or create a `.env` file with your Cohere and Elasticsearch settings.

Both are real runtime dependencies for the intended baseline:
- Cohere is required for embeddings, rerank, and answer generation
- Elasticsearch is required for lexical retrieval
- the repo does not provide local fallback retrieval paths for the baseline architecture

3. Inspect the default profile:

```powershell
bgrag inspect
```

4. Run the test suite:

```powershell
pytest
```

5. Run the main judged eval or the secondary Ragas eval:

```powershell
bgrag eval datasets/eval/parity/smoke_one.jsonl --profile baseline
bgrag eval-ragas datasets/eval/parity/smoke_one.jsonl --profile baseline
bgrag eval-pairwise datasets/runs/control.json datasets/runs/candidate.json
```

6. Start the local Elasticsearch dev node used by the baseline runtime:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_elasticsearch.ps1
```

## Key documents

- [Implementation Plan](docs/implementation_plan.md)
- [Research Log](docs/research-log.md)
- [Architecture Survey](docs/architecture-survey/README.md)
