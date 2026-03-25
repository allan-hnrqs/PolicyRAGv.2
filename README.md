# PolicyRAGv.2

This repository is a clean-room, backend-only rebuild of the Buyer’s Guide RAG
system. It is inspired by `DepartmentDefence-Winter2026-feat-retrieval-expensive-methods-eval`
as a benchmark and behavior reference, but it does not copy implementation code
from that branch.

## Start Here

If you only need the current high-level state:

- [Presentation Snapshot](docs/presentation_snapshot.md)
- [Current State](docs/current_state.md)
- [Decision Log](docs/decision_log.md)

## Repo Status

This Git repository is now the primary home for ongoing development.

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

## Current Baseline

The current promoted baseline uses:
- Buyer’s Guide-first source topology
- Cohere hybrid retrieval with Elasticsearch lexical search
- LLM query decomposition before retrieval
- inline evidence answer generation with Command A
- strict runtime with namespaced index and run manifests

Current quality read:
- usable as a supervised internal assistant
- not yet reliable enough to be treated as a fully trusted final authority

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
- `profiles/`: named runtime profiles
- `datasets/`: evaluation suites, manifests, and seed inputs
- `docs/`: architecture, ADRs, and research log
- `tests/`: unit and integration tests
- `scripts/`: helper entrypoints for repeatable local workflows

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

## Key Documents

- [Presentation Snapshot](docs/presentation_snapshot.md)
- [Current State](docs/current_state.md)
- [Decision Log](docs/decision_log.md)
- [Experiment Log](docs/experiment_log.md)
- [Research Log](docs/research-log.md)
- [Implementation Plan](docs/implementation_plan.md)
- [Architecture Survey](docs/architecture-survey/README.md)
