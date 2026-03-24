# Parallel Worker Protocol

This project can safely run multiple background workers if they follow strict
isolation rules.

## File isolation

- Never let multiple workers edit the same repo copy.
- Each worker must operate only inside its assigned disposable repo copy.
- The control repo remains:
  - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean`

## Elasticsearch and index isolation

- Read-only evals may share the existing snapshot namespace:
  - `baseline_embed_english_v3_0_dafb9a0708`
- If a worker needs to build a new index, it must use a unique namespace that
  includes its lane name.
- Workers must never delete or overwrite another worker's namespace.

## Methodology

- Research before implementation.
- Prefer existing libraries and frameworks over bespoke code.
- Do not let test convenience limit method choice, but still validate every
  serious change.
- Keep experiments human-readable:
  - update `docs/experiment_log.md`
  - update `docs/decision_log.md`
  - update `docs/current_state.md`

## Promotion discipline

- Disposable worker copies are for exploration.
- Nothing from a worker copy is trusted automatically.
- Promotion into control requires an audit from the control side.

## Active lanes (2026-03-24)

- disagreement lane:
  - repo copy:
    - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean-worker_disagreement_20260324_163228`
  - role:
    - inspect rebuilt-39 disagreement cases and look for bounded answer-side refinements
- research lane:
  - repo copy:
    - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean-worker_research_20260324_163228`
  - role:
    - research answer-verification / specialization methods and comparable evaluation practices
- infra lane:
  - repo copy:
    - `c:\Users\14164\Documents\CohereThing\buyers-guide-rag-clean-worker_infra_20260324_163228`
  - role:
    - improve namespace-safe background run and attribution infrastructure

## Practical Elasticsearch rule

- The shared baseline namespace stays read-only:
  - `baseline_embed_english_v3_0_dafb9a0708`
- Workers may run read-only evals against that namespace.
- If any worker needs a new build or index mutation, it must:
  - use a unique namespace
  - keep that namespace inside its own disposable lane
  - record the namespace in that lane's `docs/current_state.md`
  - record the reason in that lane's `docs/experiment_log.md`

## Worker CLI safety

- The control repo now detects project root from the current working tree first,
  with optional override via `BGRAG_PROJECT_ROOT`.
- This fixes a worker issue where the installed `bgrag` entrypoint could resolve
  back to the control repo package path and write artifacts into the wrong copy.
- Worker commands should still be run from the worker repo root.
