# Parallel Worker Protocol

This repo can use parallel workers or subagents, but only if they follow strict
isolation and evidence rules.

## File isolation

- Never let multiple workers edit the same repo copy.
- Each worker must operate only inside its assigned disposable repo copy.
- The control repo for this protocol is:
  - `c:\Users\14164\Documents\CohereThing\PolicyRAGv.2`

## Elasticsearch and index isolation

- Read-only evals may share the current active snapshot namespace when the
  cycle is explicitly using a shared control index.
- If a worker needs to build a new index, it must use a unique namespace that
  includes its lane name.
- Workers must never delete or overwrite another worker's namespace.

## Methodology

- Research before implementation.
- Prefer existing libraries and frameworks over bespoke code.
- Do not let test convenience limit method choice, but still validate every
  serious change.
- Keep experiments human-readable:
  - update `docs/optimization_loop_protocol.md`
  - update `docs/decision_log.md`
  - update `docs/experiment_index.md`

## Promotion discipline

- Disposable worker copies are for exploration.
- Nothing from a worker copy is trusted automatically.
- Promotion into control requires an audit from the control side.

## Recommended lane roles

- retrieval auditor:
  - inspect raw shortlist, selected, packed, and failure traces
- benchmark auditor:
  - inspect whether cases are representative, atomic, or reward-hackable
- bounded implementer:
  - own one subsystem only:
    - retrieval/packing
    - serving/escalation
    - benchmark schema/tests
    - cleanup/docs

## Practical Elasticsearch rule

- The shared baseline namespace stays read-only.
- Workers may run read-only evals against that namespace.
- If any worker needs a new build or index mutation, it must:
  - use a unique namespace
  - keep that namespace inside its own disposable lane
  - record the namespace in that lane's notes or cycle artifact
  - record the reason in `docs/decision_log.md` or the loop artifact

## Worker CLI safety

- The control repo now detects project root from the current working tree first,
  with optional override via `BGRAG_PROJECT_ROOT`.
- This fixes a worker issue where the installed `bgrag` entrypoint could resolve
  back to the control repo package path and write artifacts into the wrong copy.
- Worker commands should still be run from the worker repo root.
