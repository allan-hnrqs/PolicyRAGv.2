# Unit Audit Log

Each implementation unit should be audited for:

- code correctness
- test coverage
- architectural soundness
- anti-overfitting / anti-ad-hoc risk
- whether the result should remain in the repo

## Unit 1: Foundation

Scope:
- pinned implementation plan
- project metadata
- README and docs scaffolding
- settings
- shared domain models
- profile models and loader
- baseline profiles
- initial tests

Audit result:
- passed
- code compiled
- initial tests passed
- no ad hoc logic introduced

## Unit 2: Collection / normalization / chunking

Scope:
- collector
- normalizer
- metadata enrichers
- chunkers
- corpus persistence

Audit result:
- passed after one fix
- initial issue was registry registration order for metadata enrichers
- fixed by making registration explicit and adding subpackage entrypoints
- tests now pass

## Unit 3: Retrieval / answering / eval runtime skeleton

Scope:
- embedding wrapper
- Elasticsearch indexing helpers
- source-topology policies
- hybrid retriever
- answer strategies
- eval loader and runner
- parity freeze helper
- pipeline orchestration
- CLI wiring

Audit result:
- passed
- code compiled
- test suite remained green
- registry-based runtime pieces loaded correctly
- architecture stayed within the clean phase-1 boundary:
  - Buyer’s Guide-first topology
  - inline evidence default
  - documents mode retained only as an alternate strategy

Methodology note:
- the 19-case suite remains canonical
- the 39-case surface is still a rebuild task, not a frozen inherited benchmark
- the baseline runtime should not silently degrade when core dependencies are
  absent; strict failures are preferable to fallbacks for parity work
- provider-imposed limits should be handled explicitly and generically
  (for example, batched embedding requests), not via one-off hacks or manual
  chunk-count trimming
