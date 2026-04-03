# Product Acceptance And Promotion Contract

This repo should stop treating "slightly better than yesterday" as enough. A
candidate becomes product-credible only if it clears explicit quality,
exactness, multi-turn, and latency bars on the same authored surfaces.

## Current anchor
- clean checkpoint commit: `58746ac`
- current strongest answer transport: inline evidence
- grounded `documents` path is now experimental only and is documented as a
  failed mainline phase in `docs/phase4_documents_note.md`

## Required benchmark surfaces
Every promotable serving change must be measured on all of these together:

1. Canonical judged eval on dev
2. Canonical judged eval on holdout
3. Retrieval-only benchmark
4. Product-facing serving benchmarks
   - `datasets/eval/manifests/product_serving_benchmark_v1.json`
   - `datasets/eval/manifests/multiturn_benchmark_v1.json`
5. Official-site live browsing baseline on the same authored eval surface
   - `scripts/run_official_site_baseline.py`
6. Tool-using official-site agent baseline on the same authored eval surface
   - `scripts/run_tool_agent_baseline.py`
7. Open official-site browsing agent comparator on the same authored eval surface
   - freer live browsing restricted to official domains only
   - full answer generation happens in the browsing agent, not in the indexed RAG answerer
   - this comparator may use official pages beyond the case's original primary and supporting URLs

## Provisional product-usable bar
These are the current minimum bars for calling the system usable for real
people. They are not paper metrics; they are shipping thresholds.

- Canonical dev and holdout judged eval:
  - no regression greater than `0.02` versus the accepted anchor unless the
    baseline reset is deliberate and documented
- Product benchmark:
  - at least `90%` of cases judged acceptable in manual review
- Exactness-sensitive slice:
  - `0` invented exact identifiers, form numbers, template names, file names,
    or contact details
- Multi-turn benchmark:
  - at least `85%` of cases must preserve the topic, comparison axes, and
    follow-up branch context
- Latency:
  - warm `p50 <= 8s`
  - warm `p95 <= 15s`

These numbers are intentionally quality-first. If the system cannot answer
reliably, shaving a few seconds is not a meaningful win.

## Comparator rule
The indexed RAG system is not automatically justified just because it is the
current architecture.

Before calling a serving profile product-worthy, compare it against:
- the bounded official-site live browsing baseline, and
- the bounded tool-using official-site agent baseline

Both should use the same authored eval surface.

- If the live browsing baseline is clearly better on quality and exactness, the
  indexed system needs a strong justification in latency, cost, determinism, or
  reproducibility.
- If the indexed system is not meaningfully better on any of those dimensions,
  the architecture should be reconsidered rather than defended out of inertia.

The live baseline in this repo is intentionally bounded:
- it uses the local canonical URL inventory as the page universe
- it reranks candidate official pages
- it fetches live official pages before answering

It is not a full autonomous browser agent. That limitation should be stated
explicitly whenever the results are used.

The tool-using agent baseline is also intentionally bounded:
- it can only search the official-site inventory, fetch live official pages,
  rerank fetched chunks, and answer
- it has a fixed step budget
- it logs every tool action

It is closer to a real assistant-with-tools competitor, but it is still not an
unbounded browser agent.

The open official-site browsing agent comparator is a different lane:
- extra official pages beyond the original primary and supporting URLs are
  allowed
- only official domains in the declared scope are allowed
- the run must record the visited URLs and wall-clock timing per case

The point of that lane is to answer the real product question:
- what happens if a user gives a strong browsing agent the official site and
  lets it work?

Do not quietly treat that as the same task as restricted-source RAG.

For open-browse comparisons:
- cite official URLs actually used
- score exactness and hallucination the same way as indexed RAG
- do not fail a case just because the agent relied on an additional official
  page
- do fail the case if the answer overstates or invents unsupported specifics

Cases authored around restricted-source abstention must be handled explicitly:
- if the abstention premise is "the provided primary/supporting sources do not
  contain this detail," that is a restricted-source case
- if the official site itself contains the missing detail elsewhere, the case
  should either be duplicated into an open-browse variant or excluded from
  open-browse aggregate comparisons

Example:
- `HR_017` can be a valid restricted-source abstain case and an invalid
  open-browse abstain case if the official site has the contact details on a
  separate official page

## External benchmark interpretation rule
Do not import generic "need `0.9`" claims from papers or blog posts without
understanding what those numbers actually measure.

Before using any published number as a target, record:
- the dataset name
- the case count
- the task type:
  - retrieval
  - grounded answer generation
  - citation finding
  - pairwise preference
- the difficulty profile:
  - lookup
  - synthesis
  - multi-hop
  - exactness-sensitive
  - long-context
- whether the metric is end-to-end or only retrieval

If those details are missing or not comparable to this repo, the external number
is advisory only.

## Benchmark governance
The authored benchmark is allowed to change. It is not sacred.

Cases should be retired, rewritten, or replaced when they are:
- unrepresentative of real user asks
- redundant with other cases
- too easy to discriminate useful system changes
- mismatched to the product's actual use cases
- based on outdated or no-longer-authoritative source content

Each case should be tagged by problem shape:
- lookup
- synthesis
- exactness
- multi-turn
- navigation
- workflow

Each case should also declare whether it is valid for:
- restricted-source comparison
- open-browse comparison
- both

This repo should prefer a smaller representative benchmark over a larger but
muddy benchmark.

## Existing-tool rule
Do not rebuild generic infra from scratch without checking whether a mature tool
already covers the need.

This especially applies to:
- query planning / routing
- web-search fallbacks
- reranking
- hybrid retrieval orchestration

But imported tools still need to earn their place on the benchmark surfaces
above. Tool novelty is not a promotion criterion.

## Timing rule
Timing claims must match what we can actually observe.

- Repo-controlled baselines should record stage timings whenever possible:
  - retrieval
  - rerank
  - packing
  - answer generation
- External browsing-agent comparisons should, at minimum, record per-case
  dispatch-to-completion wall time
- If the polling method only gives an upper bound on completion time, label it
  as an upper bound instead of pretending it is exact

Do not compare stage timings from repo-controlled runs to black-box browsing
agent runs as if they were the same measurement type.

## Benchmark commands
Canonical judged eval:

```powershell
.\.venv\Scripts\python.exe -m bgrag.cli eval datasets/eval/dev/parity19_dev.jsonl --profile baseline
.\.venv\Scripts\python.exe -m bgrag.cli eval datasets/eval/holdout/parity19_holdout.jsonl --profile baseline
```

Retrieval-only:

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_benchmark.py --profile baseline --eval datasets/eval/dev/parity19_dev.jsonl --query-mode profile
.\.venv\Scripts\python.exe scripts\run_retrieval_benchmark.py --profile baseline --eval datasets/eval/holdout/parity19_holdout.jsonl --query-mode profile
```

Serving/product:

```powershell
.\.venv\Scripts\python.exe scripts\run_product_benchmark.py --profile demo
.\.venv\Scripts\python.exe scripts\run_multiturn_benchmark.py --profile demo
```

Official-site live baseline:

```powershell
.\.venv\Scripts\python.exe scripts\run_official_site_baseline.py --eval datasets/eval/dev/parity19_dev.jsonl --answer-profile baseline_vector
```

Tool-using official-site agent baseline:

```powershell
.\.venv\Scripts\python.exe scripts\run_tool_agent_baseline.py --eval datasets/eval/dev/parity19_dev.jsonl --answer-profile baseline_vector
```

## Reversibility rule
Every major serving change should land behind:
- a new profile, or
- a new retrieval/answering mode field

Do not overwrite the current default path until the new path has benchmark
artifacts on all required surfaces.
