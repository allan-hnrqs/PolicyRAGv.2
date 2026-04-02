# Product Serving Benchmark

This file defines the small prompt bank used to evaluate whether a profile is
getting closer to a usable chat product.

It is intentionally different from the canonical parity benchmark surfaces.
The parity surfaces are still the benchmark history for retrieval and answer
quality. This serving benchmark exists to check product behavior that the
canonical eval bank does not represent well:

- greeting/help handling
- short multi-turn follow-ups
- latency at each stage
- exactness-sensitive prompts that deserve manual inspection

## Scope

The current manifest is:

- [`datasets/eval/manifests/product_serving_benchmark_v1.json`](../datasets/eval/manifests/product_serving_benchmark_v1.json)

It includes:

- chat-shell sanity cases
- first-turn procurement questions
- follow-up conversational questions
- exactness-sensitive prompts
- navigation and workflow prompts

## How To Run

From the repo root:

```powershell
.\.venv\Scripts\python.exe scripts\run_product_benchmark.py --profile demo
```

To compare against baseline:

```powershell
.\.venv\Scripts\python.exe scripts\run_product_benchmark.py --profile baseline
```

The script writes two artifacts under `datasets/runs/product_benchmark/`:

- a JSON artifact with full answers, timings, and metadata
- a Markdown summary for quick inspection

## What It Checks

For each case, the runner records:

- resolved question
- answer text
- citation count
- response mode
- stage timings
  - contextualization
  - query planning
  - query embedding
  - retrieval
  - answer generation
  - total answer path
- total request

It does not apply automatic answer-content pass/fail rules. The point is to
capture the system's actual behavior and make manual review easier, not to
smuggle in weak proxy metrics.

## What It Is Not

This is not:

- a replacement for parity or holdout evaluation
- a statistically strong promotion gate
- a full conversational quality benchmark
- an automatic pass/fail grader for chat quality

Use it as a serving-quality and latency surface, not as the only basis for
architectural promotion.
