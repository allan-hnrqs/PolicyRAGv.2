# Product Promotion Contract

This repo now treats product-serving changes as promotable only when they are measured on four surfaces together:

1. Canonical judged eval on dev
2. Canonical judged eval on holdout
3. Retrieval-only benchmark
4. Product-facing serving benchmarks
   - `datasets/eval/manifests/product_serving_benchmark_v1.json`
   - `datasets/eval/manifests/multiturn_benchmark_v1.json`

## Current anchor
- clean checkpoint commit: `58746ac`

## Promotion gates
- No canonical dev or holdout regression greater than `0.02` unless the change is explicitly intended to reset the baseline and the tradeoff is documented.
- Retrieval-only holdout packed expected URL recall must improve or hold.
- Warm product latency must improve on the same benchmark surface used for comparison.
- Exactness-sensitive cases must not introduce invented exact identifiers, file names, or form numbers.
- Multi-turn cases must preserve the current topic, comparison axes, and follow-up branch context.

## Benchmark commands
Canonical judged eval:

```powershell
.\.venv\Scripts\python.exe -m bgrag.cli eval datasets/eval/parity19_dev.jsonl --profile baseline
.\.venv\Scripts\python.exe -m bgrag.cli eval datasets/eval/parity19_holdout.jsonl --profile baseline
```

Retrieval-only:

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_benchmark.py --profile baseline --eval datasets/eval/parity19_dev.jsonl --query-mode profile
.\.venv\Scripts\python.exe scripts\run_retrieval_benchmark.py --profile baseline --eval datasets/eval/parity19_holdout.jsonl --query-mode profile
```

Serving/product:

```powershell
.\.venv\Scripts\python.exe scripts\run_product_benchmark.py --profile demo
.\.venv\Scripts\python.exe scripts\run_multiturn_benchmark.py --profile demo
```

## Reversibility rule
Every major serving change should land behind:
- a new profile, or
- a new retrieval/answering mode field

Do not overwrite the current default path until the new path has benchmark artifacts on all four surfaces.
