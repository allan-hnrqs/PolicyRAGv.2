# OpenSearch Spike Note

This note records the bounded OpenSearch backend spike that preserved the
current retrieval and answer architecture while swapping the search backend.

## Scope

The spike kept these pieces fixed:

- corpus
- chunking
- answer strategy
- Cohere shortlist rerank
- eval harness and structured judge

It changed only the indexed search backend:

- control:
  - Elasticsearch
- spike:
  - OpenSearch

The spike profile is:

- [`../profiles/baseline_vector_rerank_shortlist_opensearch.yaml`](../profiles/baseline_vector_rerank_shortlist_opensearch.yaml)

## Retrieval-only Results

Artifacts:

- dev:
  - [`../datasets/runs/retrieval_benchmark/baseline_vector_rerank_shortlist_opensearch_retrieval_profile_20260404_084644_082191_a228.json`](../datasets/runs/retrieval_benchmark/baseline_vector_rerank_shortlist_opensearch_retrieval_profile_20260404_084644_082191_a228.json)
- holdout:
  - [`../datasets/runs/retrieval_benchmark/baseline_vector_rerank_shortlist_opensearch_retrieval_profile_20260404_084807_194214_bb11.json`](../datasets/runs/retrieval_benchmark/baseline_vector_rerank_shortlist_opensearch_retrieval_profile_20260404_084807_194214_bb11.json)

Compared with the current control retrieval runs:

- dev:
  - packed expected-URL recall improved from `0.8333` to `0.8889`
  - packed claim-evidence recall improved from `0.9722` to `1.0`
  - packed MRR stayed `0.9444`
  - mean retrieval benchmark case time worsened from `1.99s` to `6.16s`
- holdout:
  - packed expected-URL recall stayed `0.7667`
  - packed claim-evidence recall stayed `0.975`
  - packed MRR improved slightly from `0.9067` to `0.9111`
  - mean retrieval benchmark case time worsened from `1.72s` to `4.75s`

## End-to-end Judged Results

Artifacts:

- dev:
  - [`../datasets/runs/baseline_vector_rerank_shortlist_opensearch_20260404_091516_233651_03dc.json`](../datasets/runs/baseline_vector_rerank_shortlist_opensearch_20260404_091516_233651_03dc.json)
- holdout:
  - [`../datasets/runs/baseline_vector_rerank_shortlist_opensearch_20260404_092100_119420_1734.json`](../datasets/runs/baseline_vector_rerank_shortlist_opensearch_20260404_092100_119420_1734.json)

Compared with the current control runs:

- dev:
  - required-claim recall improved from `0.8611` to `0.8889`
  - mean case time worsened from `17.30s` to `58.75s`
  - mean answer-path time worsened from `10.57s` to `27.12s`
- holdout:
  - required-claim recall stayed `0.7750`
  - mean case time worsened from `15.60s` to `31.21s`
  - mean answer-path time worsened from `9.73s` to `22.92s`

## Read

Repo reality:

- the OpenSearch spike is viable as an indexed backend
- retrieval-only behavior is competitive
- end-to-end quality does not beat the current serious control cleanly
- latency is materially worse

Decision:

- do not promote OpenSearch as the new primary backend
- keep the current Elasticsearch/Python/Cohere stack as the control
- if backend alternatives are revisited, OpenSearch should now be treated as
  tested and unconvincing in its bounded like-for-like form

This does **not** prove that OpenSearch can never help. It does prove that a
clean backend swap alone is not the missing win for this repo.
