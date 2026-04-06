"""Compatibility shim for the canonical benchmark namespace."""

from bgrag.benchmarks.retrieval import *  # noqa: F401,F403
from bgrag.benchmarks.retrieval import _compute_overall_metrics, _first_expected_rank  # noqa: F401
