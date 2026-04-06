from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bgrag.profiles.loader import load_profile
from bgrag.config import Settings
from bgrag.retrieval_budget_sweep import (
    RetrievalBudgetSweepRun,
    RetrievalBudgetSweepVariantResult,
    RetrievalBudgetVariant,
    build_variant_profile,
    default_budget_variants,
    render_retrieval_budget_sweep_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_budget_variants_include_current_shortlist_shape() -> None:
    variants = default_budget_variants()

    assert any(
        variant.top_k == 16
        and variant.candidate_k == 48
        and variant.rerank_top_n == 48
        and variant.per_query_candidate_k == 24
        for variant in variants
    )


def test_build_variant_profile_overrides_retrieval_budget_fields() -> None:
    settings = Settings(project_root=REPO_ROOT)
    base_profile = load_profile("baseline_vector_rerank_shortlist", settings)
    variant = RetrievalBudgetVariant(
        label="top24_c64_r64_p32",
        top_k=24,
        candidate_k=64,
        rerank_top_n=64,
        per_query_candidate_k=32,
    )

    profile = build_variant_profile(base_profile, variant)

    assert profile.name.endswith("top24_c64_r64_p32")
    assert profile.retrieval.top_k == 24
    assert profile.retrieval.candidate_k == 64
    assert profile.retrieval.rerank_top_n == 64
    assert profile.retrieval.per_query_candidate_k == 32


def test_render_retrieval_budget_sweep_markdown_lists_variants() -> None:
    run = RetrievalBudgetSweepRun(
        run_name="budget_sweep_123",
        created_at=datetime.now(timezone.utc),
        base_profile_name="baseline_vector_rerank_shortlist",
        eval_path="datasets/eval/parity/parity19.jsonl",
        query_mode="profile",
        variants=[
            RetrievalBudgetSweepVariantResult(
                label="top16_c48_r48_p24",
                effective_profile_name="baseline_vector_rerank_shortlist_top16_c48_r48_p24",
                retrieval_overrides={
                    "top_k": 16,
                    "candidate_k": 48,
                    "rerank_top_n": 48,
                    "per_query_candidate_k": 24,
                    "max_expanded_queries": 2,
                },
                retrieval_json_path="retrieval.json",
                retrieval_markdown_path="retrieval.md",
                eval_json_path="eval.json",
                eval_overall_metrics={
                    "required_claim_recall_mean": 0.82,
                    "mean_case_seconds": 22.1,
                },
                retrieval_overall_metrics={
                    "packed_expected_url_recall_mean": 0.77,
                    "packed_claim_evidence_recall_mean_annotated": 0.95,
                },
            )
        ],
        summary={
            "variant_count": 1,
            "best_quality_label": "top16_c48_r48_p24",
            "best_quality_required_claim_recall_mean": 0.82,
            "fastest_label": "top16_c48_r48_p24",
            "fastest_mean_case_seconds": 22.1,
            "best_retrieval_label": "top16_c48_r48_p24",
            "best_retrieval_packed_expected_url_recall_mean": 0.77,
        },
    )

    markdown = render_retrieval_budget_sweep_markdown(run)

    assert "# Retrieval Budget Sweep" in markdown
    assert "top16_c48_r48_p24" in markdown
    assert "eval_required_claim_recall_mean: 0.82" in markdown
