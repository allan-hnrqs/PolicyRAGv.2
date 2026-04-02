from bgrag.retrieval.query_expansion import (
    _planner_prompt,
    normalize_expanded_queries,
    parse_query_plan,
)


def test_normalize_expanded_queries_deduplicates_and_limits() -> None:
    result = normalize_expanded_queries(
        "Original procurement question",
        [
            "Original procurement question",
            " standing offer legal obligation ",
            "standing offer legal obligation",
            "supply arrangement contract formation",
            "extra query that should be dropped",
        ],
        max_expanded_queries=2,
    )

    assert result == [
        "standing offer legal obligation",
        "supply arrangement contract formation",
    ]


def test_parse_query_plan_skips_expansion_when_planner_says_not_to_decompose() -> None:
    result = parse_query_plan(
        "What is the minimum solicitation period?",
        {
            "should_decompose": False,
            "queries": [
                "minimum solicitation period",
                "solicitation time requirements",
            ],
        },
        max_expanded_queries=2,
    )

    assert result == []


def test_parse_query_plan_remains_backward_compatible_with_old_shape() -> None:
    result = parse_query_plan(
        "Original procurement question",
        {
            "queries": [
                "Original procurement question",
                "standing offer legal obligation",
                "supply arrangement contract formation",
            ]
        },
        max_expanded_queries=2,
    )

    assert result == [
        "standing offer legal obligation",
        "supply arrangement contract formation",
    ]


def test_planner_prompt_keeps_general_decomposition_bias_without_exact_identifier_special_case() -> None:
    prompt = _planner_prompt(
        "What exact form number do I have to use to get ADM approval for a reciprocal procurement exception?",
        2,
    )

    assert "Prefer not decomposing unless the question clearly needs multiple distinct retrieval aspects." in prompt
    assert "governing policy/exception/approval requirement" not in prompt
