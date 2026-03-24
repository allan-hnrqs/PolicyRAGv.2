from bgrag.retrieval.query_expansion import normalize_expanded_queries


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
