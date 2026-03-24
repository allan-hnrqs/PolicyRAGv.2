from bgrag.types import EvalCase


def test_eval_case_minimal_schema() -> None:
    case = EvalCase.model_validate(
        {
            "id": "T1",
            "question": "Q?",
            "primary_urls": [],
            "supporting_urls": [],
        }
    )
    assert case.id == "T1"
