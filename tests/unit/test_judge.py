import json

from bgrag.eval.judge import _normalize_judgment
from bgrag.types import EvalCase


def test_normalize_judgment_computes_metrics() -> None:
    case = EvalCase.model_validate(
        {
            "id": "T1",
            "question": "Q?",
            "primary_urls": [],
            "supporting_urls": [],
            "required_claims": ["A", "B"],
            "forbidden_claims": ["X"],
        }
    )
    parsed = json.loads(
        """
        {
          "required_claims": [
            {"claim": "A", "supported": true, "reason": "present"},
            {"claim": "B", "supported": false, "reason": "missing"}
          ],
          "forbidden_claims": [
            {"claim": "X", "violated": false, "reason": "not present"}
          ],
          "answer_abstains": false,
          "abstain_correct": null,
          "overall_notes": "ok"
        }
        """
    )

    judgment = _normalize_judgment(parsed, case)

    assert judgment["required_claim_recall"] == 0.5
    assert judgment["required_claim_supported_count"] == 1
    assert judgment["forbidden_claim_violation_count"] == 0
    assert judgment["forbidden_claims_clean"] is True
