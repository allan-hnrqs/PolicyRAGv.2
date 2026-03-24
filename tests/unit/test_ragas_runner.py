import math
from pathlib import Path

from bgrag.config import Settings
from bgrag.eval import ragas_runner
from bgrag.types import AnswerResult, ChunkRecord, EvidenceBundle, EvalCase, RagasCaseResult, SourceFamily


def _sample_chunk() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        canonical_url="https://example.com/page",
        title="Example",
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=1,
        chunker_name="section_chunker",
        chunk_type="section",
        text="Example supporting content.",
        heading="Heading",
        heading_path=["Parent", "Heading"],
        order=0,
        token_estimate=10,
    )


def _sample_answer() -> AnswerResult:
    chunk = _sample_chunk()
    return AnswerResult(
        question="Q?",
        answer_text="A.",
        strategy_name="inline_evidence_chat",
        model_name="command-a-03-2025",
        evidence_bundle=EvidenceBundle(
            query="Q?",
            packed_chunks=[chunk],
            candidates=[],
        ),
    )


def test_build_dataset_row_uses_reference_and_packed_context() -> None:
    case = EvalCase.model_validate(
        {
            "id": "T1",
            "question": "Q?",
            "reference_answer": "A.",
        }
    )

    row = ragas_runner._build_dataset_row(case, _sample_answer())

    assert row is not None
    assert row["user_input"] == "Q?"
    assert row["reference"] == "A."
    assert isinstance(row["retrieved_contexts"], list)
    assert "URL: https://example.com/page" in row["retrieved_contexts"][0]


def test_build_dataset_row_skips_missing_reference_or_context() -> None:
    case_without_reference = EvalCase.model_validate({"id": "T1", "question": "Q?"})
    answer = _sample_answer()

    assert ragas_runner._build_dataset_row(case_without_reference, answer) is None

    answer_without_context = answer.model_copy(
        update={"evidence_bundle": EvidenceBundle(query="Q?", packed_chunks=[], candidates=[])}
    )
    case_with_reference = EvalCase.model_validate({"id": "T2", "question": "Q?", "reference_answer": "A."})
    assert ragas_runner._build_dataset_row(case_with_reference, answer_without_context) is None


def test_compute_overall_metrics_averages_only_evaluated_cases() -> None:
    results = [
        RagasCaseResult(
            case_id="A",
            question="Q1",
            answer_strategy="baseline",
            answer_text="A1",
            evaluated=True,
            metrics={"faithfulness": 0.5, "coverage_recall": 1.0},
        ),
        RagasCaseResult(
            case_id="B",
            question="Q2",
            answer_strategy="baseline",
            answer_text="A2",
            evaluated=True,
            metrics={"faithfulness": 1.0, "coverage_recall": 0.0},
        ),
        RagasCaseResult(
            case_id="C",
            question="Q3",
            answer_strategy="baseline",
            answer_text="A3",
            evaluated=False,
            skip_reason="reference_answer_missing",
        ),
    ]

    overall = ragas_runner._compute_overall_metrics(results)

    assert overall["case_count"] == 3
    assert overall["evaluated_case_count"] == 2
    assert overall["skipped_case_count"] == 1
    assert overall["faithfulness_mean"] == 0.75
    assert overall["coverage_recall_mean"] == 0.5


def test_compute_overall_metrics_ignores_nan_values() -> None:
    results = [
        RagasCaseResult(
            case_id="A",
            question="Q1",
            answer_strategy="baseline",
            answer_text="A1",
            evaluated=True,
            metrics={"faithfulness": 0.5, "correctness_precision": math.nan},
        ),
        RagasCaseResult(
            case_id="B",
            question="Q2",
            answer_strategy="baseline",
            answer_text="A2",
            evaluated=True,
            metrics={"faithfulness": math.nan, "correctness_precision": 1.0},
        ),
    ]

    overall = ragas_runner._compute_overall_metrics(results)

    assert overall["faithfulness_mean"] == 0.5
    assert overall["correctness_precision_mean"] == 1.0


def test_normalize_metric_value_converts_nan_to_none() -> None:
    assert ragas_runner._normalize_metric_value(math.nan) is None


def test_build_ragas_llm_uses_cache_and_removes_top_p(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeDiskCacheBackend:
        def __init__(self, cache_dir: str) -> None:
            captured["cache_dir"] = cache_dir

    class FakeInstructorLLM:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs
            self.model_args = {"temperature": 0.0, "top_p": 0.1, "max_tokens": 3000}

    monkeypatch.setattr(ragas_runner, "DiskCacheBackend", FakeDiskCacheBackend)
    monkeypatch.setattr(ragas_runner, "InstructorLLM", FakeInstructorLLM)
    monkeypatch.setattr(ragas_runner, "from_cohere", lambda client: "patched-client")
    monkeypatch.setattr(ragas_runner.cohere, "ClientV2", lambda api_key: {"api_key": api_key})

    settings = Settings(
        project_root=tmp_path,
        cache_dir=Path(".cache"),
        cohere_api_key="test-key",
        ragas_max_output_tokens=4321,
    )

    llm = ragas_runner._build_ragas_llm(settings)

    assert llm.model_args["max_tokens"] == 3000
    assert "top_p" not in llm.model_args
    assert str(tmp_path / ".cache" / "ragas") == captured["cache_dir"]
    kwargs = captured["kwargs"]
    assert kwargs["model"] == settings.cohere_judge_model
    assert kwargs["provider"] == "cohere"
