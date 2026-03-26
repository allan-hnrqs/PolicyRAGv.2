import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bgrag.demo_server as demo_server
from bgrag.config import Settings
from bgrag.manifests import set_active_index_namespace, write_index_manifest
from bgrag.types import AnswerCitation, AnswerResult, EvidenceBundle


def _write_ready_index(settings: Settings, namespace: str = "baseline_ns") -> str:
    set_active_index_namespace(settings, namespace)
    write_index_manifest(settings, namespace, {"namespace": namespace, "chunk_count": 1})
    embeddings_path = settings.project_root / "datasets" / "index" / namespace / "chunk_embeddings.json"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.write_text(json.dumps({"chunk1": [0.1, 0.2]}), encoding="utf-8")
    return namespace


def test_evaluate_demo_health_reports_missing_cohere_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="")
    settings.ensure_directories()
    monkeypatch.setattr(demo_server, "build_es_client", lambda settings: object())
    monkeypatch.setattr(demo_server, "require_es_available", lambda client, url: None)

    status = demo_server.evaluate_demo_health(settings)

    assert status.ok is False
    assert status.cohere_configured is False
    assert status.elasticsearch_reachable is True
    assert status.active_index_namespace is None
    assert "COHERE_API_KEY" in status.status_message


def test_evaluate_demo_health_reports_ready_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    namespace = _write_ready_index(settings)
    monkeypatch.setattr(demo_server, "build_es_client", lambda settings: object())
    monkeypatch.setattr(demo_server, "require_es_available", lambda client, url: None)

    status = demo_server.evaluate_demo_health(settings)

    assert status.ok is True
    assert status.active_index_namespace == namespace
    assert status.index_manifest_present is True
    assert status.chunk_embeddings_present is True
    assert status.status_message == "Live backend ready."


def test_run_demo_query_rejects_empty_question(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()

    with pytest.raises(ValueError, match="Enter a message before sending"):
        demo_server.run_demo_query(settings, "   ")


def test_run_demo_query_returns_general_guidance_without_rag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()

    monkeypatch.setattr(
        demo_server,
        "classify_demo_intent",
        lambda settings, question: demo_server.IntentDecision(
            intent=demo_server.CAPABILITY_INTENT,
            reply_text="I can help with procurement workflows.",
            response_mode="general_guidance",
        ),
    )

    def _fail_build_answer_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("RAG should not run for capability/help messages")

    monkeypatch.setattr(demo_server, "build_answer_callback", _fail_build_answer_callback)

    payload = demo_server.run_demo_query(settings, "What can you do?")

    assert payload["question"] == "What can you do?"
    assert payload["answer_text"] == "I can help with procurement workflows."
    assert payload["citations"] == []
    assert payload["index_namespace"] is None
    assert payload["intent"] == demo_server.CAPABILITY_INTENT
    assert payload["response_mode"] == "general_guidance"
    assert payload["notes"] == ["intent_gate:capability_help"]


def test_run_demo_query_returns_live_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    namespace = _write_ready_index(settings)
    demo_server.reset_demo_callback_cache()

    monkeypatch.setattr(demo_server, "build_es_client", lambda settings: object())
    monkeypatch.setattr(demo_server, "require_es_available", lambda client, url: None)
    monkeypatch.setattr(
        demo_server,
        "classify_demo_intent",
        lambda settings, question: demo_server.IntentDecision(
            intent=demo_server.PROCUREMENT_INTENT,
            reply_text="",
            response_mode="rag",
        ),
    )

    def _fake_build_answer_callback(settings: Settings, profile_name: str, index_namespace: str):
        assert profile_name == "baseline"
        assert index_namespace == namespace

        def _answer(case: SimpleNamespace) -> AnswerResult:
            return AnswerResult(
                question=case.question,
                answer_text="Live backend answer.",
                strategy_name="inline_evidence_chat",
                model_name="command-a-03-2025",
                citations=[
                    AnswerCitation(
                        chunk_id="chunk1",
                        canonical_url="https://example.com/policy",
                        snippet="snippet",
                    )
                ],
                evidence_bundle=EvidenceBundle(
                    query=case.question,
                    retrieval_queries=[case.question],
                    notes=["llm_query_decomposition_applied"],
                ),
                timings={"total_answer_path_seconds": 1.25},
            )

        return _answer

    monkeypatch.setattr(demo_server, "build_answer_callback", _fake_build_answer_callback)

    payload = demo_server.run_demo_query(settings, "What is the rule?")

    assert payload["question"] == "What is the rule?"
    assert payload["answer_text"] == "Live backend answer."
    assert payload["profile_name"] == "baseline"
    assert payload["index_namespace"] == namespace
    assert payload["intent"] == demo_server.PROCUREMENT_INTENT
    assert payload["response_mode"] == "rag"
    assert payload["notes"] == ["intent_gate:procurement_policy", "llm_query_decomposition_applied"]
    assert payload["citations"] == [
        {
            "chunk_id": "chunk1",
            "canonical_url": "https://example.com/policy",
            "snippet": "snippet",
        }
    ]


def test_run_demo_query_strips_rendered_chunk_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    namespace = _write_ready_index(settings)
    demo_server.reset_demo_callback_cache()

    monkeypatch.setattr(demo_server, "build_es_client", lambda settings: object())
    monkeypatch.setattr(demo_server, "require_es_available", lambda client, url: None)
    monkeypatch.setattr(
        demo_server,
        "classify_demo_intent",
        lambda settings, question: demo_server.IntentDecision(
            intent=demo_server.PROCUREMENT_INTENT,
            reply_text="",
            response_mode="rag",
        ),
    )

    def _fake_build_answer_callback(settings: Settings, profile_name: str, index_namespace: str):
        assert profile_name == "baseline"
        assert index_namespace == namespace

        def _answer(case: SimpleNamespace) -> AnswerResult:
            return AnswerResult(
                question=case.question,
                answer_text=(
                    "First supported point [doc__section__15]\n"
                    "Second supported point [doc__section__6, doc__section__47]"
                ),
                strategy_name="inline_evidence_chat",
                model_name="command-a-03-2025",
                citations=[
                    AnswerCitation(
                        chunk_id="doc__section__15",
                        canonical_url="https://example.com/policy",
                        snippet="snippet",
                    )
                ],
                evidence_bundle=EvidenceBundle(
                    query=case.question,
                    retrieval_queries=[case.question],
                ),
            )

        return _answer

    monkeypatch.setattr(demo_server, "build_answer_callback", _fake_build_answer_callback)

    payload = demo_server.run_demo_query(settings, "What is the rule?")

    assert payload["answer_text"] == "First supported point\nSecond supported point"
    assert payload["citations"] == [
        {
            "chunk_id": "doc__section__15",
            "canonical_url": "https://example.com/policy",
            "snippet": "snippet",
        }
    ]
