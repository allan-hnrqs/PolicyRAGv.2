import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bgrag.demo_server as demo_server
from bgrag.config import Settings
from bgrag.manifests import derive_index_namespace, set_active_index_namespace, write_index_manifest
from bgrag.types import AnswerCitation, AnswerResult, EvidenceBundle


def _write_ready_index(
    settings: Settings,
    namespace: str | None = None,
    *,
    profile_name: str = demo_server.DEMO_PROFILE_NAME,
    search_backend: str = "elasticsearch",
) -> str:
    resolved_namespace = namespace or f"{profile_name}_ns"
    set_active_index_namespace(settings, resolved_namespace)
    write_index_manifest(
        settings,
        resolved_namespace,
        {
            "namespace": resolved_namespace,
            "chunk_count": 1,
            "profile_name": profile_name,
            "search_backend": search_backend,
        },
    )
    embeddings_path = settings.project_root / "datasets" / "index" / resolved_namespace / "chunk_embeddings.json"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.write_text(json.dumps({"chunk1": [0.1, 0.2]}), encoding="utf-8")
    return resolved_namespace


def _patch_demo_profile(monkeypatch: pytest.MonkeyPatch, *, search_backend: str = "elasticsearch") -> None:
    profile = SimpleNamespace(
        retrieval=SimpleNamespace(
            retrieval_mode="hybrid_es_rerank",
            dense_retrieval_backend="elasticsearch_knn",
            enable_mmr_diversity=False,
            search_backend=search_backend,
        ),
    )
    monkeypatch.setattr(demo_server, "load_profile", lambda profile_name, settings: profile)
    monkeypatch.setattr(demo_server, "build_runtime_settings", lambda settings, profile: settings)
    monkeypatch.setattr(demo_server, "derive_index_namespace", lambda settings, profile_name: f"{profile_name}_ns")


def test_evaluate_demo_health_reports_missing_cohere_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="")
    settings.ensure_directories()
    _patch_demo_profile(monkeypatch)
    monkeypatch.setattr(demo_server, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(demo_server, "require_search_available", lambda client, settings, backend: None)

    status = demo_server.evaluate_demo_health(settings)

    assert status.ok is False
    assert status.cohere_configured is False
    assert status.search_backend == "elasticsearch"
    assert status.search_backend_reachable is True
    assert status.active_index_namespace is None
    assert "COHERE_API_KEY" in status.status_message


def test_evaluate_demo_health_reports_ready_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    _patch_demo_profile(monkeypatch)
    namespace = _write_ready_index(settings)
    monkeypatch.setattr(demo_server, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(demo_server, "require_search_available", lambda client, settings, backend: None)

    status = demo_server.evaluate_demo_health(settings)

    assert status.ok is True
    assert status.search_backend == "elasticsearch"
    assert status.active_index_namespace == namespace
    assert status.index_manifest_present is True
    assert status.chunk_embeddings_present is True
    assert status.status_message == "Live backend ready."


def test_evaluate_demo_health_prefers_profile_matching_namespace_over_active_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    _patch_demo_profile(monkeypatch, search_backend="opensearch")
    active_namespace = _write_ready_index(settings, namespace="elastic_active", profile_name="baseline_vector_rerank_shortlist")
    _write_ready_index(
        settings,
        profile_name=demo_server.DEMO_PROFILE_NAME,
        search_backend="opensearch",
    )
    set_active_index_namespace(settings, active_namespace)
    monkeypatch.setattr(demo_server, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(demo_server, "require_search_available", lambda client, settings, backend: None)

    status = demo_server.evaluate_demo_health(settings)

    assert status.ok is True
    assert status.search_backend == "opensearch"
    assert status.active_index_namespace == f"{demo_server.DEMO_PROFILE_NAME}_ns"


def test_run_demo_query_rejects_empty_question(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()

    with pytest.raises(ValueError, match="Enter a message before sending"):
        demo_server.run_demo_query(settings, "   ")


def test_run_demo_query_uses_direct_question_without_contextualizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    namespace = _write_ready_index(settings)
    demo_server.reset_demo_callback_cache()
    _patch_demo_profile(monkeypatch)

    monkeypatch.setattr(demo_server, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(demo_server, "require_search_available", lambda client, settings, backend: None)
    monkeypatch.setattr(
        demo_server,
        "contextualize_conversation_turn",
        lambda settings, messages: (_ for _ in ()).throw(
            AssertionError("Contextualizer should not run for first-turn standalone questions")
        ),
    )

    def _fake_build_answer_callback(settings: Settings, profile_name: str, index_namespace: str):
        assert profile_name == demo_server.DEMO_PROFILE_NAME
        assert index_namespace == namespace

        def _answer(case: SimpleNamespace) -> AnswerResult:
            assert case.question == "How are late offers handled?"
            return AnswerResult(
                question=case.question,
                answer_text="Late offer answer.",
                strategy_name="inline_evidence_chat",
                model_name="command-a-03-2025",
                citations=[],
                evidence_bundle=EvidenceBundle(query=case.question, retrieval_queries=[case.question]),
            )

        return _answer

    monkeypatch.setattr(demo_server, "build_answer_callback", _fake_build_answer_callback)

    payload = demo_server.run_demo_query(settings, "How are late offers handled?")

    assert payload["intent"] == demo_server.PROCUREMENT_INTENT
    assert payload["response_mode"] == "rag"
    assert payload["resolved_question"] == "How are late offers handled?"
    assert payload["notes"] == ["conversation_route:direct_question"]
    assert payload["timings"]["contextualization_seconds"] >= 0.0


def test_run_demo_query_returns_live_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    namespace = _write_ready_index(settings)
    demo_server.reset_demo_callback_cache()
    _patch_demo_profile(monkeypatch)

    monkeypatch.setattr(demo_server, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(demo_server, "require_search_available", lambda client, settings, backend: None)
    monkeypatch.setattr(
        demo_server,
        "contextualize_conversation_turn",
        lambda settings, messages: (_ for _ in ()).throw(
            AssertionError("Contextualizer should not run without prior history")
        ),
    )

    def _fake_build_answer_callback(settings: Settings, profile_name: str, index_namespace: str):
        assert profile_name == demo_server.DEMO_PROFILE_NAME
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
    assert payload["resolved_question"] == "What is the rule?"
    assert payload["answer_text"] == "Live backend answer."
    assert payload["profile_name"] == demo_server.DEMO_PROFILE_NAME
    assert payload["index_namespace"] == namespace
    assert payload["intent"] == demo_server.PROCUREMENT_INTENT
    assert payload["response_mode"] == "rag"
    assert payload["notes"] == ["conversation_route:direct_question", "llm_query_decomposition_applied"]
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
    _patch_demo_profile(monkeypatch)

    monkeypatch.setattr(demo_server, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(demo_server, "require_search_available", lambda client, settings, backend: None)
    monkeypatch.setattr(
        demo_server,
        "contextualize_conversation_turn",
        lambda settings, messages: (_ for _ in ()).throw(
            AssertionError("Contextualizer should not run without prior history")
        ),
    )

    def _fake_build_answer_callback(settings: Settings, profile_name: str, index_namespace: str):
        assert profile_name == demo_server.DEMO_PROFILE_NAME
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


def test_run_demo_query_uses_contextualized_standalone_question_for_rag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    namespace = _write_ready_index(settings)
    demo_server.reset_demo_callback_cache()
    _patch_demo_profile(monkeypatch)

    monkeypatch.setattr(demo_server, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(demo_server, "require_search_available", lambda client, settings, backend: None)

    captured_messages: list[dict[str, str]] = []

    def _fake_contextualize(settings: Settings, messages) -> str:
        captured_messages.extend(messages)
        return "What is the minimum solicitation period for RFSAs?"

    monkeypatch.setattr(demo_server, "contextualize_conversation_turn", _fake_contextualize)

    def _fake_build_answer_callback(settings: Settings, profile_name: str, index_namespace: str):
        assert profile_name == demo_server.DEMO_PROFILE_NAME
        assert index_namespace == namespace

        def _answer(case: SimpleNamespace) -> AnswerResult:
            assert case.question == "What is the minimum solicitation period for RFSAs?"
            return AnswerResult(
                question=case.question,
                answer_text="Use the RFSA solicitation-period rule.",
                strategy_name="inline_evidence_chat",
                model_name="command-a-03-2025",
                citations=[],
                evidence_bundle=EvidenceBundle(
                    query=case.question,
                    retrieval_queries=[case.question],
                ),
            )

        return _answer

    monkeypatch.setattr(demo_server, "build_answer_callback", _fake_build_answer_callback)

    payload = demo_server.run_demo_query(
        settings,
        "What about for RFSAs?",
        messages=[
            {"role": "user", "content": "What is the minimum solicitation period?"},
            {"role": "assistant", "content": "It depends on the solicitation method."},
            {"role": "user", "content": "What about for RFSAs?"},
        ],
    )

    assert payload["question"] == "What about for RFSAs?"
    assert payload["resolved_question"] == "What is the minimum solicitation period for RFSAs?"
    assert payload["notes"] == ["conversation_route:history_contextualizer", "conversation_context_applied"]
    assert captured_messages[-1] == {"role": "user", "text": "What about for RFSAs?"}


def test_run_demo_query_accepts_cohere_style_content_segments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    namespace = _write_ready_index(settings)
    demo_server.reset_demo_callback_cache()
    _patch_demo_profile(monkeypatch)

    monkeypatch.setattr(demo_server, "build_search_client", lambda settings, backend: object())
    monkeypatch.setattr(demo_server, "require_search_available", lambda client, settings, backend: None)

    captured_messages: list[dict[str, str]] = []

    def _fake_contextualize(settings: Settings, messages) -> str:
        captured_messages.extend(messages)
        return "When is Canada legally bound under a standing offer and under a supply arrangement?"

    monkeypatch.setattr(demo_server, "contextualize_conversation_turn", _fake_contextualize)

    def _fake_build_answer_callback(settings: Settings, profile_name: str, index_namespace: str):
        assert profile_name == demo_server.DEMO_PROFILE_NAME
        assert index_namespace == namespace

        def _answer(case: SimpleNamespace) -> AnswerResult:
            assert case.question == "When is Canada legally bound under a standing offer and under a supply arrangement?"
            return AnswerResult(
                question=case.question,
                answer_text="Standing offer call-ups and supply arrangement contract awards bind Canada.",
                strategy_name="inline_evidence_chat",
                model_name="command-a-03-2025",
                citations=[],
                evidence_bundle=EvidenceBundle(
                    query=case.question,
                    retrieval_queries=[case.question],
                ),
            )

        return _answer

    monkeypatch.setattr(demo_server, "build_answer_callback", _fake_build_answer_callback)

    payload = demo_server.run_demo_query(
        settings,
        "What about under each one, when is Canada legally bound?",
        messages=[
            {"role": "user", "content": "What is the contractual difference between a standing offer and a supply arrangement?"},
            {
                "role": "assistant",
                "content": [{"text": "A standing offer is not a contract; each call-up creates one."}],
            },
            {"role": "user", "content": "What about under each one, when is Canada legally bound?"},
        ],
    )

    assert payload["resolved_question"] == (
        "When is Canada legally bound under a standing offer and under a supply arrangement?"
    )
    assert payload["notes"] == ["conversation_route:history_contextualizer", "conversation_context_applied"]
    assert captured_messages[1] == {
        "role": "assistant",
        "text": "A standing offer is not a contract; each call-up creates one.",
    }


def test_contextualize_conversation_turn_uses_chat_history_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(project_root=tmp_path, cohere_api_key="test-key")
    settings.ensure_directories()
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

        def chat(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                message=SimpleNamespace(
                    content=[SimpleNamespace(text='{"standalone_question":"Resolved procurement question"}')]
                )
            )

    monkeypatch.setattr(demo_server.cohere, "ClientV2", _FakeClient)

    standalone = demo_server.contextualize_conversation_turn(
        settings,
        [
            {"role": "user", "text": "What is the contractual difference between a standing offer and a supply arrangement?"},
            {"role": "assistant", "text": "They differ in when Canada becomes legally bound."},
            {"role": "user", "text": "What about under each one?"},
        ],
    )

    sent_messages = captured["messages"]
    assert [message.role for message in sent_messages] == ["system", "user", "assistant", "user"]
    assert sent_messages[-1].content == "What about under each one?"
    assert standalone == "Resolved procurement question"
