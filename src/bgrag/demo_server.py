"""Local HTTP server for the live frontend demo."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import unquote, urlparse

import cohere
import cohere.types as ct
from dotenv import dotenv_values
from bgrag.config import Settings, detect_project_root
from bgrag.indexing.elastic import build_es_client, require_es_available
from bgrag.manifests import (
    get_active_index_namespace,
    index_embeddings_path,
    index_manifest_path,
)
from bgrag.pipeline import build_answer_callback

DEMO_PROFILE_NAME = "baseline"
PROCUREMENT_INTENT = "procurement_policy"
GREETING_INTENT = "greeting_small_talk"
CAPABILITY_INTENT = "capability_help"
OUT_OF_SCOPE_INTENT = "out_of_scope"
_CALLBACK_CACHE: dict[tuple[str, str, str], Any] = {}
_CALLBACK_LOCK = threading.Lock()


@dataclass(frozen=True)
class DemoHealthStatus:
    ok: bool
    status_message: str
    cohere_configured: bool
    elasticsearch_reachable: bool
    active_index_namespace: str | None
    index_manifest_present: bool
    chunk_embeddings_present: bool


@dataclass(frozen=True)
class DemoServerState:
    settings: Settings
    frontend_dir: Path
    profile_name: str = DEMO_PROFILE_NAME


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    reply_text: str
    response_mode: str


class DemoHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: DemoServerState) -> None:
        super().__init__(server_address, DemoRequestHandler)
        self.state = state


def build_demo_settings(project_root: Path | None = None) -> Settings:
    resolved_root = detect_project_root(project_root or Path.cwd())
    explicit_values = {
        key.lower(): value
        for key, value in dotenv_values(resolved_root / ".env").items()
        if value is not None
    }
    settings = Settings(project_root=resolved_root, **explicit_values)
    settings.ensure_directories()
    return settings


def _extract_text_from_chat_response(response: object) -> str:
    message = getattr(response, "message", None)
    contents = getattr(message, "content", None)
    if not contents:
        return ""
    parts: list[str] = []
    for item in contents:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
    return "".join(parts).strip()


def _intent_gate_prompt(question: str) -> str:
    return (
        "You are the front-door intent gate for PolicyAI, a Canadian procurement policy assistant.\n"
        "Classify the user input and, when needed, draft the assistant's response.\n\n"
        "Return JSON only in this exact shape:\n"
        '{"intent":"procurement_policy","reply_text":"text"}\n\n'
        "Allowed intent values:\n"
        '- "procurement_policy"\n'
        '- "greeting_small_talk"\n'
        '- "capability_help"\n'
        '- "out_of_scope"\n\n'
        "Rules:\n"
        "1. Use procurement_policy only when the user is asking for procurement rules, Buyer’s Guide guidance, solicitation, evaluation, contract management, ACAN, standing offers, bids, offers, suppliers, or related policy workflows.\n"
        "2. Use greeting_small_talk for greetings, thanks, pleasantries, or conversational check-ins.\n"
        "3. Use capability_help for questions like what can you do, help me use this, how should I ask, or how should I structure a procurement question.\n"
        "4. Use out_of_scope for topics outside Canadian procurement policy support.\n"
        "5. If intent is procurement_policy, set reply_text to an empty string.\n"
        "6. If intent is greeting_small_talk, write a short friendly greeting that introduces PolicyAI and invites a procurement policy question.\n"
        "7. If intent is capability_help, explain briefly what PolicyAI can help with, give exactly three example procurement questions, and invite the user to rephrase their question.\n"
        "8. If intent is out_of_scope, politely say the assistant is focused on Canadian procurement policy questions, mention two or three areas it can help with, and invite the user to ask a procurement-focused question instead.\n"
        "9. Keep non-procurement replies concise, polished, and free of citations, markdown headings, or JSON inside the reply text.\n\n"
        f"User input:\n{question}"
    )


def _normalize_intent(raw_intent: object) -> str:
    value = str(raw_intent or "").strip().lower()
    if value in {PROCUREMENT_INTENT, GREETING_INTENT, CAPABILITY_INTENT, OUT_OF_SCOPE_INTENT}:
        return value
    if value in {"greeting", "small_talk", "small-talk"}:
        return GREETING_INTENT
    if value in {"help", "capabilities", "capability"}:
        return CAPABILITY_INTENT
    if value in {"out_of_scope", "out-of-scope", "out_of_domain"}:
        return OUT_OF_SCOPE_INTENT
    return PROCUREMENT_INTENT


def _fallback_intent_reply(intent: str) -> str:
    if intent == GREETING_INTENT:
        return (
            "Hello, I’m PolicyAI. I can help with Canadian procurement policy questions, Buyer’s Guide workflows, "
            "and contract-management guidance.\n\nAsk me a procurement question whenever you’re ready."
        )
    if intent == CAPABILITY_INTENT:
        return (
            "I can help with Canadian procurement policy questions, including solicitation rules, evaluation steps, "
            "ACANs, and contract close-out guidance.\n\nTry asking: When is an ACAN appropriate? How are late offers "
            "handled? What is required before contract close-out?\n\nIf you want, rephrase your question in a specific "
            "procurement scenario and I’ll take it from there."
        )
    return (
        "I’m focused on Canadian procurement policy support. I can help with Buyer’s Guide workflows, solicitation and "
        "evaluation rules, and contract-management questions.\n\nIf you’d like, ask a procurement-focused question and "
        "I’ll help you work through it."
    )


def classify_demo_intent(settings: Settings, question: str) -> IntentDecision:
    settings.require_cohere_key("Intent gating")
    client = cohere.ClientV2(settings.cohere_api_key)
    response = client.chat(
        model=settings.cohere_query_planner_model,
        messages=[ct.UserChatMessageV2(content=_intent_gate_prompt(question))],
        response_format=ct.JsonObjectResponseFormatV2(),
        temperature=0,
        max_tokens=500,
    )
    text = _extract_text_from_chat_response(response)
    parsed = json.loads(text)
    intent = _normalize_intent(parsed.get("intent", PROCUREMENT_INTENT))
    reply_text = str(parsed.get("reply_text", "")).strip()
    if intent == PROCUREMENT_INTENT:
        reply_text = ""
        response_mode = "rag"
    else:
        if not reply_text:
            reply_text = _fallback_intent_reply(intent)
        response_mode = "general_guidance"
    return IntentDecision(intent=intent, reply_text=reply_text, response_mode=response_mode)


def _build_status_message(status: DemoHealthStatus) -> str:
    if not status.cohere_configured:
        return "COHERE_API_KEY is missing from this repo's .env file."
    if not status.elasticsearch_reachable:
        return "Elasticsearch is not reachable at the configured ELASTIC_URL."
    if not status.active_index_namespace:
        return "No active index is configured. Run `bgrag build-index --profile baseline` first."
    if not status.index_manifest_present:
        return f"Index manifest is missing for active namespace `{status.active_index_namespace}`."
    if not status.chunk_embeddings_present:
        return f"Chunk embeddings are missing for active namespace `{status.active_index_namespace}`."
    return "Live backend ready."


def evaluate_demo_health(settings: Settings) -> DemoHealthStatus:
    cohere_configured = settings.has_cohere_key()
    elasticsearch_reachable = False
    active_index_namespace: str | None = None
    index_manifest_present = False
    chunk_embeddings_present = False

    try:
        require_es_available(build_es_client(settings), settings.elastic_url)
        elasticsearch_reachable = True
    except Exception:
        elasticsearch_reachable = False

    try:
        active_index_namespace = get_active_index_namespace(settings)
    except Exception:
        active_index_namespace = None

    if active_index_namespace:
        index_manifest_present = index_manifest_path(settings, active_index_namespace).exists()
        chunk_embeddings_present = index_embeddings_path(settings, active_index_namespace).exists()

    ok = (
        cohere_configured
        and elasticsearch_reachable
        and active_index_namespace is not None
        and index_manifest_present
        and chunk_embeddings_present
    )
    provisional = DemoHealthStatus(
        ok=ok,
        status_message="",
        cohere_configured=cohere_configured,
        elasticsearch_reachable=elasticsearch_reachable,
        active_index_namespace=active_index_namespace,
        index_manifest_present=index_manifest_present,
        chunk_embeddings_present=chunk_embeddings_present,
    )
    return DemoHealthStatus(
        ok=ok,
        status_message=_build_status_message(provisional),
        cohere_configured=cohere_configured,
        elasticsearch_reachable=elasticsearch_reachable,
        active_index_namespace=active_index_namespace,
        index_manifest_present=index_manifest_present,
        chunk_embeddings_present=chunk_embeddings_present,
    )


def _get_answer_callback(settings: Settings, profile_name: str, index_namespace: str):
    cache_key = (str(settings.project_root.resolve()), profile_name, index_namespace)
    with _CALLBACK_LOCK:
        callback = _CALLBACK_CACHE.get(cache_key)
        if callback is None:
            callback = build_answer_callback(settings, profile_name, index_namespace=index_namespace)
            _CALLBACK_CACHE[cache_key] = callback
    return callback


def reset_demo_callback_cache() -> None:
    with _CALLBACK_LOCK:
        _CALLBACK_CACHE.clear()


def run_demo_query(settings: Settings, question: str, profile_name: str = DEMO_PROFILE_NAME) -> dict[str, object]:
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Enter a message before sending.")

    classify_start = perf_counter()
    decision = classify_demo_intent(settings, clean_question)
    classify_end = perf_counter()

    if decision.intent != PROCUREMENT_INTENT:
        return {
            "question": clean_question,
            "answer_text": decision.reply_text,
            "citations": [],
            "timings": {
                "intent_classification_seconds": classify_end - classify_start,
                "total_answer_path_seconds": classify_end - classify_start,
            },
            "profile_name": profile_name,
            "index_namespace": None,
            "notes": [f"intent_gate:{decision.intent}"],
            "intent": decision.intent,
            "response_mode": decision.response_mode,
        }

    health = evaluate_demo_health(settings)
    if not health.ok or not health.active_index_namespace:
        raise RuntimeError(health.status_message)

    class AdHocCase:
        def __init__(self, prompt: str) -> None:
            self.question = prompt

    answer_callback = _get_answer_callback(settings, profile_name, health.active_index_namespace)
    result = answer_callback(AdHocCase(clean_question))
    evidence_notes = [f"intent_gate:{decision.intent}"]
    if result.evidence_bundle is not None:
        evidence_notes.extend(result.evidence_bundle.notes)
    citations = [
        {
            "chunk_id": citation.chunk_id,
            "canonical_url": citation.canonical_url,
            "snippet": citation.snippet,
        }
        for citation in result.citations
    ]
    timings = dict(result.timings)
    timings["intent_classification_seconds"] = classify_end - classify_start
    timings["total_request_seconds"] = (classify_end - classify_start) + float(
        result.timings.get("total_answer_path_seconds", 0.0)
    )
    return {
        "question": clean_question,
        "answer_text": result.answer_text,
        "citations": citations,
        "timings": timings,
        "profile_name": profile_name,
        "index_namespace": health.active_index_namespace,
        "notes": evidence_notes,
        "intent": decision.intent,
        "response_mode": "rag",
    }


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(HTTPStatus.OK, asdict(evaluate_demo_health(self.server.state.settings)))
            return
        self._serve_frontend_asset(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/query":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "status_message": "Unknown endpoint."})
            return

        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json", "status_message": str(exc)})
            return

        question = str(payload.get("question", ""))
        try:
            response_payload = run_demo_query(self.server.state.settings, question, self.server.state.profile_name)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "status_message": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "backend_unavailable", "status_message": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive production path
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "backend_error",
                    "status_message": "The backend could not answer that question right now.",
                    "detail": str(exc),
                },
            )
            return

        self._send_json(HTTPStatus.OK, response_payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _read_json_body(self) -> dict[str, object]:
        content_length_raw = self.headers.get("Content-Length", "").strip()
        if not content_length_raw:
            raise ValueError("Request body is required.")
        try:
            content_length = int(content_length_raw)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if content_length <= 0 or content_length > 65536:
            raise ValueError("Request body must be between 1 byte and 64 KB.")
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_frontend_asset(self, request_path: str) -> None:
        if request_path.startswith("/api/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "status_message": "Unknown endpoint."})
            return
        frontend_dir = self.server.state.frontend_dir.resolve()
        relative = request_path.lstrip("/") or "index.html"
        candidate = (frontend_dir / unquote(relative)).resolve()
        if frontend_dir not in candidate.parents and candidate != frontend_dir:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.exists() or candidate.is_dir():
            candidate = frontend_dir / "index.html"
        self._send_file(candidate)

    def _send_file(self, path: Path) -> None:
        content = path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local live demo server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=4173, type=int)
    parser.add_argument("--profile", default=DEMO_PROFILE_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = build_demo_settings()
    frontend_dir = settings.project_root / "frontend"
    if not frontend_dir.exists():
        raise FileNotFoundError(f"Frontend directory not found: {frontend_dir}")

    server = DemoHTTPServer(
        (args.host, args.port),
        DemoServerState(settings=settings, frontend_dir=frontend_dir, profile_name=args.profile),
    )
    url = f"http://{args.host}:{args.port}"
    print(f"PolicyAI live demo running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
