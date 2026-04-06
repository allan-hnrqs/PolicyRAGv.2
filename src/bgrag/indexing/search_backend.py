"""Search backend helpers for reversible retrieval/indexing spikes."""

from __future__ import annotations

from typing import Any

from bgrag.config import Settings
from bgrag.indexing import elastic as elastic_backend
from bgrag.indexing import opensearch_backend
from bgrag.types import ChunkRecord

SEARCH_BACKEND_ELASTICSEARCH = "elasticsearch"
SEARCH_BACKEND_OPENSEARCH = "opensearch"
SUPPORTED_SEARCH_BACKENDS = {
    SEARCH_BACKEND_ELASTICSEARCH,
    SEARCH_BACKEND_OPENSEARCH,
}


def normalize_search_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SEARCH_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_SEARCH_BACKENDS))
        raise ValueError(f"Unsupported search backend `{value}`. Expected one of: {supported}.")
    return normalized


def search_backend_label(backend: str) -> str:
    normalized = normalize_search_backend(backend)
    if normalized == SEARCH_BACKEND_OPENSEARCH:
        return "OpenSearch"
    return "Elasticsearch"


def search_backend_url(settings: Settings, backend: str) -> str:
    normalized = normalize_search_backend(backend)
    if normalized == SEARCH_BACKEND_OPENSEARCH:
        return settings.opensearch_url
    return settings.elastic_url


def build_search_client(settings: Settings, backend: str) -> Any:
    normalized = normalize_search_backend(backend)
    if normalized == SEARCH_BACKEND_ELASTICSEARCH:
        return elastic_backend.build_es_client(settings)
    return opensearch_backend.build_opensearch_client(settings)


def require_search_available(client: Any, settings: Settings, backend: str) -> None:
    normalized = normalize_search_backend(backend)
    url = search_backend_url(settings, normalized)
    if normalized == SEARCH_BACKEND_ELASTICSEARCH:
        elastic_backend.require_es_available(client, url)
        return
    opensearch_backend.require_opensearch_available(client, url)


def index_chunks_for_backend(
    client: Any,
    chunks: list[ChunkRecord],
    namespace: str,
    *,
    embeddings: dict[str, list[float]] | None = None,
    backend: str,
) -> None:
    normalized = normalize_search_backend(backend)
    if normalized == SEARCH_BACKEND_ELASTICSEARCH:
        elastic_backend.index_chunks(client, chunks, namespace, embeddings=embeddings)
        return
    opensearch_backend.index_chunks(
        client,
        chunks,
        namespace,
        elastic_backend.chunk_index_name,
        embeddings=embeddings,
    )
