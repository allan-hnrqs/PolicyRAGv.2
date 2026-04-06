"""OpenSearch index helpers for bounded backend spikes."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from bgrag.config import Settings
from bgrag.types import ChunkRecord

DEFAULT_BULK_CHUNK_SIZE = 50
VECTOR_FIELD_NAME = "embedding"


def build_opensearch_client(settings: Settings) -> Any:
    try:
        module = import_module("opensearchpy")
    except ImportError as exc:
        raise RuntimeError(
            "OpenSearch spike support requires `opensearch-py` to be installed in this environment."
        ) from exc
    return module.OpenSearch(settings.opensearch_url, timeout=settings.elastic_request_timeout)


def require_opensearch_available(client: Any, url: str) -> None:
    try:
        if not client.ping():
            raise RuntimeError(f"OpenSearch is not reachable at {url}")
    except Exception as exc:
        raise RuntimeError(f"OpenSearch is not reachable at {url}") from exc


def ensure_chunk_index(client: Any, index_name: str, *, vector_dims: int | None = None) -> None:
    if client.indices.exists(index=index_name):
        return
    properties: dict[str, object] = {
        "chunk_id": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "canonical_url": {"type": "keyword"},
        "title": {"type": "text"},
        "source_family": {"type": "keyword"},
        "authority_rank": {"type": "integer"},
        "chunk_type": {"type": "keyword"},
        "heading": {"type": "text"},
        "heading_path": {"type": "text"},
        "text": {"type": "text"},
        "metadata": {"type": "object", "enabled": True},
    }
    settings_payload: dict[str, object] = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
    if vector_dims:
        settings_payload["index"] = {"knn": True}
        properties[VECTOR_FIELD_NAME] = {
            "type": "knn_vector",
            "dimension": vector_dims,
            "method": {
                "name": "hnsw",
                "engine": "lucene",
                "space_type": "cosinesimil",
            },
        }
    client.indices.create(
        index=index_name,
        body={
            "settings": settings_payload,
            "mappings": {"properties": properties},
        },
    )


def index_chunks(
    client: Any,
    chunks: list[ChunkRecord],
    namespace: str,
    chunk_index_name,
    *,
    embeddings: dict[str, list[float]] | None = None,
) -> None:
    try:
        module = import_module("opensearchpy.helpers")
    except ImportError as exc:
        raise RuntimeError(
            "OpenSearch spike support requires `opensearch-py` to be installed in this environment."
        ) from exc
    by_family: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_family.setdefault(chunk.source_family.value, []).append(chunk)
    vector_dims = len(next(iter(embeddings.values()))) if embeddings else None
    for family, family_chunks in by_family.items():
        index_name = chunk_index_name(family, namespace)
        ensure_chunk_index(client, index_name, vector_dims=vector_dims)
        actions: list[dict[str, object]] = []
        for chunk in family_chunks:
            payload = chunk.model_dump(mode="json")
            if embeddings:
                payload[VECTOR_FIELD_NAME] = embeddings.get(chunk.chunk_id)
            actions.append({"_index": index_name, "_id": chunk.chunk_id, "_source": payload})
        if actions:
            module.bulk(client, actions, chunk_size=DEFAULT_BULK_CHUNK_SIZE, refresh=False)
            client.indices.refresh(index=index_name)
