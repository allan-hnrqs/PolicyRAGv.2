"""Elasticsearch index helpers."""

from __future__ import annotations

from elasticsearch import Elasticsearch
from elastic_transport import ConnectionError as ElasticConnectionError, ConnectionTimeout

from bgrag.config import Settings
from bgrag.types import ChunkRecord

DEFAULT_BULK_CHUNK_SIZE = 50
VECTOR_FIELD_NAME = "embedding"


def chunk_index_name(source_family: str, namespace: str) -> str:
    return f"bgrag_chunks_{namespace}_{source_family}".lower()


def build_es_client(settings: Settings) -> Elasticsearch:
    return Elasticsearch(settings.elastic_url, request_timeout=settings.elastic_request_timeout)


def require_es_available(client: Elasticsearch, url: str) -> None:
    try:
        if not client.ping():
            raise RuntimeError(f"Elasticsearch is not reachable at {url}")
    except (ElasticConnectionError, ConnectionTimeout) as exc:
        raise RuntimeError(f"Elasticsearch is not reachable at {url}") from exc


def ensure_chunk_index(client: Elasticsearch, index_name: str, *, vector_dims: int | None = None) -> None:
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
    if vector_dims:
        properties[VECTOR_FIELD_NAME] = {
            "type": "dense_vector",
            "dims": vector_dims,
            "index": True,
            "similarity": "cosine",
        }
    client.indices.create(
        index=index_name,
        settings={
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        mappings={"properties": properties},
    )


def _batched_operations(
    operations: list[dict[str, object]],
    batch_size: int,
) -> list[list[dict[str, object]]]:
    return [operations[index : index + batch_size] for index in range(0, len(operations), batch_size)]


def index_chunks(
    client: Elasticsearch,
    chunks: list[ChunkRecord],
    namespace: str,
    *,
    embeddings: dict[str, list[float]] | None = None,
) -> None:
    by_family: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_family.setdefault(chunk.source_family.value, []).append(chunk)
    vector_dims = len(next(iter(embeddings.values()))) if embeddings else None
    for family, family_chunks in by_family.items():
        index_name = chunk_index_name(family, namespace)
        ensure_chunk_index(client, index_name, vector_dims=vector_dims)
        operations: list[dict[str, object]] = []
        for chunk in family_chunks:
            operations.append({"index": {"_index": index_name, "_id": chunk.chunk_id}})
            payload = chunk.model_dump(mode="json")
            if embeddings:
                payload[VECTOR_FIELD_NAME] = embeddings.get(chunk.chunk_id)
            operations.append(payload)
        if operations:
            for batch in _batched_operations(operations, DEFAULT_BULK_CHUNK_SIZE * 2):
                client.bulk(operations=batch, refresh=False)
            client.indices.refresh(index=index_name)
