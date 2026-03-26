"""Elasticsearch index helpers."""

from __future__ import annotations

from elasticsearch import Elasticsearch
from elastic_transport import ConnectionError as ElasticConnectionError, ConnectionTimeout

from bgrag.config import Settings
from bgrag.types import ChunkRecord

DEFAULT_BULK_CHUNK_SIZE = 50


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


def ensure_chunk_index(client: Elasticsearch, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        return
    client.indices.create(
        index=index_name,
        settings={
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        mappings={
            "properties": {
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
        },
    )


def _batched_operations(
    operations: list[dict[str, object]],
    batch_size: int,
) -> list[list[dict[str, object]]]:
    return [operations[index : index + batch_size] for index in range(0, len(operations), batch_size)]


def index_chunks(client: Elasticsearch, chunks: list[ChunkRecord], namespace: str) -> None:
    by_family: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_family.setdefault(chunk.source_family.value, []).append(chunk)
    for family, family_chunks in by_family.items():
        index_name = chunk_index_name(family, namespace)
        ensure_chunk_index(client, index_name)
        operations: list[dict[str, object]] = []
        for chunk in family_chunks:
            operations.append({"index": {"_index": index_name, "_id": chunk.chunk_id}})
            operations.append(chunk.model_dump(mode="json"))
        if operations:
            for batch in _batched_operations(operations, DEFAULT_BULK_CHUNK_SIZE * 2):
                client.bulk(operations=batch, refresh=False)
            client.indices.refresh(index=index_name)
