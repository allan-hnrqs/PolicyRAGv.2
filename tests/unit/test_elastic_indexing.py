from bgrag.indexing.elastic import DEFAULT_BULK_CHUNK_SIZE, index_chunks
from bgrag.types import ChunkRecord, SourceFamily


class _FakeIndices:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.settings_by_index: dict[str, dict[str, object]] = {}
        self.refreshed: list[str] = []

    def exists(self, index: str) -> bool:
        return index in self.created

    def create(self, index: str, mappings: dict[str, object], settings: dict[str, object]) -> None:
        del mappings
        self.created.append(index)
        self.settings_by_index[index] = dict(settings)

    def refresh(self, index: str) -> None:
        self.refreshed.append(index)


class _FakeClient:
    def __init__(self) -> None:
        self.indices = _FakeIndices()
        self.bulk_batches: list[list[dict[str, object]]] = []

    def bulk(self, operations: list[dict[str, object]], refresh: bool) -> None:
        assert refresh is False
        self.bulk_batches.append(list(operations))


def _chunk(chunk_id: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        canonical_url=f"https://example.com/{chunk_id}",
        title=chunk_id,
        source_family=SourceFamily.BUYERS_GUIDE,
        authority_rank=3,
        chunker_name="section_chunker",
        chunk_type="paragraph",
        text="Example text.",
    )


def test_index_chunks_batches_bulk_operations() -> None:
    client = _FakeClient()
    chunks = [_chunk(f"chunk_{index}") for index in range(DEFAULT_BULK_CHUNK_SIZE + 1)]

    index_chunks(client, chunks, namespace="baseline_ns")

    assert len(client.bulk_batches) == 2
    assert len(client.bulk_batches[0]) == DEFAULT_BULK_CHUNK_SIZE * 2
    assert len(client.bulk_batches[1]) == 2
    assert client.indices.settings_by_index["bgrag_chunks_baseline_ns_buyers_guide"] == {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
    assert client.indices.refreshed == ["bgrag_chunks_baseline_ns_buyers_guide"]
