from pathlib import Path
from types import SimpleNamespace

from bgrag.config import Settings
from bgrag.indexing.embedder import CohereEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeEmbedClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, *, model: str, input_type: str, texts: list[str], embedding_types: list[str]):
        self.calls.append(list(texts))
        return SimpleNamespace(
            embeddings=SimpleNamespace(
                float_=[[float(len(text)), float(index)] for index, text in enumerate(texts)]
            )
        )


def test_embedder_batches_large_requests_and_preserves_order() -> None:
    settings = Settings(
        project_root=REPO_ROOT,
        cohere_api_key="test-key",
        cohere_embed_batch_size=3,
    )
    embedder = CohereEmbedder(settings)
    fake_client = _FakeEmbedClient()
    embedder.client = fake_client

    vectors = embedder.embed_texts(["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"], input_type="search_document")

    assert fake_client.calls == [
        ["a", "bb", "ccc"],
        ["dddd", "eeeee", "ffffff"],
        ["ggggggg"],
    ]
    assert vectors == [
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 2.0],
        [4.0, 0.0],
        [5.0, 1.0],
        [6.0, 2.0],
        [7.0, 0.0],
    ]
