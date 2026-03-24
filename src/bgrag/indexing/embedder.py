"""Embedding client wrappers and embedding-store persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cohere

from bgrag.config import Settings


class CohereEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = cohere.ClientV2(settings.cohere_api_key)

    def embed_texts(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        batch_size = max(1, self.settings.cohere_embed_batch_size)
        all_vectors: list[list[float]] = []
        items = list(texts)
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            response = self.client.embed(
                model=self.settings.cohere_embed_model,
                input_type=input_type,
                texts=batch,
                embedding_types=["float"],
            )
            all_vectors.extend(list(row) for row in response.embeddings.float_)
        return all_vectors


def write_embedding_store(path: Path, vectors: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vectors), encoding="utf-8")


def read_embedding_store(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
