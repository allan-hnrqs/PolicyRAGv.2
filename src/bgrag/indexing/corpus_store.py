"""Corpus persistence helpers."""

from __future__ import annotations

from pathlib import Path

from bgrag.types import ChunkRecord, NormalizedDocument


def write_normalized_documents(output_dir: Path, documents: list[NormalizedDocument]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.json"):
        existing.unlink()
    for document in documents:
        path = output_dir / f"{document.doc_id}.json"
        path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def read_normalized_documents(input_dir: Path) -> list[NormalizedDocument]:
    return [
        NormalizedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(input_dir.glob("*.json"))
    ]


def write_chunks(output_path: Path, chunks: list[ChunkRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")


def read_chunks(input_path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(ChunkRecord.model_validate_json(line))
    return chunks
