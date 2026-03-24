from datetime import datetime, timezone

from bgrag.chunking.chunkers import block_chunker, section_chunker, sliding_window_chunker
from bgrag.normalize.normalizer import normalize_document
from bgrag.types import SourceDocument


def _doc():
    return normalize_document(
        SourceDocument(
            source_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/example",
            fetched_at=datetime.now(timezone.utc),
            html="""
            <html><head><title>Example</title></head><body>
            <h1>Title</h1>
            <p>Alpha paragraph.</p>
            <p>Beta paragraph.</p>
            <h2>Next</h2>
            <p>Gamma paragraph.</p>
            </body></html>
            """,
        )
    )


def test_section_chunker() -> None:
    chunks = section_chunker(_doc(), enrichers=["authority_metadata"])
    assert chunks
    assert all(chunk.chunker_name == "section_chunker" for chunk in chunks)


def test_block_chunker() -> None:
    chunks = block_chunker(_doc())
    assert chunks
    assert any(chunk.chunk_type == "block_group" for chunk in chunks)


def test_sliding_window_chunker() -> None:
    chunks = sliding_window_chunker(_doc(), window_chars=20, overlap_chars=5)
    assert len(chunks) >= 2
