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


def test_section_chunker_groups_consecutive_short_list_items() -> None:
    document = normalize_document(
        SourceDocument(
            source_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/example-lists",
            fetched_at=datetime.now(timezone.utc),
            html="""
            <html><head><title>Example Lists</title></head><body>
            <main>
              <h1>Types of basis of payment</h1>
              <ul>
                <li>fixed price competitive</li>
                <li>fixed price non-competitive</li>
                <li>provisional price</li>
              </ul>
            </main>
            </body></html>
            """,
        )
    )
    chunks = section_chunker(document)
    assert len(chunks) == 1
    assert "fixed price competitive" in chunks[0].text
    assert "fixed price non-competitive" in chunks[0].text
    assert "provisional price" in chunks[0].text


def test_section_chunker_merges_short_title_like_label_with_following_content() -> None:
    document = normalize_document(
        SourceDocument(
            source_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/example-label",
            fetched_at=datetime.now(timezone.utc),
            html="""
            <html><head><title>Example Label</title></head><body>
            <main>
              <h1>Accessibility</h1>
              <h2>Procurement instruments</h2>
              <p>Performance testing</p>
              <p>Ensure that procurement instruments include accessibility requirements.</p>
            </main>
            </body></html>
            """,
        )
    )
    chunks = section_chunker(document)
    assert len(chunks) == 1
    assert "Performance testing" in chunks[0].text
    assert "accessibility requirements" in chunks[0].text


def test_section_chunker_splits_long_table_row_into_smaller_chunks() -> None:
    document = normalize_document(
        SourceDocument(
            source_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/example-table",
            fetched_at=datetime.now(timezone.utc),
            html="""
            <html><head><title>Example Table</title></head><body>
            <main>
              <h1>Choose the method of supply</h1>
              <table>
                <tr>
                  <th>Methods of supply</th>
                  <th>Contract with TA</th>
                  <th>Standing offer</th>
                  <th>Supply arrangement</th>
                </tr>
                <tr>
                  <td>Contractual obligation</td>
                  <td>A contract with task authorizations creates contractual obligation for the defined contract scope and then authorizes work as needed over time for recurring requirements, recurring work authorizations, recurring deliverables, recurring approvals, and recurring service authorizations.</td>
                  <td>A standing offer is not a contract and only becomes binding when a call-up is issued for a defined requirement, a defined quantity, a defined delivery point, a defined schedule, and a defined authorization under the standing offer terms.</td>
                  <td>A supply arrangement is not a contract and only leads to contractual obligation after a resulting solicitation and contract award for a defined requirement, a defined scope, a defined solicitation process, a defined bid response, and a defined contract award.</td>
                </tr>
              </table>
            </main>
            </body></html>
            """,
        )
    )
    chunks = section_chunker(document)
    assert len(chunks) >= 3
    assert all(len(chunk.text) <= 700 for chunk in chunks)
    assert any("Standing offer:" in chunk.text for chunk in chunks)
    assert any("Supply arrangement:" in chunk.text for chunk in chunks)
