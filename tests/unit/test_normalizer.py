from datetime import datetime, timezone

from bgrag.normalize.normalizer import html_to_text_blocks, normalize_document, trim_buyers_guide_chrome
from bgrag.types import SourceDocument, StructureBlock


def _block(order: int, block_type: str, text: str) -> StructureBlock:
    return StructureBlock(
        block_id=f"block_{order:04d}",
        block_type=block_type,
        heading=text if block_type == "heading" else "Heading",
        heading_path=[text] if block_type == "heading" else ["Heading"],
        text=text,
        order=order,
    )


def test_trim_buyers_guide_chrome_drops_title_nav_and_keeps_first_real_content() -> None:
    blocks = [
        _block(0, "heading", "SAP Ariba system maintenance"),
        _block(1, "paragraph", "maintenance text"),
        _block(2, "heading", "Buyer's Guide Receive offers - Handle late or delayed offers"),
        _block(3, "list_item", "Plan"),
        _block(4, "list_item", "Receive and evaluate"),
        _block(5, "paragraph", "Handle late or delayed offers introduction."),
        _block(6, "heading", "Late and delayed offer submission"),
        _block(7, "list_item", "The closing date and time are firm."),
    ]

    trimmed = trim_buyers_guide_chrome(blocks)

    assert [block.text for block in trimmed] == [
        "Handle late or delayed offers introduction.",
        "Late and delayed offer submission",
        "The closing date and time are firm.",
    ]


def test_html_to_text_blocks_skips_malformed_parent_paragraph_and_keeps_children() -> None:
    html = """
    <html>
      <body>
        <main>
          <div class="field--name-field-main-content">
            <div class="field--item">
              <p>
                <h2>Choose the method of supply</h2>
                <p>You can use contracts or standing offers.</p>
                <ul>
                  <li>contracts</li>
                  <li>standing offers</li>
                </ul>
              </p>
            </div>
          </div>
        </main>
      </body>
    </html>
    """
    _, blocks, _ = html_to_text_blocks(html)
    assert [block.block_type for block in blocks] == ["heading", "paragraph", "list_item", "list_item"]
    assert [block.text for block in blocks] == [
        "Choose the method of supply",
        "You can use contracts or standing offers.",
        "contracts",
        "standing offers",
    ]


def test_html_to_text_blocks_keeps_parent_list_text_without_child_list_duplication() -> None:
    html = """
    <html>
      <body>
        <main>
          <ul>
            <li>
              Define the requirement
              <ul>
                <li>Requisition</li>
                <li>Standards and quality assurance</li>
              </ul>
            </li>
          </ul>
        </main>
      </body>
    </html>
    """
    _, blocks, _ = html_to_text_blocks(html)
    assert [block.text for block in blocks] == [
        "Define the requirement",
        "Requisition",
        "Standards and quality assurance",
    ]


def test_html_to_text_blocks_keeps_nested_list_item_label_without_absorbing_block_children() -> None:
    html = """
    <html>
      <body>
        <main>
          <ul>
            <li>
              <span>A.3.2</span>
              <strong>Emergency contracting limits: applicable to specific departments</strong>
              <h3>Department of National Defence</h3>
              <p>Non-competitive contract up to $16,000,000.</p>
              <p>Non-competitive contract up to $11,000,000.</p>
            </li>
          </ul>
        </main>
      </body>
    </html>
    """
    _, blocks, _ = html_to_text_blocks(html)
    assert [block.block_type for block in blocks] == ["list_item", "heading", "paragraph", "paragraph"]
    assert blocks[0].text == "A.3.2 Emergency contracting limits: applicable to specific departments"
    assert blocks[1].text == "Department of National Defence"
    assert blocks[2].text == "Non-competitive contract up to $16,000,000."
    assert blocks[3].text == "Non-competitive contract up to $11,000,000."


def test_html_to_text_blocks_serializes_table_without_cell_paragraph_spillover() -> None:
    html = """
    <html>
      <body>
        <main>
          <table>
            <caption>Payment instruments</caption>
            <tr><th>Instrument</th><th>Electronic?</th></tr>
            <tr><td><p>direct deposit</p></td><td><p>yes</p></td></tr>
            <tr><td><p>cheque</p></td><td><p>no</p></td></tr>
          </table>
        </main>
      </body>
    </html>
    """
    _, blocks, _ = html_to_text_blocks(html)
    assert len(blocks) == 3
    assert [block.block_type for block in blocks] == ["table_row", "table_row", "table_row"]
    assert blocks[0].text == "Payment instruments | Columns: Instrument | Electronic?"
    assert blocks[1].text == "Instrument: direct deposit | Electronic?: yes"
    assert blocks[2].text == "Instrument: cheque | Electronic?: no"


def test_html_to_text_blocks_extracts_definition_lists_without_parent_paragraph_flattening() -> None:
    html = """
    <html>
      <body>
        <main>
          <p>
            <dl>
              <dt>1-Year GIC Rate:</dt>
              <dd>
                <p>A 3-year rolling average is applied.</p>
                <ul>
                  <li>Found in Annex 5.1</li>
                  <li>Use the rate at contract award</li>
                </ul>
              </dd>
              <dt>Total Acceptable Contract Cost:</dt>
              <dd>The estimated acceptable contract costs.</dd>
            </dl>
          </p>
        </main>
      </body>
    </html>
    """
    _, blocks, _ = html_to_text_blocks(html)
    assert [block.block_type for block in blocks] == [
        "definition_term",
        "paragraph",
        "list_item",
        "list_item",
        "definition_term",
        "definition_detail",
    ]
    assert blocks[0].text == "1-Year GIC Rate:"
    assert blocks[1].text == "A 3-year rolling average is applied."
    assert blocks[2].text == "Found in Annex 5.1"
    assert blocks[3].text == "Use the rate at contract award"
    assert blocks[4].text == "Total Acceptable Contract Cost:"
    assert blocks[5].text == "The estimated acceptable contract costs."


def test_normalize_document_prefers_main_content_field_over_portal_chrome() -> None:
    html = """
    <html>
      <head><title>Interim Policy on Reciprocal Procurement | CanadaBuys</title></head>
      <body>
        <main>
          <h2>SAP Ariba system maintenance</h2>
          <p>SAP Ariba will be unavailable for scheduled maintenance.</p>
          <div class="field--name-field-main-content">
            <div class="field--item">
              <p><h2>Interim Policy on Reciprocal Procurement</h2><p>The policy limits access to certain procurements.</p></p>
            </div>
          </div>
        </main>
      </body>
    </html>
    """
    normalized = normalize_document(
        SourceDocument(
            source_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/plan/determine-procurement-strategy/interim-policy-reciprocal-procurement",
            fetched_at=datetime.now(timezone.utc),
            html=html,
        )
    )
    texts = [block.text for block in normalized.structure_blocks if block.block_type != "heading"]
    assert texts == ["The policy limits access to certain procurements."]


def test_html_to_text_blocks_clamps_heading_depth_when_markup_skips_levels() -> None:
    html = """
    <html>
      <body>
        <main>
          <h2>Procedure</h2>
          <h4>Step 1</h4>
          <p>First step.</p>
          <h4>Step 2</h4>
          <p>Second step.</p>
        </main>
      </body>
    </html>
    """
    _, blocks, _ = html_to_text_blocks(html)
    headings = [block for block in blocks if block.block_type == "heading"]
    assert headings[0].heading_path == ["Procedure"]
    assert headings[1].heading_path == ["Procedure", "Step 1"]
    assert headings[2].heading_path == ["Procedure", "Step 2"]
