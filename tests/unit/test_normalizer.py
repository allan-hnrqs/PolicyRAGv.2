from bgrag.normalize.normalizer import trim_buyers_guide_chrome
from bgrag.types import StructureBlock


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
