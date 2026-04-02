"""Normalization from fetched HTML to typed normalized documents."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from bs4 import BeautifulSoup, NavigableString, Tag

from bgrag.collect.collector import canonicalize_url
from bgrag.types import (
    NormalizedDocument,
    SourceDocument,
    SourceFamily,
    SourceGraph,
    SourceLink,
    StructureBlock,
)

NOISE_PREFIXES = (
    "Skip to main content",
    "Skip to \"About Canada.ca\"",
    "Skip to \"About this site\"",
    "Top of page",
    "Report a problem",
)

EXTRACTION_TAGS = ("h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "table")
NESTED_BLOCK_TAGS = ("h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "dl", "table", "ul", "ol")


def infer_source_family(url: str) -> SourceFamily:
    normalized = canonicalize_url(url).lower()
    if (
        "buy-canadian-policy" in normalized
        or "/policies-and-guidelines/policies-directives-and-regulations/" in normalized
        or "/buyer-s-portal/legislation-and-policies/" in normalized
    ):
        return SourceFamily.BUY_CANADIAN_POLICY
    if "tbs-sct.canada.ca" in normalized or "pol/doc-eng.aspx" in normalized:
        return SourceFamily.TBS_DIRECTIVE
    return SourceFamily.BUYERS_GUIDE


def infer_authority_rank(source_family: SourceFamily) -> int:
    if source_family is SourceFamily.TBS_DIRECTIVE:
        return 1
    if source_family is SourceFamily.BUY_CANADIAN_POLICY:
        return 2
    return 3


def slug_hash(value: str, size: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:size]


def _class_tokens(node: Tag) -> set[str]:
    raw = node.get("class") or []
    if isinstance(raw, str):
        return set(raw.split())
    return {str(value) for value in raw}


def _text_weight(node: Tag) -> int:
    return len(_normalize_text(node.get_text(" ", strip=True)))


def _first_field_item_with_class(soup: BeautifulSoup, class_name: str) -> Tag | None:
    for node in soup.find_all(["div", "section", "article"]):
        classes = _class_tokens(node)
        if class_name in classes and "field--item" in classes:
            return node
        if class_name in classes:
            descendant = node.find(
                lambda tag: isinstance(tag, Tag) and "field--item" in _class_tokens(tag)
            )
            if descendant is not None:
                return descendant
    return None


def _content_root(soup: BeautifulSoup):
    for class_name in ("field--name-field-main-content", "field--name-body"):
        candidate = _first_field_item_with_class(soup, class_name)
        if candidate is not None and _text_weight(candidate) > 0:
            return candidate

    scored_candidates: list[tuple[int, Tag]] = []
    for node in soup.find_all(["div", "section", "article", "main"]):
        classes = _class_tokens(node)
        if not classes and node.name != "main":
            continue

        text_weight = _text_weight(node)
        if text_weight == 0:
            continue

        score: int | None = None
        if "field--name-field-main-content" in classes and "field--item" in classes:
            score = 300_000 + text_weight
        elif "field--name-body" in classes and "field--item" in classes:
            score = 100_000 + text_weight
        elif node.get("id") == "wb-cont__content":
            score = 50_000 + text_weight
        elif node.name == "main":
            score = 10_000 + text_weight

        if score is not None:
            scored_candidates.append((score, node))

    if scored_candidates:
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return scored_candidates[0][1]

    selectors = ("[role='main']", "article", "#wb-cont", ".main-content", ".layout-content")
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return soup.body or soup


def _should_drop_text(text: str) -> bool:
    if not text:
        return True
    if any(text.startswith(prefix) for prefix in NOISE_PREFIXES):
        return True
    return False


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _node_has_nested_blocks(element: Tag) -> bool:
    return element.find(NESTED_BLOCK_TAGS) is not None


def _direct_text_without_nested_blocks(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in NESTED_BLOCK_TAGS:
            continue
        text = _direct_text_without_nested_blocks(child)
        if text:
            parts.append(text)
    return _normalize_text(" ".join(parts))


def _serialize_table(element: Tag) -> str:
    caption = _normalize_text(element.caption.get_text(" ", strip=True)) if element.caption else ""
    rows: list[str] = []
    for row in element.find_all("tr"):
        cells: list[str] = []
        for cell in row.find_all(["th", "td"], recursive=False):
            cell_text = _normalize_text(cell.get_text(" ", strip=True))
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    parts = []
    if caption:
        parts.append(caption)
    parts.extend(rows)
    return _normalize_text("\n".join(parts)) if parts else _normalize_text(element.get_text(" ", strip=True))


def _table_to_row_texts(element: Tag) -> list[str]:
    caption = _normalize_text(element.caption.get_text(" ", strip=True)) if element.caption else ""
    rows: list[list[str]] = []
    first_row_has_header = False

    for row_index, row in enumerate(element.find_all("tr")):
        cells = row.find_all(["th", "td"], recursive=False)
        values = [_normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
        values = [value for value in values if value]
        if not values:
            continue
        if row_index == 0 and any(cell.name == "th" for cell in cells):
            first_row_has_header = True
        rows.append(values)

    if not rows:
        return []

    headers = rows[0]
    data_rows = rows[1:] if first_row_has_header else rows

    if not first_row_has_header and len(rows) > 1 and len(headers) > 1 and all(len(row) == len(headers) for row in rows[1:]):
        if all(len(cell) <= 80 for cell in headers):
            data_rows = rows[1:]
        else:
            headers = []

    row_texts: list[str] = []
    if headers and len(data_rows) >= 1:
        if caption:
            row_texts.append(f"{caption} | Columns: {' | '.join(headers)}")
        for row in data_rows:
            if len(row) == len(headers):
                pairs = [f"{headers[index]}: {value}" for index, value in enumerate(row)]
                row_texts.append(" | ".join(pairs))
            else:
                row_texts.append(" | ".join(row))
        return row_texts

    serialized = _serialize_table(element)
    return [serialized] if serialized else []


def _renumber_blocks(blocks: list[StructureBlock]) -> list[StructureBlock]:
    renumbered: list[StructureBlock] = []
    for order, block in enumerate(blocks):
        renumbered.append(
            block.model_copy(
                update={
                    "block_id": f"block_{order:04d}",
                    "order": order,
                }
            )
        )
    return renumbered


def trim_buyers_guide_chrome(blocks: list[StructureBlock]) -> list[StructureBlock]:
    title_heading_index: int | None = None
    for index, block in enumerate(blocks):
        if block.block_type == "heading" and block.text.startswith("Buyer's Guide "):
            title_heading_index = index
            break
    if title_heading_index is None:
        return blocks

    first_content_index: int | None = None
    for index, block in enumerate(blocks[title_heading_index + 1 :], start=title_heading_index + 1):
        if block.block_type != "list_item":
            first_content_index = index
            break
    if first_content_index is None:
        return blocks[title_heading_index:]
    return blocks[first_content_index:]


def html_to_text_blocks(html: str) -> tuple[str, list[StructureBlock], list[SourceLink]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
    root = _content_root(soup)

    breadcrumbs: list[SourceLink] = []
    for link in soup.select("nav a[href], .breadcrumb a[href], ol.breadcrumb a[href]"):
        text = " ".join(link.get_text(" ", strip=True).split())
        href = link.get("href", "").strip()
        if text and href and not _should_drop_text(text):
            breadcrumbs.append(
                SourceLink(
                    title=text,
                    url=href,
                    canonical_url=canonicalize_url(href),
                    in_scope="canadabuys" in href or "tbs-sct" in href,
                )
            )

    blocks: list[StructureBlock] = []
    current_heading_path: list[str] = []
    current_heading_levels: list[int] = []
    order = 0
    for element in root.find_all(EXTRACTION_TAGS):
        if element.name != "table" and element.find_parent("table") is not None:
            continue

        if re.fullmatch(r"h[1-4]", element.name or ""):
            text = _normalize_text(element.get_text(" ", strip=True))
            if _should_drop_text(text) or not text:
                continue
            level = int(element.name[1])
            while current_heading_levels and current_heading_levels[-1] >= level:
                current_heading_levels.pop()
                current_heading_path.pop()
            current_heading_levels.append(level)
            current_heading_path.append(text)
            blocks.append(
                StructureBlock(
                    block_id=f"block_{order:04d}",
                    block_type="heading",
                    heading=current_heading_path[-1],
                    heading_path=list(current_heading_path),
                    text=text,
                    order=order,
                )
            )
            order += 1
            continue

        if element.name == "table":
            row_texts = _table_to_row_texts(element)
            for row_index, row_text in enumerate(row_texts):
                if _should_drop_text(row_text) or not row_text:
                    continue
                blocks.append(
                    StructureBlock(
                        block_id=f"block_{order:04d}",
                        block_type="table" if row_index == 0 and len(row_texts) == 1 else "table_row",
                        heading=current_heading_path[-1] if current_heading_path else None,
                        heading_path=list(current_heading_path),
                        text=row_text,
                        order=order,
                    )
                )
                order += 1
            continue

        if element.name == "p":
            if _node_has_nested_blocks(element):
                text = _direct_text_without_nested_blocks(element)
            else:
                text = _normalize_text(element.get_text(" ", strip=True))
        elif element.name in {"li", "dd"}:
            if _node_has_nested_blocks(element):
                text = _direct_text_without_nested_blocks(element)
            else:
                text = _normalize_text(element.get_text(" ", strip=True))
        elif element.name == "dt":
            text = _normalize_text(element.get_text(" ", strip=True))
        else:
            text = _normalize_text(element.get_text(" ", strip=True))

        if _should_drop_text(text):
            continue
        if not text:
            continue
        if element.name == "li":
            block_type = "list_item"
        elif element.name == "dt":
            block_type = "definition_term"
        elif element.name == "dd":
            block_type = "definition_detail"
        else:
            block_type = "paragraph"
        blocks.append(
            StructureBlock(
                block_id=f"block_{order:04d}",
                block_type=block_type,
                heading=current_heading_path[-1] if current_heading_path else None,
                heading_path=list(current_heading_path),
                text=text,
                order=order,
            )
        )
        order += 1

    return title, blocks, breadcrumbs


def normalize_document(document: SourceDocument) -> NormalizedDocument:
    canonical_url = canonicalize_url(str(document.final_url or document.source_url))
    source_family = infer_source_family(canonical_url)
    title, blocks, breadcrumbs = html_to_text_blocks(document.html)
    if source_family is SourceFamily.BUYERS_GUIDE:
        blocks = _renumber_blocks(trim_buyers_guide_chrome(blocks))
    raw_text = "\n\n".join(block.text for block in blocks)
    doc_id = slug_hash(canonical_url)
    return NormalizedDocument(
        doc_id=doc_id,
        title=title,
        source_url=str(document.source_url),
        canonical_url=canonical_url,
        source_family=source_family,
        authority_rank=infer_authority_rank(source_family),
        fetched_at=document.fetched_at,
        content_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        word_count=len(raw_text.split()),
        extraction_method="html_regions_v3_block_safe",
        breadcrumbs=breadcrumbs,
        graph=SourceGraph(
            lineage_urls=[canonical_url],
            lineage_doc_ids=[doc_id],
            depth=max(0, canonical_url.count("/") - 3),
            outgoing_in_scope_links=list(document.discovered_links),
        ),
        structure_blocks=blocks,
        raw_text=raw_text,
        markdown_text=raw_text,
    )


def _parent_from_url(canonical_url: str, known_urls: set[str]) -> str | None:
    if "buyer-s-guide" not in canonical_url:
        return None
    candidates: list[str] = []
    current = canonical_url.rstrip("/")
    while "/" in current.rsplit("/", 1)[0]:
        current = current.rsplit("/", 1)[0]
        candidates.append(current)
        if current.endswith("/buyer-s-guide"):
            break
    for candidate in candidates:
        if candidate in known_urls:
            return candidate
    return None


def assign_graph_relationships(documents: list[NormalizedDocument]) -> list[NormalizedDocument]:
    by_url = {document.canonical_url: document for document in documents}
    incoming: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        outgoing = [url for url in document.graph.outgoing_in_scope_links if url in by_url and url != document.canonical_url]
        document.graph.outgoing_in_scope_links = sorted(set(outgoing))
        for target in document.graph.outgoing_in_scope_links:
            incoming[target].add(document.canonical_url)

    known_urls = set(by_url.keys())
    for document in documents:
        parent_url = _parent_from_url(document.canonical_url, known_urls)
        document.graph.parent_url = parent_url
        document.graph.parent_doc_id = by_url[parent_url].doc_id if parent_url and parent_url in by_url else None
        document.graph.incoming_in_scope_links = sorted(incoming.get(document.canonical_url, set()))

    for document in documents:
        document.graph.child_urls = sorted(
            [candidate.canonical_url for candidate in documents if candidate.graph.parent_url == document.canonical_url]
        )
        document.graph.child_doc_ids = [by_url[url].doc_id for url in document.graph.child_urls if url in by_url]

    for document in documents:
        lineage_urls: list[str] = []
        current = document
        seen: set[str] = set()
        while current and current.canonical_url not in seen:
            seen.add(current.canonical_url)
            lineage_urls.append(current.canonical_url)
            parent_url = current.graph.parent_url
            current = by_url.get(parent_url) if parent_url else None
        lineage_urls.reverse()
        document.graph.lineage_urls = lineage_urls
        document.graph.lineage_doc_ids = [by_url[url].doc_id for url in lineage_urls if url in by_url]
        document.graph.depth = max(0, len(lineage_urls) - 1)
    return documents
