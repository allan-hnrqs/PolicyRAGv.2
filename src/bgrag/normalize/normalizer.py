"""Normalization from fetched HTML to typed normalized documents."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from bs4 import BeautifulSoup

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


def infer_source_family(url: str) -> SourceFamily:
    normalized = canonicalize_url(url).lower()
    if "buy-canadian-policy" in normalized or "/policies-and-guidelines/policies-directives-and-regulations/" in normalized:
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


def _content_root(soup: BeautifulSoup):
    selectors = [
        "main",
        "[role='main']",
        "article",
        "#wb-cont",
        ".main-content",
        ".layout-content",
    ]
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
    order = 0
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if _should_drop_text(text):
            continue
        if re.fullmatch(r"h[1-4]", element.name or ""):
            level = int(element.name[1])
            current_heading_path = current_heading_path[: level - 1]
            current_heading_path.append(text)
            block_type = "heading"
        elif element.name == "li":
            block_type = "list_item"
        elif element.name == "table":
            block_type = "table"
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
        extraction_method="html_regions_v2_main_root",
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
