"""Source collection from the Buyer’s Guide and supporting policy pages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from bgrag.source_catalog import SOURCE_CATALOG
from bgrag.types import SourceDocument

BUYERS_GUIDE_ROOT = SOURCE_CATALOG.buyers_guide_root
BUY_CANADIAN_ROOT = SOURCE_CATALOG.buy_canadian_root
BUY_CANADIAN_POLICY_PREFIX = SOURCE_CATALOG.buy_canadian_policy_prefix
TBS_DIRECTIVE_HTML = SOURCE_CATALOG.tbs_directive_html
DEFAULT_SEED_URLS = SOURCE_CATALOG.default_seed_urls
TBS_HOSTS = set(SOURCE_CATALOG.tbs_hosts)
CANONICAL_TBS_HOST = SOURCE_CATALOG.canonical_tbs_host

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def tbs_query_to_canonical(query_items: list[tuple[str, str]]) -> str:
    return SOURCE_CATALOG.canonical_tbs_query(query_items)


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme or "https"
    host = parsed.netloc.lower()
    if host in SOURCE_CATALOG.tbs_hosts:
        host = SOURCE_CATALOG.canonical_tbs_host
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    if host == "canadabuys.canada.ca" and path.endswith("-0"):
        path = path[:-2]
    query = ""
    if host == SOURCE_CATALOG.canonical_tbs_host and path == SOURCE_CATALOG.tbs_document_path:
        query = tbs_query_to_canonical(parse_qsl(parsed.query, keep_blank_values=True))
    return urlunsplit((scheme, host, path, query, ""))


def in_scope_url(url: str) -> bool:
    canonical = canonicalize_url(url)
    return SOURCE_CATALOG.is_in_scope_canonical_url(canonical)


def should_follow_links(url: str) -> bool:
    canonical = canonicalize_url(url)
    return SOURCE_CATALOG.should_follow_canonical_url(canonical)


def raw_snapshot_stem(url: str) -> str:
    final_url = canonicalize_url(url)
    return final_url.replace("https://", "").replace("http://", "").replace("/", "_").replace("?", "_")[:160]


def extract_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()
    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        absolute = canonicalize_url(urljoin(base_url, href))
        if in_scope_url(absolute):
            links.add(absolute)
    return sorted(links)


@dataclass
class FetchResult:
    document: SourceDocument


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_url(client: httpx.Client, url: str) -> FetchResult:
    response = client.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    final_url = canonicalize_url(str(response.url))
    discovered_links = extract_links(final_url, response.text)
    document = SourceDocument(
        source_url=url,
        fetched_at=datetime.now(timezone.utc),
        final_url=final_url,
        status_code=response.status_code,
        html=response.text,
        headers={key: value for key, value in response.headers.items()},
        discovered_links=discovered_links,
    )
    return FetchResult(document=document)


def crawl_scope(seed_urls: list[str] | None = None, max_pages: int = 300) -> list[FetchResult]:
    queue: deque[str] = deque(canonicalize_url(url) for url in (seed_urls or DEFAULT_SEED_URLS))
    seen: set[str] = set(queue)
    results: list[FetchResult] = []
    with httpx.Client(headers={"User-Agent": BROWSER_UA}) as client:
        while queue and len(results) < max_pages:
            current = queue.popleft()
            if not in_scope_url(current):
                continue
            try:
                fetched = fetch_url(client, current)
            except Exception:
                continue
            results.append(fetched)
            if not should_follow_links(current):
                continue
            for link in fetched.document.discovered_links:
                if link not in seen:
                    seen.add(link)
                    queue.append(link)
    return results


def write_raw_snapshot(raw_dir: Path, results: list[FetchResult]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for existing in raw_dir.glob("*.html"):
        existing.unlink()
    for result in results:
        final_url = str(result.document.final_url or result.document.source_url)
        path = raw_dir / f"{raw_snapshot_stem(final_url)}.html"
        path.write_text(result.document.html, encoding="utf-8")
