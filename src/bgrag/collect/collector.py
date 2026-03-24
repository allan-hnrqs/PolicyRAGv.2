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

from bgrag.types import SourceDocument

BUYERS_GUIDE_ROOT = "https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide"
BUY_CANADIAN_ROOT = "https://canadabuys.canada.ca/en/buy-canadian-policy"
BUY_CANADIAN_POLICY_PREFIX = (
    "https://canadabuys.canada.ca/en/how-procurement-works/"
    "policies-and-guidelines/policies-directives-and-regulations"
)
TBS_DIRECTIVE_HTML = "https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32692"

DEFAULT_SEED_URLS = [
    BUYERS_GUIDE_ROOT,
    BUY_CANADIAN_ROOT,
    TBS_DIRECTIVE_HTML,
]

TBS_HOSTS = {
    "www.tbs-sct.canada.ca",
    "tbs-sct.canada.ca",
    "www.tbs-sct.gc.ca",
    "tbs-sct.gc.ca",
}
CANONICAL_TBS_HOST = "www.tbs-sct.canada.ca"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def tbs_query_to_canonical(query_items: list[tuple[str, str]]) -> str:
    query = dict(query_items)
    if query.get("id") != "32692":
        return ""
    section = (query.get("section") or "").lower()
    if section in {"", "html"}:
        return urlencode([("id", "32692")])
    kept: list[tuple[str, str]] = [("id", "32692"), ("section", section)]
    p_value = query.get("p")
    if p_value:
        kept.append(("p", p_value))
    return urlencode(kept)


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme or "https"
    host = parsed.netloc.lower()
    if host in TBS_HOSTS:
        host = CANONICAL_TBS_HOST
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    if host == "canadabuys.canada.ca" and path.endswith("-0"):
        path = path[:-2]
    query = ""
    if host == CANONICAL_TBS_HOST and path == "/pol/doc-eng.aspx":
        query = tbs_query_to_canonical(parse_qsl(parsed.query, keep_blank_values=True))
    return urlunsplit((scheme, host, path, query, ""))


def in_scope_url(url: str) -> bool:
    canonical = canonicalize_url(url)
    parsed = urlsplit(canonical)
    if parsed.netloc == "canadabuys.canada.ca":
        return (
            parsed.path == "/en/buy-canadian-policy"
            or parsed.path.startswith("/en/buyer-s-portal/buyer-s-guide")
            or canonical.startswith(BUY_CANADIAN_POLICY_PREFIX)
        )
    if parsed.netloc == CANONICAL_TBS_HOST:
        return parsed.path == "/pol/doc-eng.aspx" and "id=32692" in parsed.query
    return False


def should_follow_links(url: str) -> bool:
    canonical = canonicalize_url(url)
    if canonical.startswith(BUYERS_GUIDE_ROOT):
        return True
    if canonical == BUY_CANADIAN_ROOT or canonical.startswith(BUY_CANADIAN_POLICY_PREFIX):
        return True
    return False


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
        stem = final_url.replace("https://", "").replace("http://", "").replace("/", "_").replace("?", "_")
        path = raw_dir / f"{stem[:160]}.html"
        path.write_text(result.document.html, encoding="utf-8")
