"""Explicit source-scope catalog for crawl and normalization rules."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode

from bgrag.types import SourceFamily


@dataclass(frozen=True)
class SourceCatalog:
    buyers_guide_root: str
    buy_canadian_root: str
    buy_canadian_policy_prefix: str
    tbs_directive_html: str
    tbs_hosts: frozenset[str]
    canonical_tbs_host: str
    tbs_document_path: str
    tbs_document_id: str
    noise_prefixes: tuple[str, ...]
    buyers_guide_chrome_heading_prefix: str
    buyers_guide_prefixes: tuple[str, ...]
    buy_canadian_family_prefixes: tuple[str, ...]

    @property
    def default_seed_urls(self) -> list[str]:
        return [
            self.buyers_guide_root,
            self.buy_canadian_root,
            self.tbs_directive_html,
        ]

    def canonical_tbs_query(self, query_items: list[tuple[str, str]]) -> str:
        query = dict(query_items)
        if query.get("id") != self.tbs_document_id:
            return ""
        section = (query.get("section") or "").lower()
        if section in {"", "html"}:
            return urlencode([("id", self.tbs_document_id)])
        kept: list[tuple[str, str]] = [("id", self.tbs_document_id), ("section", section)]
        p_value = query.get("p")
        if p_value:
            kept.append(("p", p_value))
        return urlencode(kept)

    def infer_source_family(self, canonical_url: str) -> SourceFamily:
        normalized = canonical_url.lower()
        if any(prefix in normalized for prefix in self.buy_canadian_family_prefixes):
            return SourceFamily.BUY_CANADIAN_POLICY
        if any(host in normalized for host in self.tbs_hosts) or self.tbs_document_path in normalized:
            return SourceFamily.TBS_DIRECTIVE
        return SourceFamily.BUYERS_GUIDE

    def is_in_scope_canonical_url(self, canonical_url: str) -> bool:
        normalized = canonical_url.lower()
        if normalized.startswith(self.buy_canadian_root):
            return True
        if normalized.startswith(self.buy_canadian_policy_prefix):
            return True
        if normalized.startswith(self.buyers_guide_root):
            return True
        if normalized.startswith(f"https://{self.canonical_tbs_host}{self.tbs_document_path}") and (
            f"id={self.tbs_document_id}" in normalized
        ):
            return True
        return False

    def should_follow_canonical_url(self, canonical_url: str) -> bool:
        normalized = canonical_url.lower()
        return any(normalized.startswith(prefix) for prefix in self.buyers_guide_prefixes) or any(
            normalized.startswith(prefix) for prefix in (self.buy_canadian_root, self.buy_canadian_policy_prefix)
        )

    def should_drop_noise_text(self, text: str) -> bool:
        return any(text.startswith(prefix) for prefix in self.noise_prefixes)


SOURCE_CATALOG = SourceCatalog(
    buyers_guide_root="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide",
    buy_canadian_root="https://canadabuys.canada.ca/en/buy-canadian-policy",
    buy_canadian_policy_prefix=(
        "https://canadabuys.canada.ca/en/how-procurement-works/"
        "policies-and-guidelines/policies-directives-and-regulations"
    ),
    tbs_directive_html="https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32692",
    tbs_hosts=frozenset(
        {
            "www.tbs-sct.canada.ca",
            "tbs-sct.canada.ca",
            "www.tbs-sct.gc.ca",
            "tbs-sct.gc.ca",
        }
    ),
    canonical_tbs_host="www.tbs-sct.canada.ca",
    tbs_document_path="/pol/doc-eng.aspx",
    tbs_document_id="32692",
    noise_prefixes=(
        "Skip to main content",
        'Skip to "About Canada.ca"',
        'Skip to "About this site"',
        "Top of page",
        "Report a problem",
    ),
    buyers_guide_chrome_heading_prefix="Buyer's Guide ",
    buyers_guide_prefixes=("https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide",),
    buy_canadian_family_prefixes=(
        "buy-canadian-policy",
        "/policies-and-guidelines/policies-directives-and-regulations/",
        "/buyer-s-portal/legislation-and-policies/",
    ),
)
