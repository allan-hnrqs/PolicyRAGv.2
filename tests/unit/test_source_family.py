from datetime import datetime, timezone

from bgrag.collect.collector import canonicalize_url, in_scope_url, should_follow_links
from bgrag.normalize.normalizer import assign_graph_relationships, infer_authority_rank, infer_source_family, normalize_document
from bgrag.types import SourceDocument, SourceFamily


def test_infer_source_family() -> None:
    assert infer_source_family("https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide") == SourceFamily.BUYERS_GUIDE
    assert infer_source_family("https://canadabuys.canada.ca/en/buy-canadian-policy") == SourceFamily.BUY_CANADIAN_POLICY
    assert infer_source_family("https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32692") == SourceFamily.TBS_DIRECTIVE


def test_authority_rank() -> None:
    assert infer_authority_rank(SourceFamily.TBS_DIRECTIVE) == 1
    assert infer_authority_rank(SourceFamily.BUY_CANADIAN_POLICY) == 2
    assert infer_authority_rank(SourceFamily.BUYERS_GUIDE) == 3


def test_normalize_document_extracts_blocks() -> None:
    document = SourceDocument(
        source_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/example",
        fetched_at=datetime.now(timezone.utc),
        html="<html><head><title>Example</title></head><body><h1>Header</h1><p>Body text.</p></body></html>",
    )
    normalized = normalize_document(document)
    assert normalized.title == "Example"
    assert normalized.source_family == SourceFamily.BUYERS_GUIDE
    assert normalized.structure_blocks


def test_collector_scope_rules() -> None:
    assert in_scope_url("https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide")
    assert in_scope_url("https://canadabuys.canada.ca/en/buy-canadian-policy")
    assert in_scope_url("https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32692")
    assert not in_scope_url("https://www.canada.ca/en/services/jobs.html")
    assert should_follow_links("https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide")
    assert not should_follow_links("https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32692")


def test_tbs_canonicalization() -> None:
    assert canonicalize_url("https://tbs-sct.gc.ca/pol/doc-eng.aspx?id=32692&section=html#x") == "https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32692"


def test_assign_graph_relationships() -> None:
    root = normalize_document(
        SourceDocument(
            source_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide",
            fetched_at=datetime.now(timezone.utc),
            final_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide",
            html="<html><body><main><h1>Root</h1><a href='/en/buyer-s-portal/buyer-s-guide/plan'>Plan</a></main></body></html>",
            discovered_links=["https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/plan"],
        )
    )
    child = normalize_document(
        SourceDocument(
            source_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/plan",
            fetched_at=datetime.now(timezone.utc),
            final_url="https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide/plan",
            html="<html><body><main><h1>Plan</h1></main></body></html>",
        )
    )
    docs = assign_graph_relationships([root, child])
    by_url = {doc.canonical_url: doc for doc in docs}
    assert by_url[child.canonical_url].graph.parent_url == root.canonical_url
    assert by_url[root.canonical_url].graph.child_urls == [child.canonical_url]
