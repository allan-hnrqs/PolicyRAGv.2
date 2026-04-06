from pathlib import Path

import pytest

from bgrag.config import Settings
from bgrag.indexing.search_backend import (
    SEARCH_BACKEND_ELASTICSEARCH,
    SEARCH_BACKEND_OPENSEARCH,
    normalize_search_backend,
    search_backend_label,
    search_backend_url,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_normalize_search_backend_accepts_supported_values() -> None:
    assert normalize_search_backend("Elasticsearch") == SEARCH_BACKEND_ELASTICSEARCH
    assert normalize_search_backend("opensearch") == SEARCH_BACKEND_OPENSEARCH


def test_normalize_search_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported search backend"):
        normalize_search_backend("solr")


def test_search_backend_label_is_human_readable() -> None:
    assert search_backend_label("elasticsearch") == "Elasticsearch"
    assert search_backend_label("opensearch") == "OpenSearch"


def test_search_backend_url_uses_backend_specific_setting() -> None:
    settings = Settings(
        project_root=REPO_ROOT,
        elastic_url="http://elastic.local:9200",
        opensearch_url="http://opensearch.local:9200",
    )
    assert search_backend_url(settings, "elasticsearch") == "http://elastic.local:9200"
    assert search_backend_url(settings, "opensearch") == "http://opensearch.local:9200"
