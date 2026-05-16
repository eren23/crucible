"""Tests for crucible.researcher.arxiv_search — Phase 3.1."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest

from crucible.researcher import arxiv_search


# ---------------------------------------------------------------------------
# Atom fixtures
# ---------------------------------------------------------------------------


SAMPLE_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">ArXiv Query: search_query=all:JEPA&amp;start=0&amp;max_results=2</title>
  <id>http://arxiv.org/api/example</id>
  <updated>2025-01-01T00:00:00-05:00</updated>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">2</opensearch:totalResults>

  <entry>
    <id>http://arxiv.org/abs/2105.04906v3</id>
    <updated>2022-01-27T22:18:34Z</updated>
    <published>2021-05-11T08:15:00Z</published>
    <title>VICReg: Variance-Invariance-Covariance Regularization for
        Self-Supervised Learning</title>
    <summary>  Recent self-supervised methods for image representation learning
        maximize the agreement between embedding vectors from different views of
        the same image. A robust solution is...
    </summary>
    <author><name>Adrien Bardes</name></author>
    <author><name>Jean Ponce</name></author>
    <author><name>Yann LeCun</name></author>
    <link href="http://arxiv.org/abs/2105.04906v3" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2105.04906v3" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
  </entry>

  <entry>
    <id>http://arxiv.org/abs/2412.10925v1</id>
    <updated>2024-12-15T00:00:00Z</updated>
    <published>2024-12-14T00:00:00Z</published>
    <title>VJ-VCR: VICReg for Video JEPA</title>
    <summary>We extend VICReg to video JEPA predictors.</summary>
    <author><name>Test Author</name></author>
    <link href="http://arxiv.org/abs/2412.10925v1" rel="alternate" type="text/html"/>
    <category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


@pytest.fixture
def offline_arxiv(monkeypatch):
    """Stub the urllib request so search never hits the network."""
    def fake_urlopen(req, *args, **kwargs):
        class _Resp:
            def __enter__(self_): return self_
            def __exit__(self_, *exc): return False
            def read(self_): return SAMPLE_ATOM
        return _Resp()

    monkeypatch.setattr(arxiv_search.urllib.request, "urlopen", fake_urlopen)


# ---------------------------------------------------------------------------
# search_arxiv happy path
# ---------------------------------------------------------------------------


class TestSearchArxiv:
    def test_empty_query_returns_empty(self):
        assert arxiv_search.search_arxiv("") == []
        assert arxiv_search.search_arxiv("   ") == []

    def test_returns_normalized_papers(self, offline_arxiv):
        results = arxiv_search.search_arxiv("JEPA", limit=5)
        assert len(results) == 2
        first = results[0]
        # Field shape matches the hf_search/literature contract.
        for k in ("arxiv_id", "id", "title", "summary", "abstract",
                  "authors", "categories", "published_at", "url", "source"):
            assert k in first, f"missing {k}"
        assert first["arxiv_id"] == "2105.04906"
        assert first["arxiv_id_versioned"] == "2105.04906v3"
        assert "VICReg" in first["title"]
        assert "Bardes" in first["authors"][0]
        assert "cs.LG" in first["categories"]
        assert first["url"] == "http://arxiv.org/abs/2105.04906v3"
        assert first["source"] == "arxiv"

    def test_limit_caps_results(self, offline_arxiv):
        results = arxiv_search.search_arxiv("JEPA", limit=1)
        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2105.04906"

    def test_summary_truncated_to_500_chars(self, offline_arxiv):
        results = arxiv_search.search_arxiv("JEPA", limit=2)
        for r in results:
            assert len(r["summary"]) <= 500


# ---------------------------------------------------------------------------
# _build_search_query
# ---------------------------------------------------------------------------


class TestBuildSearchQuery:
    def test_no_categories_omits_filter(self):
        q = arxiv_search._build_search_query("attention", None)
        assert q == "all:attention"

    def test_empty_categories_omits_filter(self):
        q = arxiv_search._build_search_query("attention", [])
        assert q == "all:attention"

    def test_categories_ored(self):
        q = arxiv_search._build_search_query(
            "attention", ["cs.LG", "cs.AI"]
        )
        assert "all:attention AND" in q
        assert "cat:cs.LG OR cat:cs.AI" in q


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_network_failure_returns_empty(self, monkeypatch):
        import urllib.error
        def boom(req, *a, **kw):
            raise urllib.error.URLError("network is down")
        monkeypatch.setattr(arxiv_search.urllib.request, "urlopen", boom)
        results = arxiv_search.search_arxiv("anything")
        assert results == []

    def test_malformed_xml_returns_empty(self, monkeypatch):
        def fake_urlopen(req, *a, **kw):
            class _R:
                def __enter__(self_): return self_
                def __exit__(self_, *exc): return False
                def read(self_): return b"<<<not-xml<<<"
            return _R()
        monkeypatch.setattr(arxiv_search.urllib.request, "urlopen", fake_urlopen)
        results = arxiv_search.search_arxiv("anything")
        assert results == []


# ---------------------------------------------------------------------------
# MCP dispatch round-trip
# ---------------------------------------------------------------------------


def test_mcp_dispatcher_invokes_search(monkeypatch, offline_arxiv):
    from crucible.mcp.tools import TOOL_DISPATCH

    handler = TOOL_DISPATCH["research_arxiv_search"]
    out = handler({"query": "JEPA", "limit": 2})
    assert out["query"] == "JEPA"
    assert out["count"] == 2
    assert len(out["results"]) == 2
    assert out["results"][0]["arxiv_id"] == "2105.04906"
