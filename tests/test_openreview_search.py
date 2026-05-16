"""Tests for crucible.researcher.openreview_search — Phase 3.2."""
from __future__ import annotations

import json
from typing import Any

import pytest

from crucible.researcher import openreview_search as ors


# ---------------------------------------------------------------------------
# Fixture: stub urlopen
# ---------------------------------------------------------------------------


def _v2_payload() -> bytes:
    """OpenReview v2 response shape: content values wrapped in {value: ...}."""
    return json.dumps({
        "notes": [
            {
                "id": "abc123",
                "forum": "abc123",
                "cdate": 1_700_000_000_000,  # epoch ms
                "tcdate": 1_690_000_000_000,
                "content": {
                    "title": {"value": "VICReg for JEPA"},
                    "abstract": {"value": "We extend VICReg to JEPA encoders."},
                    "authors": {"value": ["Alice", "Bob"]},
                    "keywords": {"value": ["JEPA", "self-supervised"]},
                    "venue": {"value": "ICLR 2024 Spotlight"},
                    "venueid": {"value": "ICLR.cc/2024/Conference"},
                    "pdf": {"value": "/pdf?id=abc123"},
                },
            },
            {
                "id": "xyz789",
                "forum": "xyz789",
                "cdate": 1_650_000_000_000,
                "content": {
                    "title": {"value": "Older Predictor Work"},
                    "abstract": {"value": "Earlier paper."},
                    "authors": {"value": ["Carol"]},
                    "venue": {"value": "NeurIPS 2022"},
                    "venueid": {"value": "NeurIPS.cc/2022/Conference"},
                },
            },
        ]
    }).encode("utf-8")


def _v1_payload() -> bytes:
    """OpenReview v1 response shape: content values bare."""
    return json.dumps({
        "notes": [
            {
                "id": "v1abc",
                "forum": "v1abc",
                "cdate": 1_650_000_000_000,
                "content": {
                    "title": "Legacy Format",
                    "abstract": "v1 API style.",
                    "authors": "Alice, Bob",  # comma-string variant
                    "keywords": "k1, k2",
                    "venue": "ICLR 2021",
                },
            },
        ]
    }).encode("utf-8")


def _make_urlopen(body: bytes):
    def fake(req, *args, **kwargs):
        class _R:
            def __enter__(self_): return self_
            def __exit__(self_, *exc): return False
            def read(self_): return body
        return _R()
    return fake


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestSearchOpenReview:
    def test_empty_query_returns_empty(self):
        assert ors.search_openreview("") == []

    def test_v2_response_normalizes_correctly(self, monkeypatch):
        monkeypatch.setattr(
            ors.urllib.request, "urlopen", _make_urlopen(_v2_payload())
        )
        out = ors.search_openreview("JEPA", limit=5)
        assert len(out) == 2
        first = out[0]
        for key in ("openreview_id", "id", "title", "summary", "abstract",
                    "authors", "venue", "venueid", "url", "pdf_url", "source"):
            assert key in first, f"missing {key}"
        assert first["openreview_id"] == "abc123"
        assert first["title"] == "VICReg for JEPA"
        assert first["authors"] == ["Alice", "Bob"]
        assert first["venue"] == "ICLR 2024 Spotlight"
        assert first["venueid"] == "ICLR.cc/2024/Conference"
        assert first["url"] == "https://openreview.net/forum?id=abc123"
        assert first["pdf_url"] == "https://openreview.net/pdf?id=abc123"
        assert first["source"] == "openreview"

    def test_v1_response_normalizes_correctly(self, monkeypatch):
        # v2 endpoint 404s (returns non-JSON) → v1 fallback fires.
        def fake_urlopen(req, *args, **kwargs):
            url = req.get_full_url() if hasattr(req, "get_full_url") else str(req)
            class _R:
                def __enter__(self_): return self_
                def __exit__(self_, *exc): return False
                def read(self_):
                    if "api2" in url:
                        return b"<html>404 not found</html>"  # bad payload
                    return _v1_payload()
            return _R()
        monkeypatch.setattr(ors.urllib.request, "urlopen", fake_urlopen)

        out = ors.search_openreview("legacy", limit=5)
        assert len(out) == 1
        first = out[0]
        assert first["openreview_id"] == "v1abc"
        assert first["title"] == "Legacy Format"
        # v1 stored authors as comma string → normalized to list.
        assert first["authors"] == ["Alice", "Bob"]
        assert first["keywords"] == ["k1", "k2"]

    def test_year_filter_post_normalization(self, monkeypatch):
        monkeypatch.setattr(
            ors.urllib.request, "urlopen", _make_urlopen(_v2_payload())
        )
        # _v2 has two papers — ICLR 2024 + NeurIPS 2022.
        out = ors.search_openreview("anything", year=2024)
        assert len(out) == 1
        assert "2024" in out[0]["venue"]

    def test_limit_caps_results(self, monkeypatch):
        monkeypatch.setattr(
            ors.urllib.request, "urlopen", _make_urlopen(_v2_payload())
        )
        out = ors.search_openreview("anything", limit=1)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_both_api_versions_fail_returns_empty(self, monkeypatch):
        import urllib.error
        def boom(req, *a, **kw):
            raise urllib.error.URLError("network is down")
        monkeypatch.setattr(ors.urllib.request, "urlopen", boom)
        assert ors.search_openreview("anything") == []

    def test_malformed_json_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            ors.urllib.request, "urlopen",
            _make_urlopen(b"<<<not-json<<<"),
        )
        assert ors.search_openreview("anything") == []

    def test_missing_notes_key_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            ors.urllib.request, "urlopen",
            _make_urlopen(json.dumps({"unrelated": []}).encode("utf-8")),
        )
        assert ors.search_openreview("anything") == []


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestOptionalAuth:
    def test_openreview_token_added_when_set(self, monkeypatch):
        captured = {"auth": None}
        def fake_urlopen(req, *a, **kw):
            captured["auth"] = req.headers.get("Authorization")
            class _R:
                def __enter__(self_): return self_
                def __exit__(self_, *exc): return False
                def read(self_): return _v2_payload()
            return _R()
        monkeypatch.setattr(ors.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setenv("OPENREVIEW_TOKEN", "secret-token-123")

        ors.search_openreview("test")
        assert captured["auth"] == "Bearer secret-token-123"

    def test_no_auth_header_when_env_unset(self, monkeypatch):
        captured = {"auth": "default"}
        def fake_urlopen(req, *a, **kw):
            captured["auth"] = req.headers.get("Authorization")
            class _R:
                def __enter__(self_): return self_
                def __exit__(self_, *exc): return False
                def read(self_): return _v2_payload()
            return _R()
        monkeypatch.setattr(ors.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.delenv("OPENREVIEW_TOKEN", raising=False)

        ors.search_openreview("test")
        assert captured["auth"] is None


# ---------------------------------------------------------------------------
# MCP dispatch
# ---------------------------------------------------------------------------


def test_mcp_dispatcher_invokes_search(monkeypatch):
    monkeypatch.setattr(
        ors.urllib.request, "urlopen", _make_urlopen(_v2_payload())
    )
    from crucible.mcp.tools import TOOL_DISPATCH
    handler = TOOL_DISPATCH["research_openreview_search"]
    out = handler({"query": "JEPA", "limit": 5})
    assert out["query"] == "JEPA"
    assert out["count"] == 2
    assert out["results"][0]["openreview_id"] == "abc123"
