"""Unit tests for researcher/hf_search.py."""
from __future__ import annotations

import pytest

from crucible.researcher import hf_search


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="Unknown HF search kind"):
        hf_search.search("weird", "query")


def test_empty_query_returns_empty():
    assert hf_search.search("datasets", "") == []
    assert hf_search.search("models", "   ") == []


def test_search_datasets_normalizes_via_hub(monkeypatch):
    fake = [
        _DatasetInfo("user/ds1", downloads=100, likes=5, tags=["nlp"], description="First"),
        _DatasetInfo("user/ds2", downloads=50, likes=2, tags=["vision"], description="Second"),
    ]

    def fake_via_hub(query, method, limit, **kwargs):
        assert method == "list_datasets"
        return fake

    monkeypatch.setattr(hf_search, "_via_hub_api", fake_via_hub)
    results = hf_search.search_datasets("foo", limit=5)
    assert len(results) == 2
    assert results[0]["id"] == "user/ds1"
    assert results[0]["downloads"] == 100
    assert results[0]["tags"] == ["nlp"]


def test_search_models_normalizes(monkeypatch):
    fake = [_ModelInfo("org/m1", downloads=1000, pipeline_tag="text-generation", likes=10)]
    monkeypatch.setattr(hf_search, "_via_hub_api", lambda *a, **kw: fake)
    results = hf_search.search_models("foo", limit=3)
    assert results[0]["id"] == "org/m1"
    assert results[0]["pipeline_tag"] == "text-generation"


def test_search_spaces(monkeypatch):
    fake = [_SpaceInfo("user/s1", sdk="gradio", likes=7)]
    monkeypatch.setattr(hf_search, "_via_hub_api", lambda *a, **kw: fake)
    results = hf_search.search_spaces("foo")
    assert results[0]["sdk"] == "gradio"


def test_docs_http_failure_returns_empty(monkeypatch):
    def failing_open(*args, **kwargs):
        raise __import__("urllib").error.URLError("no network")

    monkeypatch.setattr("urllib.request.urlopen", failing_open)
    results = hf_search.search_docs("foo")
    assert results == []


def test_multi_angle_uses_expansion(monkeypatch):
    monkeypatch.setattr(
        "crucible.researcher.literature.expand_query",
        lambda q: ["foo", "bar", "baz"],
    )
    call_log: list[str] = []

    def fake_datasets(q, limit=10):
        call_log.append(q)
        return [{"id": f"ds-{q}-{i}"} for i in range(3)]

    monkeypatch.setattr(hf_search, "search_datasets", fake_datasets)
    results = hf_search.search("datasets", "foo", limit=9, multi_angle=True)
    # Each angle contributes
    assert len(results) == 9
    assert len(set(r["id"] for r in results)) == 9
    assert call_log == ["foo", "bar", "baz"]


def test_multi_angle_skipped_for_docs(monkeypatch):
    """Docs search bypasses multi-angle (no stable ID to dedup on)."""
    call = {"n": 0}

    def fake_docs(q, limit=10):
        call["n"] += 1
        return [{"id": "doc-1"}]

    monkeypatch.setattr(hf_search, "search_docs", fake_docs)
    hf_search.search("docs", "foo", multi_angle=True)
    assert call["n"] == 1


# --- Test doubles ----------------------------------------------------------


class _DatasetInfo:
    def __init__(self, id, downloads, likes, tags, description):
        self.id = id
        self.downloads = downloads
        self.likes = likes
        self.tags = tags
        self.description = description
        self.last_modified = None


class _ModelInfo:
    def __init__(self, id, downloads, pipeline_tag, likes):
        self.id = id
        self.downloads = downloads
        self.pipeline_tag = pipeline_tag
        self.likes = likes
        self.tags = []
        self.last_modified = None


class _SpaceInfo:
    def __init__(self, id, sdk, likes):
        self.id = id
        self.sdk = sdk
        self.likes = likes
        self.tags = []
        self.last_modified = None
