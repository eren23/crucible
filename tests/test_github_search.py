"""Unit tests for researcher/github_search.py."""
from __future__ import annotations

import base64
import io
import json
import urllib.error

import pytest

from crucible.core.errors import ResearcherError
from crucible.researcher import github_search as gs


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_opener(payload):
    def _open(req, timeout=0):
        return _FakeResponse(payload)

    return _open


def test_search_code_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ResearcherError, match="GITHUB_TOKEN"):
        gs.search_code("foo")


def test_search_code_happy_path(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    payload = {
        "items": [
            {
                "repository": {"full_name": "a/b"},
                "path": "src/x.py",
                "html_url": "https://example.com/x",
                "sha": "deadbeef",
                "text_matches": [{"fragment": "def foo(): pass"}],
            },
        ],
    }
    monkeypatch.setattr("urllib.request.urlopen", _fake_opener(payload))

    out = gs.search_code("foo", language="python", limit=5)
    assert len(out) == 1
    assert out[0]["repo"] == "a/b"
    assert out[0]["path"] == "src/x.py"
    assert "def foo" in out[0]["match_snippets"][0]


def test_search_code_rate_limit_message(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    def raising(*args, **kwargs):
        raise urllib.error.HTTPError(
            "url",
            403,
            "Forbidden",
            hdrs={"X-RateLimit-Reset": "9999999"},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"message": "API rate limit exceeded"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", raising)
    with pytest.raises(ResearcherError, match="rate limit"):
        gs.search_code("foo")


def test_search_code_auth_failure(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "bad")

    def raising(*args, **kwargs):
        raise urllib.error.HTTPError(
            "url",
            401,
            "Unauthorized",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"message": "Bad credentials"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", raising)
    with pytest.raises(ResearcherError, match="auth failed"):
        gs.search_code("foo")


def test_list_repos_no_auth(monkeypatch):
    payload = {
        "items": [
            {
                "full_name": "a/b",
                "description": "demo",
                "stargazers_count": 42,
                "forks_count": 1,
                "language": "Python",
                "html_url": "https://example.com/a/b",
                "updated_at": "2026-01-01",
            },
        ],
    }
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _fake_opener(payload))
    out = gs.list_repos("foo")
    assert out[0]["stars"] == 42
    assert out[0]["language"] == "Python"


def test_read_file_repo_shape():
    with pytest.raises(ResearcherError, match="owner/name"):
        gs.read_file("bad-format", "README.md")


def test_read_file_base64_decoded(monkeypatch):
    raw_content = "hello world"
    payload = {
        "encoding": "base64",
        "content": base64.b64encode(raw_content.encode()).decode(),
        "size": len(raw_content),
        "html_url": "https://example.com/f",
    }
    monkeypatch.setattr("urllib.request.urlopen", _fake_opener(payload))
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    out = gs.read_file("a/b", "README.md")
    assert out["content"] == raw_content


def test_read_file_directory_raises(monkeypatch):
    """GitHub returns a list for directories; we reject it."""
    monkeypatch.setattr("urllib.request.urlopen", _fake_opener([{"name": "x"}]))
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    with pytest.raises(ResearcherError, match="directory"):
        gs.read_file("a/b", "src")
