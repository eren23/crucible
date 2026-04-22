"""GitHub code + repo search for the research loop.

Wraps the public GitHub REST API (``/search/code``, ``/search/repositories``,
``/repos/{owner}/{repo}/contents``). Requires ``GITHUB_TOKEN`` env for
code-search and higher rate limits. Errors are surfaced as
:class:`crucible.core.errors.ResearcherError` so the research loop can
decide whether to retry or give up.

The module is deliberately thin: no caching, no rate-limit scheduler.
Callers that loop over many queries should introduce their own
back-off. A single failure-response path handles 403 (rate limit) and
401 (bad token) with clear error messages.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from crucible.core.errors import ResearcherError
from crucible.core.log import log_warn


_GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_code(
    query: str,
    language: str | None = None,
    limit: int = 10,
    *,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Search GitHub for code matches.

    Requires ``GITHUB_TOKEN`` — the unauthenticated code-search endpoint
    is disabled. Raises :class:`ResearcherError` if the token is missing
    or the API returns a non-success status.
    """
    tok = token or os.environ.get("GITHUB_TOKEN")
    if not tok:
        raise ResearcherError(
            "GitHub code search requires GITHUB_TOKEN env var (unauthenticated /search/code is disabled)"
        )
    q = query if not language else f"{query} language:{language}"
    data = _request(
        "/search/code",
        {"q": q, "per_page": str(max(1, min(limit, 100)))},
        token=tok,
        accept="application/vnd.github.text-match+json",
    )
    items = data.get("items", []) if isinstance(data, dict) else []
    out: list[dict[str, Any]] = []
    for item in items[:limit]:
        repo = item.get("repository", {}) or {}
        out.append(
            {
                "repo": str(repo.get("full_name", "")),
                "path": str(item.get("path", "")),
                "url": str(item.get("html_url", "")),
                "sha": str(item.get("sha", "")),
                "match_snippets": _extract_text_matches(item.get("text_matches", [])),
            }
        )
    return out


def list_repos(
    query: str,
    limit: int = 10,
    *,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Search GitHub repositories."""
    tok = token or os.environ.get("GITHUB_TOKEN")
    data = _request(
        "/search/repositories",
        {"q": query, "per_page": str(max(1, min(limit, 100)))},
        token=tok,
        required_auth=False,
    )
    items = data.get("items", []) if isinstance(data, dict) else []
    return [
        {
            "full_name": str(i.get("full_name", "")),
            "description": _short(i.get("description") or ""),
            "stars": int(i.get("stargazers_count") or 0),
            "forks": int(i.get("forks_count") or 0),
            "language": str(i.get("language") or ""),
            "url": str(i.get("html_url", "")),
            "updated_at": str(i.get("updated_at", "")),
        }
        for i in items[:limit]
    ]


def read_file(
    repo: str,
    path: str,
    ref: str = "main",
    *,
    token: str | None = None,
) -> dict[str, Any]:
    """Read a single file from a GitHub repository.

    ``repo`` is ``owner/name``. Returns ``{path, ref, size, encoding, content}``.
    Binary files return raw base64 content in ``content``. Text files are
    decoded to UTF-8 best-effort.
    """
    if "/" not in repo:
        raise ResearcherError(f"read_file: repo must be 'owner/name', got {repo!r}")
    tok = token or os.environ.get("GITHUB_TOKEN")
    encoded_path = urllib.parse.quote(path.strip("/"))
    data = _request(
        f"/repos/{repo}/contents/{encoded_path}",
        {"ref": ref},
        token=tok,
        required_auth=False,
    )
    if isinstance(data, list):
        raise ResearcherError(f"read_file: {path!r} is a directory, not a file")
    encoding = data.get("encoding")
    raw = data.get("content", "") or ""
    if encoding == "base64":
        try:
            decoded = base64.b64decode(raw).decode("utf-8")
        except UnicodeDecodeError:
            decoded = raw  # Binary; keep base64
    else:
        decoded = raw
    return {
        "path": path,
        "ref": ref,
        "size": int(data.get("size") or 0),
        "encoding": encoding or "",
        "content": decoded,
        "url": str(data.get("html_url", "")),
    }


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _request(
    path: str,
    params: dict[str, str],
    *,
    token: str | None,
    accept: str = "application/vnd.github+json",
    required_auth: bool = True,
) -> Any:
    qs = urllib.parse.urlencode(params)
    url = f"{_GITHUB_API}{path}?{qs}" if qs else f"{_GITHUB_API}{path}"
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "crucible-researcher",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif required_auth:
        raise ResearcherError("GitHub endpoint requires GITHUB_TOKEN")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        if exc.code == 403 and "rate limit" in body.lower():
            reset = exc.headers.get("X-RateLimit-Reset") if hasattr(exc, "headers") else None
            raise ResearcherError(
                f"GitHub rate limit hit (reset={reset}). "
                "Authenticate via GITHUB_TOKEN or wait for reset."
            ) from exc
        if exc.code == 401:
            raise ResearcherError("GitHub auth failed — invalid GITHUB_TOKEN") from exc
        if exc.code == 404:
            raise ResearcherError(f"GitHub 404: {path}") from exc
        log_warn(f"GitHub API {path} returned {exc.code}: {body[:200]}")
        raise ResearcherError(f"GitHub API {exc.code} on {path}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ResearcherError(f"GitHub API call failed ({path}): {exc}") from exc
    return data


def _extract_text_matches(matches: list[Any]) -> list[str]:
    if not isinstance(matches, list):
        return []
    out: list[str] = []
    for m in matches[:3]:
        if isinstance(m, dict):
            frag = m.get("fragment", "")
            if frag:
                out.append(_short(str(frag), limit=200))
    return out


def _short(text: str, limit: int = 300) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
