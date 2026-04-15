"""Literature search via HuggingFace Papers API.

Enriches hypothesis generation with relevant published research.
All functions are best-effort -- failures return empty results,
never blocking the research loop.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from crucible.core.log import log_warn


# Simple in-memory cache with 1-hour TTL
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL = 3600.0


def search_papers(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search HuggingFace papers.  Returns normalised paper dicts."""
    if not query.strip():
        return []

    cache_key = f"{query}:{limit}"
    now = time.monotonic()
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if now - ts < _CACHE_TTL:
            return cached

    papers = _search_via_hub(query, limit) or _search_via_api(query, limit)
    _cache[cache_key] = (now, papers)
    return papers


def _search_via_hub(query: str, limit: int) -> list[dict[str, Any]] | None:
    """Search using huggingface_hub SDK (preferred)."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        results = api.list_papers(query=query)
        papers = []
        for p in results:
            if len(papers) >= limit:
                break
            papers.append(_normalize_paper(p))
        return papers
    except ImportError:
        return None
    except Exception as exc:
        log_warn(f"Literature search via huggingface_hub failed: {exc}")
        return None


def _search_via_api(query: str, limit: int) -> list[dict[str, Any]]:
    """Fallback: direct HTTP to HF papers API."""
    try:
        encoded = urllib.request.quote(query)
        url = f"https://huggingface.co/api/papers/search?q={encoded}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        items = data if isinstance(data, list) else data.get("papers", data.get("results", []))
        papers = []
        for item in items:
            if len(papers) >= limit:
                break
            papers.append(
                {
                    "id": item.get("id", item.get("paperId", "")),
                    "title": item.get("title", ""),
                    "summary": (item.get("summary") or item.get("abstract") or "")[:300],
                    "ai_summary": item.get("ai_summary", ""),
                    "upvotes": item.get("upvotes", 0),
                    "published_at": item.get("publishedAt", item.get("published_at", "")),
                    "github_repo": item.get("github_repo", ""),
                    "keywords": item.get("ai_keywords", []),
                }
            )
        return papers
    except Exception as exc:
        log_warn(f"Literature search via HTTP API failed: {exc}")
        return []


def _normalize_paper(paper: Any) -> dict[str, Any]:
    """Normalise an HfApi PaperInfo object to a plain dict."""
    return {
        "id": getattr(paper, "id", ""),
        "title": getattr(paper, "title", ""),
        "summary": (getattr(paper, "summary", "") or "")[:300],
        "ai_summary": getattr(paper, "ai_summary", "") or "",
        "upvotes": getattr(paper, "upvotes", 0),
        "published_at": str(getattr(paper, "published_at", "")),
        "github_repo": getattr(paper, "github_repo", "") or "",
        "keywords": getattr(paper, "ai_keywords", []) or [],
    }


def get_paper_detail(paper_id: str) -> dict[str, Any] | None:
    """Fetch single paper metadata."""
    if not paper_id.strip():
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        p = api.paper_info(paper_id)
        return _normalize_paper(p)
    except ImportError:
        pass
    except Exception as exc:
        log_warn(f"Paper detail fetch via huggingface_hub failed for {paper_id!r}: {exc}")
    try:
        url = f"https://huggingface.co/api/papers/{urllib.request.quote(paper_id)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return {
            "id": data.get("id", paper_id),
            "title": data.get("title", ""),
            "summary": (data.get("summary") or "")[:500],
            "ai_summary": data.get("ai_summary", ""),
            "upvotes": data.get("upvotes", 0),
            "published_at": data.get("publishedAt", ""),
            "github_repo": data.get("github_repo", ""),
            "keywords": data.get("ai_keywords", []),
        }
    except Exception as exc:
        log_warn(f"Paper detail fetch via HTTP API failed for {paper_id!r}: {exc}")
        return None


def format_literature_context(
    papers: list[dict[str, Any]], max_papers: int = 5
) -> str:
    """Format papers into compact markdown for LLM context injection."""
    if not papers:
        return ""
    lines = ["## Related Literature"]
    for p in papers[:max_papers]:
        title = p.get("title", "Untitled")
        pid = p.get("id", "")
        summary = p.get("ai_summary") or p.get("summary", "")
        summary = summary[:150].rstrip() + ("..." if len(summary) > 150 else "")
        upvotes = p.get("upvotes", 0)
        keywords = ", ".join(p.get("keywords", [])[:5])
        lines.append(f"- **{title}** (arxiv:{pid}, {upvotes} upvotes)")
        if summary:
            lines.append(f"  {summary}")
        if keywords:
            lines.append(f"  Keywords: {keywords}")
    return "\n".join(lines)


def suggest_queries(
    program_text: str,
    beliefs: list[str],
    findings: list[dict[str, Any]],
) -> list[str]:
    """Extract 2-4 search queries from research state.  No LLM call."""
    queries: list[str] = []

    # Extract model family names from program text
    families = re.findall(r'MODEL_FAMILY["\s:=]+(\w+)', program_text)
    if families:
        queries.append(f"{families[0]} neural network architecture")

    # Extract key terms from beliefs
    stop = {
        "that", "with", "this", "from", "have", "been", "more", "than",
        "does", "when", "what", "each", "which", "their", "about",
    }
    keywords: set[str] = set()
    for belief in beliefs[:5]:
        for word in re.findall(r"\b[a-z]{4,}\b", belief.lower()):
            if word not in stop:
                keywords.add(word)
    if keywords:
        top = sorted(keywords)[:3]
        queries.append(" ".join(top) + " machine learning")

    # Extract from findings
    for f in findings[:10]:
        finding_text = f.get("finding", f.get("title", ""))
        if finding_text:
            words = [w for w in finding_text.split()[:6] if len(w) > 3]
            if words:
                queries.append(" ".join(words[:3]))

    if not queries:
        queries = ["autonomous machine learning research"]

    return queries[:4]
