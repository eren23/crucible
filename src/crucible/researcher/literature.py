"""Literature search via HuggingFace Papers API.

Enriches hypothesis generation with relevant published research.
All functions are best-effort -- failures return empty results,
never blocking the research loop.

Multi-angle search: a single query is expanded into 3-5 cross-domain
reformulations (synonyms, enabling mechanisms, applications) via a
cheap LLM call, then each angle is searched independently. This
captures papers that use different terminology for the same concept.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from crucible.core.log import log_info, log_warn


# Literature search failures are best-effort — network/parse/data-shape errors
# return empty results rather than blocking the research loop.
_LITERATURE_FAILURES = (
    urllib.error.URLError,
    TimeoutError,
    json.JSONDecodeError,
    OSError,
    KeyError,
    ValueError,
    AttributeError,
)

# Simple in-memory cache with 1-hour TTL
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL = 3600.0

# Expansion cache — keyed by original query, value is list of angle queries
_expansion_cache: dict[str, tuple[float, list[str]]] = {}


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
    except _LITERATURE_FAILURES as exc:
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
    except _LITERATURE_FAILURES as exc:
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


_EXPAND_SYSTEM = (
    "You are a research query expander. Given a search query about ML/AI, "
    "generate 3-5 alternative search queries that approach the same concept "
    "from different angles:\n"
    "- Cross-domain synonyms (e.g. 'weight sharing' -> 'parameter tying')\n"
    "- Enabling mechanisms (e.g. 'sparse attention' -> 'local attention patterns')\n"
    "- Application framing (e.g. 'model compression' -> 'efficient inference')\n"
    "- Adjacent fields (e.g. 'foveated rendering' in graphics for 'mixed-resolution tokens')\n\n"
    "Return ONLY a JSON array of query strings. No explanation."
)


def expand_query(query: str) -> list[str]:
    """Expand a search query into 3-5 cross-domain angles via LLM.

    Returns the original query plus expansions. Falls back to just the
    original query if the LLM is unavailable or fails.
    """
    if not query.strip():
        return []

    now = time.monotonic()
    if query in _expansion_cache:
        ts, cached = _expansion_cache[query]
        if now - ts < _CACHE_TTL:
            return cached

    angles = [query]
    try:
        import anthropic

        model = os.environ.get("EXPANSION_MODEL", "claude-haiku-4-5-20251001")
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=_EXPAND_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        text = response.content[0].text.strip()
        parsed = json.loads(text) if text.startswith("[") else json.loads(
            re.search(r"\[.*\]", text, re.DOTALL).group()  # type: ignore[union-attr]
        )
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str) and item.strip() and item.strip() != query:
                    angles.append(item.strip())
        log_info(f"Query expansion: {query!r} -> {len(angles)} angles")
    except ImportError:
        pass  # anthropic not installed
    except Exception as exc:  # noqa: BLE001
        # Query expansion is strictly best-effort: Anthropic SDK errors
        # (AnthropicError subclasses — not imported at module scope because
        # anthropic is an optional dep), malformed-JSON responses, and network
        # hiccups must all degrade to "use original query only".
        log_warn(f"Query expansion failed (non-fatal): {exc}")

    _expansion_cache[query] = (now, angles)
    return angles


def multi_angle_search(
    query: str, limit: int = 10, per_angle_limit: int = 5
) -> list[dict[str, Any]]:
    """Expand query into cross-domain angles, search each, dedupe results.

    Uses expand_query() for LLM-powered multi-angle reformulation, then
    searches each angle independently via search_papers(). Results are
    deduplicated by paper ID and capped at limit.
    """
    angles = expand_query(query)
    all_papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for angle in angles:
        for p in search_papers(angle, limit=per_angle_limit):
            pid = p.get("id", "")
            if pid and pid not in seen:
                seen.add(pid)
                all_papers.append(p)
    return all_papers[:limit]


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
