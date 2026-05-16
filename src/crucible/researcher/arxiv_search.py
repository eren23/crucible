"""arXiv search — pull paper records from the public arXiv API.

Phase 3.1 of the ecosystem-connections plan. Mirrors
:mod:`crucible.researcher.hf_search` and
:mod:`crucible.researcher.literature` (which covers HF Papers). The
output normalizes to the same paper-dict shape so callers can
interleave arXiv + HF results without per-source branching.

arXiv exposes a free Atom-feed API at
``http://export.arxiv.org/api/query`` — no auth, no rate-limit auth,
just a polite ~3s/req throttle. Parsed via the standard-library
``xml.etree.ElementTree`` to avoid pulling in a new dependency.

All network failures degrade gracefully to ``[]`` with a ``log_warn``;
the research loop never blocks on arXiv availability.
"""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from crucible.core.log import log_warn
from crucible.researcher.literature import multi_angle_dedup


_ARXIV_FAILURES = (
    urllib.error.URLError,
    TimeoutError,
    ET.ParseError,
    OSError,
    ValueError,
    AttributeError,
)

_ARXIV_API = "http://export.arxiv.org/api/query"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# arXiv subject categories that matter for ML/AI research. Users can
# pass arbitrary categories; this list is just the default sort-key
# when caller doesn't specify.
DEFAULT_CATEGORIES = ("cs.LG", "cs.AI", "cs.CL", "stat.ML")


def search_arxiv(
    query: str,
    *,
    limit: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    categories: list[str] | None = None,
    multi_angle: bool = False,
) -> list[dict[str, Any]]:
    """Search arXiv via the public Atom-feed API.

    Parameters
    ----------
    query:
        Free-text query. Combined with optional ``categories`` filter
        as ``all:<query> AND (cat:cs.LG OR cat:cs.AI OR ...)``.
    limit:
        Max records to return (arXiv API hard cap is 30000; we keep
        the default modest for prompt budget).
    sort_by:
        ``"relevance"`` | ``"lastUpdatedDate"`` | ``"submittedDate"``.
    sort_order:
        ``"ascending"`` | ``"descending"``.
    categories:
        Optional subject-category filter. ``None`` returns all
        categories; pass an empty list to skip the filter explicitly.
    multi_angle:
        When True, expand the query via the existing literature.py
        helper (cross-domain synonyms, application framings) and
        dedup across angles. Adds 3-5x API round-trips so the
        wall-clock cost is non-trivial.
    """
    query = (query or "").strip()
    if not query:
        return []

    def _one_angle(q: str) -> list[dict[str, Any]]:
        return _fetch(q, limit=limit, sort_by=sort_by,
                      sort_order=sort_order, categories=categories)

    if multi_angle:
        return multi_angle_dedup(
            query,
            search_fn=_one_angle,
            limit=limit,
            id_field="arxiv_id",
        )
    return _one_angle(query)


def _build_search_query(query: str, categories: list[str] | None) -> str:
    """Build arXiv's ``search_query`` parameter.

    Wraps the free-text query in ``all:<...>`` and ANDs an optional
    category filter. arXiv expects URL-encoded form per its docs.
    """
    base = f"all:{query}"
    if categories is None:
        return base
    if not categories:
        return base
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    return f"{base} AND ({cat_clause})"


def _fetch(
    query: str,
    *,
    limit: int,
    sort_by: str,
    sort_order: str,
    categories: list[str] | None,
) -> list[dict[str, Any]]:
    """Single API round-trip. Returns ``[]`` on any network/parse failure."""
    try:
        params = {
            "search_query": _build_search_query(query, categories),
            "start": "0",
            "max_results": str(max(1, int(limit))),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        url = f"{_ARXIV_API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "crucible-arxiv-search/1.0",
                "Accept": "application/atom+xml",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read()
        return _parse_atom(body, limit)
    except _ARXIV_FAILURES as exc:
        log_warn(f"arxiv_search: API call failed for {query!r}: {exc}")
        return []


def _parse_atom(body: bytes, limit: int) -> list[dict[str, Any]]:
    """Parse arXiv's Atom feed into the standard paper-dict shape."""
    root = ET.fromstring(body)
    entries = root.findall("atom:entry", _NS)
    out: list[dict[str, Any]] = []
    for entry in entries[:limit]:
        out.append(_normalize_entry(entry))
    return out


def _normalize_entry(entry: ET.Element) -> dict[str, Any]:
    """One arXiv Atom <entry> → normalized paper dict.

    Matches the field set emitted by hf_search/literature so callers
    can interleave records without per-source branching. ``arxiv_id``
    is the stable identifier (e.g., ``"2105.04906v1"`` or
    ``"2105.04906"``); strip the version suffix for dedup.
    """
    id_url = _text(entry, "atom:id")  # http://arxiv.org/abs/2105.04906v1
    arxiv_id = id_url.rsplit("/", 1)[-1] if id_url else ""
    arxiv_id_base = arxiv_id.split("v")[0] if arxiv_id else ""
    title = _text(entry, "atom:title").replace("\n", " ").strip()
    summary = _text(entry, "atom:summary").replace("\n", " ").strip()
    published = _text(entry, "atom:published")
    updated = _text(entry, "atom:updated")
    authors = [
        _text(a, "atom:name")
        for a in entry.findall("atom:author", _NS)
        if _text(a, "atom:name")
    ]
    categories = [
        c.get("term", "")
        for c in entry.findall("atom:category", _NS)
        if c.get("term")
    ]
    # The HTML abstract page is more useful than the API entry URL.
    abs_url = ""
    for link in entry.findall("atom:link", _NS):
        if link.get("rel") == "alternate":
            abs_url = link.get("href", "")
            break
    if not abs_url and arxiv_id_base:
        abs_url = f"https://arxiv.org/abs/{arxiv_id_base}"

    return {
        # Stable identifier for dedup; tests/callers compare on this.
        "arxiv_id": arxiv_id_base or arxiv_id,
        "arxiv_id_versioned": arxiv_id,
        # Aliases matching hf_search/literature shape — caller code that
        # walks both sources doesn't need per-source branching.
        "id": arxiv_id_base or arxiv_id,
        "title": title,
        "summary": summary[:500],  # trim for prompt budget
        "abstract": summary,        # full text under explicit name
        "authors": authors,
        "categories": categories,
        "published_at": published,
        "updated_at": updated,
        "url": abs_url,
        "source": "arxiv",
    }


def _text(elem: ET.Element, path: str) -> str:
    node = elem.find(path, _NS)
    return (node.text or "").strip() if node is not None and node.text else ""
