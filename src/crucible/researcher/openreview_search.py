"""OpenReview search — pull paper records from the OpenReview API.

Phase 3.2 of the ecosystem-connections plan. Complements
:mod:`crucible.researcher.arxiv_search` (preprints) with the
peer-review record from venues like ICLR / NeurIPS / TMLR. Useful
when the autonomous loop wants to weight papers by review signal
(e.g., "show me VICReg-related work accepted at ICLR 2024+").

OpenReview's public REST API at ``https://api.openreview.net``
serves ``/notes/search`` for full-text search. No auth is required
for public papers; private venues need an account token (passed via
``OPENREVIEW_TOKEN`` env), but those aren't relevant to the
default research-loop use case.

All network/parse failures degrade gracefully to ``[]`` with a
``log_warn`` — the research loop never blocks on OpenReview
availability.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC
from typing import Any

from crucible.core.log import log_warn
from crucible.researcher.literature import multi_angle_dedup

_OR_FAILURES = (
    urllib.error.URLError,
    TimeoutError,
    json.JSONDecodeError,
    OSError,
    ValueError,
    AttributeError,
    KeyError,
)

_OR_API_V2 = "https://api2.openreview.net"
_OR_API_V1 = "https://api.openreview.net"


def search_openreview(
    query: str,
    *,
    limit: int = 10,
    venue: str | None = None,
    year: int | None = None,
    multi_angle: bool = False,
) -> list[dict[str, Any]]:
    """Search OpenReview via the public ``/notes/search`` endpoint.

    Parameters
    ----------
    query:
        Free-text query against title/abstract.
    limit:
        Max records to return.
    venue:
        Optional venue filter (e.g., ``"ICLR.cc/2024/Conference"``).
        OpenReview venue IDs are namespaced; see
        ``https://docs.openreview.net/`` for the full list.
    year:
        Optional year filter applied to the venue field if no explicit
        venue was given (best-effort regex match).
    multi_angle:
        When True, expand the query via the existing literature.py
        helper (cross-domain synonyms, application framings) and dedup
        across angles.
    """
    query = (query or "").strip()
    if not query:
        return []

    def _one_angle(q: str) -> list[dict[str, Any]]:
        return _fetch(q, limit=limit, venue=venue, year=year)

    if multi_angle:
        return multi_angle_dedup(
            query,
            search_fn=_one_angle,
            limit=limit,
            id_field="openreview_id",
        )
    return _one_angle(query)


def _fetch(
    query: str,
    *,
    limit: int,
    venue: str | None,
    year: int | None,
) -> list[dict[str, Any]]:
    """Single API round-trip. Returns ``[]`` on any failure.

    Tries the v2 API first (newer, current default for ICLR 2024+),
    falls back to v1 on 404/missing-endpoint. Many OpenReview venues
    use v2 exclusively; older ones still serve from v1.
    """
    for base in (_OR_API_V2, _OR_API_V1):
        params = {
            "term": query,
            "limit": str(max(1, int(limit))),
        }
        # ``source=forum`` filters out review notes / comments and
        # returns only top-level paper submissions. The v2 API supports
        # this; v1 doesn't and may silently return empty results when
        # an unknown param is passed, so we scope the filter to v2.
        if base == _OR_API_V2:
            params["source"] = "forum"
        if venue:
            params["venue"] = venue
        url = f"{base}/notes/search?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "crucible-openreview-search/1.0",
                    "Accept": "application/json",
                },
            )
            # Optional bearer token for restricted venues.
            token = os.environ.get("OPENREVIEW_TOKEN")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read())
        except _OR_FAILURES as exc:
            log_warn(f"openreview_search: API call failed at {base}: {exc}")
            continue

        notes = payload.get("notes") if isinstance(payload, dict) else None
        if not isinstance(notes, list):
            continue
        out = [_normalize_note(n) for n in notes[:limit]]
        if year is not None:
            out = [r for r in out if str(year) in (r.get("venue") or "")]
        return out
    return []


def _normalize_note(note: dict[str, Any]) -> dict[str, Any]:
    """OpenReview note → normalized paper dict.

    Matches the field shape from arxiv_search/hf_search so callers
    walking multiple sources don't need per-source branching.
    OpenReview content lives under ``note.content``; values may be
    bare strings (v1) or ``{value: ..., readers: ...}`` envelopes (v2),
    so we resolve both shapes.
    """
    content = note.get("content") or {}
    title = _content_value(content, "title") or ""
    abstract = _content_value(content, "abstract") or ""
    authors = _content_value(content, "authors") or []
    if isinstance(authors, str):
        # v1 sometimes stores as comma-separated string.
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    keywords = _content_value(content, "keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    venue = _content_value(content, "venue") or note.get("venue", "")
    venueid = _content_value(content, "venueid") or note.get("venueid", "")

    note_id = note.get("id", "") or note.get("forum", "")
    url = f"https://openreview.net/forum?id={note_id}" if note_id else ""

    # Decision (accept/reject/etc.) lives in a separate forum reply;
    # the search endpoint includes the venue, which already encodes
    # acceptance for venues like ICLR.cc/2024/Conference.
    pdf_url = ""
    if isinstance(content.get("pdf"), dict):
        pdf_path = content["pdf"].get("value", "")
        if isinstance(pdf_path, str) and pdf_path:
            pdf_url = f"https://openreview.net{pdf_path}"

    return {
        # Stable identifier for dedup.
        "openreview_id": note_id,
        "id": note_id,
        "title": str(title).strip(),
        "summary": str(abstract).strip()[:500],
        "abstract": str(abstract).strip(),
        "authors": authors if isinstance(authors, list) else [],
        "keywords": keywords if isinstance(keywords, list) else [],
        "venue": str(venue or ""),
        "venueid": str(venueid or ""),
        "published_at": _ms_to_iso(note.get("cdate")) or _ms_to_iso(note.get("tcdate")),
        "url": url,
        "pdf_url": pdf_url,
        "source": "openreview",
    }


def _content_value(content: dict[str, Any], key: str) -> Any:
    """Extract a content value handling both API shapes.

    v1: ``content[key]`` is the raw value.
    v2: ``content[key]`` is ``{"value": ..., "readers": [...]}``.
    """
    raw = content.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _ms_to_iso(ms: Any) -> str:
    """OpenReview timestamps are epoch milliseconds. Return ISO-8601 UTC."""
    if not isinstance(ms, (int, float)):
        return ""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()
    except (ValueError, OSError):
        return ""
