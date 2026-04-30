"""HuggingFace ecosystem search: datasets, models, spaces, docs.

Complements :mod:`crucible.researcher.literature` (which covers HF
Papers). Reuses the multi-angle query-expansion helper so the searcher
matches cross-domain synonyms / application framings.

All failures are best-effort: they degrade to empty results and emit a
warn. The research loop never blocks on HF availability.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Literal

from crucible.core.log import log_warn
from crucible.researcher.literature import multi_angle_dedup


_HF_FAILURES = (
    urllib.error.URLError,
    TimeoutError,
    json.JSONDecodeError,
    OSError,
    KeyError,
    ValueError,
    AttributeError,
)

SearchKind = Literal["datasets", "models", "spaces", "docs"]
_VALID_KINDS: tuple[SearchKind, ...] = ("datasets", "models", "spaces", "docs")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search(
    kind: SearchKind,
    query: str,
    limit: int = 10,
    *,
    multi_angle: bool = False,
) -> list[dict[str, Any]]:
    """Search HF datasets, models, spaces, or docs.

    When ``multi_angle`` is True the query is first expanded via
    :func:`~crucible.researcher.literature.expand_query` and each angle
    searched independently (cross-domain synonyms + dedup).
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unknown HF search kind: {kind!r}. Valid: {_VALID_KINDS}")
    if not query.strip():
        return []

    # Resolve at call time so tests can monkeypatch search_datasets/etc.
    fn = _resolve(kind)
    if multi_angle and kind != "docs":
        return multi_angle_dedup(
            query,
            search_fn=fn,
            dedup_key=lambda r: str(r.get("id", "")),
            limit=limit,
            per_angle_limit=max(3, limit // 2),
        )
    return fn(query, limit=limit)


def _resolve(kind: SearchKind) -> Callable[..., list[dict[str, Any]]]:
    import sys

    mod = sys.modules[__name__]
    return getattr(mod, f"search_{kind}")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def search_datasets(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search the Hugging Face datasets hub."""
    raw = _via_hub_api(query, "list_datasets", limit) or _via_http(
        "https://huggingface.co/api/datasets", query, limit
    )
    return [_normalize_dataset(d) for d in raw][:limit]


def _normalize_dataset(d: Any) -> dict[str, Any]:
    get = _attr_or_key
    return {
        "id": str(get(d, "id") or ""),
        "downloads": int(get(d, "downloads") or 0),
        "likes": int(get(d, "likes") or 0),
        "tags": list(get(d, "tags") or []),
        "description": _short(get(d, "description") or ""),
        "last_modified": str(get(d, "last_modified") or get(d, "lastModified") or ""),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def search_models(
    query: str, limit: int = 10, task: str | None = None
) -> list[dict[str, Any]]:
    """Search the Hugging Face model hub."""
    raw = _via_hub_api(query, "list_models", limit, task=task) or _via_http(
        "https://huggingface.co/api/models", query, limit, extra={"filter": task} if task else None
    )
    return [_normalize_model(m) for m in raw][:limit]


def _normalize_model(m: Any) -> dict[str, Any]:
    get = _attr_or_key
    return {
        "id": str(get(m, "id") or get(m, "modelId") or ""),
        "downloads": int(get(m, "downloads") or 0),
        "likes": int(get(m, "likes") or 0),
        "pipeline_tag": str(get(m, "pipeline_tag") or ""),
        "tags": list(get(m, "tags") or []),
        "last_modified": str(get(m, "last_modified") or get(m, "lastModified") or ""),
    }


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def search_spaces(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search the Hugging Face spaces hub."""
    raw = _via_hub_api(query, "list_spaces", limit) or _via_http(
        "https://huggingface.co/api/spaces", query, limit
    )
    return [_normalize_space(s) for s in raw][:limit]


def _normalize_space(s: Any) -> dict[str, Any]:
    get = _attr_or_key
    sdk = get(s, "sdk") or ""
    return {
        "id": str(get(s, "id") or ""),
        "sdk": str(sdk),
        "likes": int(get(s, "likes") or 0),
        "tags": list(get(s, "tags") or []),
        "last_modified": str(get(s, "last_modified") or get(s, "lastModified") or ""),
    }


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


def search_docs(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search Hugging Face documentation.

    Uses the public documentation search endpoint. On failure returns an
    empty list; callers should treat docs search as best-effort.
    """
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://huggingface.co/api/docs/search?q={encoded}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except _HF_FAILURES as exc:
        log_warn(f"HF docs search failed: {exc}")
        return []

    items = data if isinstance(data, list) else data.get("results", data.get("hits", []))
    out: list[dict[str, Any]] = []
    for item in items[:limit]:
        out.append(
            {
                "id": str(item.get("id") or item.get("slug") or ""),
                "title": str(item.get("title") or item.get("heading1") or ""),
                "url": str(item.get("url") or item.get("link") or ""),
                "snippet": _short(item.get("text") or item.get("snippet") or ""),
                "product": str(item.get("product") or ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _via_hub_api(
    query: str,
    method: str,
    limit: int,
    *,
    task: str | None = None,
) -> list[Any] | None:
    """Use huggingface_hub.HfApi if importable."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    try:
        api = HfApi()
        fn = getattr(api, method, None)
        if fn is None:
            return None
        kwargs: dict[str, Any] = {"search": query, "limit": limit}
        if task and method == "list_models":
            kwargs["filter"] = task
        return list(fn(**kwargs))
    except _HF_FAILURES as exc:
        log_warn(f"HF {method} via huggingface_hub failed: {exc}")
        return None


def _via_http(
    base_url: str,
    query: str,
    limit: int,
    *,
    extra: dict[str, str] | None = None,
) -> list[Any]:
    """Fallback HTTP to a `search=...&limit=...` endpoint."""
    params = {"search": query, "limit": str(limit)}
    if extra:
        params.update({k: v for k, v in extra.items() if v})
    qs = urllib.parse.urlencode(params)
    url = f"{base_url}?{qs}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except _HF_FAILURES as exc:
        log_warn(f"HF HTTP fallback ({base_url}) failed: {exc}")
        return []
    if isinstance(data, list):
        return data
    return data.get("items", data.get("results", []))


def _attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _short(text: str, limit: int = 300) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Prior runs — read leaderboard.jsonl that peer agents published via
# ``hf_publish_leaderboard``. Different signature from search() (needs a
# repo_id, optional challenge filter, primary metric to sort by) so it
# lives as a top-level function rather than a SearchKind.
# ---------------------------------------------------------------------------


def fetch_prior_runs(
    repo_id: str,
    *,
    top_k: int = 10,
    challenge_id: str | None = None,
    primary_metric: str | None = None,
    direction: str = "minimize",
    revision: str | None = None,
    token: str | None = None,
    filename: str = "leaderboard.jsonl",
) -> list[dict[str, Any]]:
    """Pull a leaderboard JSONL from a HF Dataset repo and return the top-k rows.

    Best-effort: missing repo / network failure / malformed JSONL → ``[]``
    plus a ``log_warn``. Filters by ``challenge_id`` (case-insensitive substring
    match against ``name`` or ``challenge`` field) when supplied. Sorts by
    ``primary_metric`` if given (else uses each row's stated primary metric).
    """
    try:
        from crucible.core.hf_writer import pull_file
    except ImportError as exc:
        log_warn(f"fetch_prior_runs: hf_writer unavailable: {exc}")
        return []
    if not repo_id:
        return []

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        try:
            local = pull_file(
                repo_id=repo_id,
                filename=filename,
                dest=td,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except Exception as exc:
            # hf_writer wraps everything as HfError; treat any failure as
            # "no prior runs available" — never block the research loop.
            log_warn(f"fetch_prior_runs: pull from {repo_id!r} failed: {exc}")
            return []

        rows: list[dict[str, Any]] = []
        try:
            with open(local, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            log_warn(f"fetch_prior_runs: read {local!r} failed: {exc}")
            return []

    if challenge_id:
        needle = challenge_id.lower()
        rows = [
            r for r in rows
            if needle in str(r.get("challenge", "")).lower()
            or needle in str(r.get("name", "")).lower()
        ]

    if primary_metric:
        # Drop rows missing/non-numeric for the requested metric — keeping
        # them with an inf sentinel sorts them to the TOP under
        # direction='maximize' (since reverse=True), which is exactly
        # backwards. Filtering is the unambiguous behavior.
        def _metric(r: dict[str, Any]) -> float | None:
            v = r.get(primary_metric)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        scored = [(r, _metric(r)) for r in rows]
        rows = [r for r, m in scored if m is not None]
        rows.sort(
            key=lambda r: float(r[primary_metric]),
            reverse=(direction == "maximize"),
        )
    else:
        rows.sort(key=lambda r: int(r.get("rank") or 1_000_000))

    return rows[:top_k]


# Discussions live in :mod:`crucible.researcher.hf_discussions` — see that
# module for ``list_discussions`` and ``post_discussion``. They are a
# separate surface from ecosystem search (this module).
