"""HuggingFace Discussions — peer-agent comm channel.

Read (``list_discussions``) and write (``post_discussion``) for HF repo
Discussions tabs. Lives separate from :mod:`crucible.researcher.hf_search`
because that module is scoped to ecosystem *search* (datasets/models/spaces/
docs/prior_runs) — Discussions are a different surface (an inbox of agent-
to-agent messages on a specific repo, not a directory you query by tag).

Best-effort policy:
  - ``list_discussions`` returns ``[]`` on missing SDK or network failure
    + ``log_warn``. Read-side; never blocks the research loop.
  - ``post_discussion`` raises :class:`crucible.core.errors.HfError` on
    failure. Write-side; callers (MCP tools) wrap to ``{"error": ...}``.
"""
from __future__ import annotations

from typing import Any

from crucible.core.log import log_warn


def _attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def list_discussions(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    status: str = "all",
    limit: int = 50,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Return open / closed / all discussions on a HF repo.

    Best-effort: returns ``[]`` on missing SDK or network failure.
    Each entry: ``{num, title, status, author, created_at, url}``.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        log_warn("list_discussions: huggingface_hub not installed")
        return []
    try:
        from crucible.core.hf_writer import resolve_token

        api = HfApi(token=resolve_token(token))
        get = getattr(api, "get_repo_discussions", None)
        if get is None:
            log_warn("list_discussions: HfApi.get_repo_discussions unavailable")
            return []
        kwargs: dict[str, Any] = {"repo_id": repo_id, "repo_type": repo_type}
        if status in ("open", "closed"):
            kwargs["discussion_status"] = status
        raw = list(get(**kwargs))
    except Exception as exc:
        log_warn(f"list_discussions: failed for {repo_id!r}: {exc}")
        return []

    out: list[dict[str, Any]] = []
    for d in raw[:limit]:
        out.append({
            "num": int(_attr_or_key(d, "num") or 0),
            "title": str(_attr_or_key(d, "title") or ""),
            "status": str(_attr_or_key(d, "status") or ""),
            "author": str(_attr_or_key(d, "author") or ""),
            "created_at": str(_attr_or_key(d, "created_at") or ""),
            "url": str(_attr_or_key(d, "url") or ""),
            "is_pull_request": bool(_attr_or_key(d, "is_pull_request") or False),
        })
    return out


def post_discussion(
    repo_id: str,
    *,
    title: str,
    description: str,
    repo_type: str = "dataset",
    token: str | None = None,
) -> dict[str, Any]:
    """Open a new discussion on a HF repo. Returns ``{num, url, title}``.

    Raises ``crucible.core.errors.HfError`` on auth / network / API failure
    — callers (MCP tools) convert that to a structured error response.
    """
    from crucible.core.errors import HfError
    from crucible.core.hf_writer import resolve_token

    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise HfError(
            "huggingface_hub not installed; cannot post discussion."
        ) from exc

    try:
        api = HfApi(token=resolve_token(token))
        create = getattr(api, "create_discussion", None)
        if create is None:
            raise HfError("HfApi.create_discussion unavailable in this huggingface_hub version.")
        d = create(repo_id=repo_id, repo_type=repo_type, title=title, description=description)
    except HfError:
        raise
    except Exception as exc:
        raise HfError(f"post_discussion failed [{type(exc).__name__}]: {exc}") from exc

    return {
        "num": int(_attr_or_key(d, "num") or 0),
        "title": str(_attr_or_key(d, "title") or title),
        "url": str(_attr_or_key(d, "url") or ""),
    }
