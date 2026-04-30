"""HuggingFace Hub write helpers — push/pull/ensure for the collab backbone.

Used by:
  - hub_remotes.py (hf_dataset remote backend)
  - mcp/tools.py (hf_push_artifact, hf_publish_leaderboard, etc.)

All entry points lazy-import ``huggingface_hub`` so the module is safe to
import even when the package isn't installed. Failures wrap as ``HfError``
(a ``HubError`` subclass) so callers can treat them uniformly with
existing hub flows.

Auth: ``HF_TOKEN`` env var, optionally overridden by an explicit ``token``
argument. We never persist tokens.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from crucible.core.errors import HfError


def resolve_token(token: str | None = None) -> str | None:
    """Return ``token`` if non-empty else env ``HF_TOKEN`` else ``None``."""
    if token:
        return token
    env = os.environ.get("HF_TOKEN", "").strip()
    return env or None


def _import_hub() -> Any:
    try:
        import huggingface_hub  # type: ignore
    except ImportError as exc:
        raise HfError(
            "huggingface_hub is not installed. Add it to your environment "
            "(pip install huggingface_hub) before using hf_collab tools."
        ) from exc
    return huggingface_hub


def _wrap(action: str, exc: BaseException) -> HfError:
    """Normalize HF SDK exceptions to ``HfError`` with the originating
    type name preserved in the message for easier debugging."""
    return HfError(f"hf_{action} failed [{type(exc).__name__}]: {exc}")


def ensure_repo(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    private: bool = True,
    token: str | None = None,
    exist_ok: bool = True,
) -> str:
    """Idempotently create a repo on the HF Hub. Returns ``repo_id``.

    Raises ``HfError`` for auth / network / validation failures.
    """
    hub = _import_hub()
    api = hub.HfApi(token=resolve_token(token))
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=exist_ok,
        )
    except Exception as exc:  # broad — HF SDK raises many subtypes
        raise _wrap("ensure_repo", exc) from exc
    return repo_id


def push_file(
    local_path: str | Path,
    repo_id: str,
    *,
    path_in_repo: str | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a single file. Returns the URL/path returned by HF.

    ``path_in_repo`` defaults to the file basename.
    """
    hub = _import_hub()
    p = Path(local_path)
    if not p.is_file():
        raise HfError(f"hf_push_file: local file not found: {local_path}")
    target = path_in_repo or p.name
    try:
        return hub.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=target,
            repo_id=repo_id,
            repo_type=repo_type,
            token=resolve_token(token),
            commit_message=commit_message or f"crucible: push {target}",
        )
    except Exception as exc:
        raise _wrap("push_file", exc) from exc


def push_folder(
    local_dir: str | Path,
    repo_id: str,
    *,
    path_in_repo: str | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
    commit_message: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
) -> str:
    """Upload a folder to a HF repo. Returns the URL/path returned by HF."""
    hub = _import_hub()
    d = Path(local_dir)
    if not d.is_dir():
        raise HfError(f"hf_push_folder: local dir not found: {local_dir}")
    try:
        return hub.upload_folder(
            folder_path=str(d),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=resolve_token(token),
            commit_message=commit_message or f"crucible: push {d.name}",
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        )
    except Exception as exc:
        raise _wrap("push_folder", exc) from exc


def pull_file(
    repo_id: str,
    filename: str,
    *,
    dest: str | Path | None = None,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a single file from a HF repo. Returns the local path."""
    hub = _import_hub()
    try:
        local = hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=resolve_token(token),
            local_dir=str(dest) if dest else None,
        )
    except Exception as exc:
        raise _wrap("pull_file", exc) from exc
    return Path(local)


def pull_folder(
    repo_id: str,
    *,
    dest: str | Path,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
    allow_patterns: Iterable[str] | None = None,
) -> Path:
    """Snapshot-download a HF repo into ``dest``. Returns the local path."""
    hub = _import_hub()
    try:
        local = hub.snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=resolve_token(token),
            local_dir=str(dest),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
    except Exception as exc:
        raise _wrap("pull_folder", exc) from exc
    return Path(local)


def list_files(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
) -> list[str]:
    """List files in a HF repo. Returns list of paths."""
    hub = _import_hub()
    try:
        return list(
            hub.list_repo_files(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=resolve_token(token),
            )
        )
    except Exception as exc:
        raise _wrap("list_files", exc) from exc
