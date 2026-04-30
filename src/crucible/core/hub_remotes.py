"""Pluggable hub remotes — where shared collab payloads live.

A *hub remote* is the destination for cross-machine / cross-agent
publication of leaderboards, findings, recipes, and artifacts. The
existing local hub (``~/.crucible-hub/``) is git-backed; this module
adds an abstraction so the same publish workflow can target HF
Datasets, future S3 buckets, IPFS, or anything with a folder-shaped
remote.

Builtins:
  - ``git``        — local hub directory; relies on ``HubStore.sync()``
                     for actual remote propagation. Useful as the
                     default "I'll push later" target.
  - ``hf_dataset`` — HuggingFace Dataset repo via ``hf_writer``.

Plugin family registers under :class:`PluginRegistry` so taps and
local plugins can drop in their own remotes via
``.crucible/plugins/hub_remotes/*.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from crucible.core.errors import HfError, HubError, PluginError
from crucible.core.plugin_registry import PluginRegistry


class HubRemote(ABC):
    """Abstract destination for collab payloads.

    Implementations are stateless factories: instantiated per call with
    a config dict, then ``push`` / ``pull`` / ``list_remote`` are invoked.
    Failures must propagate as ``HubError`` (or a subclass).
    """

    name: str = ""

    def __init__(self, **config: Any) -> None:
        self.config = config

    @abstractmethod
    def push(self, local_dir: str | Path, repo_id: str, **opts: Any) -> str:
        """Upload ``local_dir`` contents to ``repo_id``. Returns a remote
        ref (URL, commit hash, or path) for logging / verification."""
        ...

    @abstractmethod
    def pull(self, repo_id: str, dest: str | Path, **opts: Any) -> Path:
        """Download ``repo_id`` contents into ``dest``. Returns the
        local destination path."""
        ...

    @abstractmethod
    def list_remote(self, repo_id: str, **opts: Any) -> list[str]:
        """List file paths visible at ``repo_id``."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HUB_REMOTE_REGISTRY: PluginRegistry[type[HubRemote]] = PluginRegistry("hub_remote")


def register_hub_remote(name: str, cls: type[HubRemote], *, source: str = "builtin") -> None:
    """Register a hub remote backend."""
    HUB_REMOTE_REGISTRY.register(name, cls, source=source)


def build_hub_remote(name: str, **config: Any) -> HubRemote:
    """Instantiate a registered hub remote by name."""
    factory = HUB_REMOTE_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(HUB_REMOTE_REGISTRY.list_plugins()) or "(none)"
        raise PluginError(
            f"Unknown hub_remote {name!r}. Registered: {available}"
        )
    return factory(**config)


def list_hub_remotes() -> list[str]:
    """Return sorted list of registered hub remote names."""
    return HUB_REMOTE_REGISTRY.list_plugins()


# ---------------------------------------------------------------------------
# Builtin: git (local hub directory; relies on HubStore.sync for upload)
# ---------------------------------------------------------------------------


class GitHubRemote(HubRemote):
    """Stage payloads into the local hub directory; defer transport to
    ``HubStore.sync()``.

    ``repo_id`` is interpreted as a path *relative to the hub root*
    (e.g. ``leaderboards/parameter-golf``). ``push`` rsync-equivalents
    the local dir into that location; ``pull`` is the inverse local
    copy. Network propagation happens later via ``hub_sync``.
    """

    name = "git"

    def __init__(self, hub_dir: str | Path | None = None, **config: Any) -> None:
        super().__init__(**config)
        # Lazy import to avoid hub.py dependency at registration time.
        from crucible.core.hub import HubStore

        self._hub_dir = HubStore.resolve_hub_dir(explicit=hub_dir)

    def _resolve(self, repo_id: str) -> Path:
        if not repo_id:
            raise HubError("git_remote: repo_id must be non-empty")
        # Reject absolute paths — Path("/hub") / "/abs" returns "/abs", which
        # would let a caller escape the hub root entirely.
        candidate = Path(repo_id)
        if candidate.is_absolute() or candidate.drive:
            raise HubError(f"git_remote: repo_id must be relative, got {repo_id!r}")
        # Normalize and verify the result still lives under hub_dir, defending
        # against ".." traversal segments and tricky relative tricks.
        hub_root = self._hub_dir.resolve()
        target = (hub_root / candidate).resolve()
        try:
            target.relative_to(hub_root)
        except ValueError as exc:
            raise HubError(
                f"git_remote: repo_id {repo_id!r} resolves outside hub_dir"
            ) from exc
        return target

    def push(self, local_dir: str | Path, repo_id: str, **opts: Any) -> str:
        import shutil

        src = Path(local_dir)
        if not src.is_dir():
            raise HubError(f"git_remote.push: local_dir not found: {local_dir}")
        dst = self._resolve(repo_id)
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            target = dst / entry.name
            if entry.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)
        return str(dst)

    def pull(self, repo_id: str, dest: str | Path, **opts: Any) -> Path:
        import shutil

        src = self._resolve(repo_id)
        if not src.is_dir():
            raise HubError(f"git_remote.pull: repo_id not found locally: {repo_id}")
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            target = out / entry.name
            if entry.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)
        return out

    def list_remote(self, repo_id: str, **opts: Any) -> list[str]:
        src = self._resolve(repo_id)
        if not src.is_dir():
            return []
        return sorted(str(p.relative_to(src)) for p in src.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Builtin: hf_dataset (HuggingFace Dataset repo)
# ---------------------------------------------------------------------------


class HfDatasetHubRemote(HubRemote):
    """HuggingFace Dataset repo as a hub remote.

    Config:
      - ``repo_type``: ``"dataset"`` (default), ``"model"``, or ``"space"``
      - ``private``: bool, default True (matches ``HfCollabConfig.private``)
      - ``token``: optional override; falls back to ``HF_TOKEN`` env var
      - ``ensure``: bool, default True — call ``ensure_repo`` before push
    """

    name = "hf_dataset"

    def __init__(
        self,
        *,
        repo_type: str = "dataset",
        private: bool = True,
        token: str | None = None,
        ensure: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.repo_type = repo_type
        self.private = private
        self.token = token
        self.ensure = ensure

    def push(self, local_dir: str | Path, repo_id: str, **opts: Any) -> str:
        from crucible.core import hf_writer

        if self.ensure:
            hf_writer.ensure_repo(
                repo_id,
                repo_type=self.repo_type,
                private=self.private,
                token=self.token,
            )
        try:
            return hf_writer.push_folder(
                local_dir,
                repo_id,
                path_in_repo=opts.get("path_in_repo"),
                repo_type=self.repo_type,
                token=self.token,
                commit_message=opts.get("commit_message"),
                allow_patterns=opts.get("allow_patterns"),
                ignore_patterns=opts.get("ignore_patterns"),
            )
        except HfError:
            raise
        except Exception as exc:  # last-resort guard
            raise HfError(f"hf_dataset.push failed [{type(exc).__name__}]: {exc}") from exc

    def pull(self, repo_id: str, dest: str | Path, **opts: Any) -> Path:
        from crucible.core import hf_writer

        return hf_writer.pull_folder(
            repo_id,
            dest=dest,
            repo_type=self.repo_type,
            revision=opts.get("revision"),
            token=self.token,
            allow_patterns=opts.get("allow_patterns"),
        )

    def list_remote(self, repo_id: str, **opts: Any) -> list[str]:
        from crucible.core import hf_writer

        return hf_writer.list_files(
            repo_id,
            repo_type=self.repo_type,
            revision=opts.get("revision"),
            token=self.token,
        )


# ---------------------------------------------------------------------------
# Self-registration on import (idempotent — supports test reload + reimport)
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    for name, cls in (("git", GitHubRemote), ("hf_dataset", HfDatasetHubRemote)):
        if name not in HUB_REMOTE_REGISTRY:
            register_hub_remote(name, cls, source="builtin")


_register_builtins()
