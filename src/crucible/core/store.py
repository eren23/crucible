"""Version store for experiment designs and research context.

Provides hybrid persistence: human-readable YAML files on disk with an
append-only JSONL ledger for fast indexed access. Optional git integration
for committing versions.

Layout under store_dir (default .crucible/):
    store.jsonl              # version ledger
    designs/{name}/v1.yaml   # experiment design versions
    designs/{name}/current.yaml
    context/{name}/v1.yaml   # research context versions
    context/{name}/current.yaml
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from crucible.core.errors import StoreError
from crucible.core.io import append_jsonl, read_jsonl
from crucible.core.log import utc_now_iso


# Map resource_type to subdirectory name
_TYPE_DIRS: dict[str, str] = {
    "experiment_design": "designs",
    "research_context": "context",
}


class VersionStore:
    """Hybrid version store: YAML files + JSONL ledger."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)
        self._ledger_path = self.store_dir / "store.jsonl"
        # In-memory index: (resource_type, resource_name) -> list[VersionMeta]
        self._index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Rebuild in-memory index from the JSONL ledger."""
        self._index.clear()
        for entry in read_jsonl(self._ledger_path):
            if entry.get("kind") != "version":
                continue
            meta = entry["data"]
            key = (meta["resource_type"], meta["resource_name"])
            self._index.setdefault(key, []).append(meta)

    def _next_version(self, resource_type: str, resource_name: str) -> int:
        """Return the next version number for a resource."""
        key = (resource_type, resource_name)
        versions = self._index.get(key, [])
        if not versions:
            return 1
        return max(v["version"] for v in versions) + 1

    # ------------------------------------------------------------------
    # YAML file operations
    # ------------------------------------------------------------------

    def _resource_dir(self, resource_type: str, resource_name: str) -> Path:
        subdir = _TYPE_DIRS.get(resource_type)
        if subdir is None:
            raise StoreError(f"Unknown resource type: {resource_type}")
        return self.store_dir / subdir / resource_name

    def _write_yaml(self, path: Path, content: dict[str, Any]) -> None:
        """Write content as YAML, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.dump(content, default_flow_style=False, sort_keys=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8")

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        """Read a YAML file and return its contents."""
        if not path.exists():
            raise StoreError(f"Version file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise StoreError(f"Invalid YAML content in {path}")
        return raw

    @staticmethod
    def _checksum(content: dict[str, Any]) -> str:
        """SHA-256 checksum of YAML-serialized content (first 16 hex chars)."""
        text = yaml.dump(content, default_flow_style=False, sort_keys=True, allow_unicode=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        resource_type: str,
        resource_name: str,
        content: dict[str, Any],
        *,
        summary: str,
        created_by: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new version of a resource. Returns the VersionMeta dict."""
        if resource_type not in _TYPE_DIRS:
            raise StoreError(f"Unknown resource type: {resource_type}")

        version = self._next_version(resource_type, resource_name)
        key = (resource_type, resource_name)
        prev_versions = self._index.get(key, [])
        parent_id = prev_versions[-1]["version_id"] if prev_versions else None

        # Capture git state (best-effort)
        git_sha = self._safe_git_sha()

        version_id = f"{resource_type}/{resource_name}@v{version}"
        meta: dict[str, Any] = {
            "resource_type": resource_type,
            "resource_name": resource_name,
            "version": version,
            "version_id": version_id,
            "created_at": utc_now_iso(),
            "created_by": created_by,
            "parent_version_id": parent_id,
            "git_sha": git_sha,
            "git_committed": False,
            "summary": summary,
            "tags": tags or [],
            "checksum": self._checksum(content),
        }

        # Append to ledger first (source of truth)
        append_jsonl(self._ledger_path, {"kind": "version", "ts": meta["created_at"], "data": meta})

        # Write YAML files
        res_dir = self._resource_dir(resource_type, resource_name)
        version_path = res_dir / f"v{version}.yaml"
        current_path = res_dir / "current.yaml"
        self._write_yaml(version_path, content)
        shutil.copy2(version_path, current_path)

        # Update in-memory index
        self._index.setdefault(key, []).append(meta)

        return meta

    def get_current(
        self, resource_type: str, resource_name: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Get the latest version of a resource. Returns (meta, content) or None."""
        key = (resource_type, resource_name)
        versions = self._index.get(key)
        if not versions:
            return None
        meta = versions[-1]
        res_dir = self._resource_dir(resource_type, resource_name)
        current_path = res_dir / "current.yaml"
        if not current_path.exists():
            return None
        content = self._read_yaml(current_path)
        return meta, content

    def get_version(
        self, version_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Get a specific version by version_id. Returns (meta, content) or None."""
        for versions in self._index.values():
            for meta in versions:
                if meta["version_id"] == version_id:
                    res_dir = self._resource_dir(meta["resource_type"], meta["resource_name"])
                    version_path = res_dir / f"v{meta['version']}.yaml"
                    if not version_path.exists():
                        return None
                    content = self._read_yaml(version_path)
                    return meta, content
        return None

    def get_version_number(
        self, resource_type: str, resource_name: str, version: int
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Get a specific version by number. Returns (meta, content) or None."""
        key = (resource_type, resource_name)
        versions = self._index.get(key, [])
        for meta in versions:
            if meta["version"] == version:
                res_dir = self._resource_dir(resource_type, resource_name)
                version_path = res_dir / f"v{version}.yaml"
                if not version_path.exists():
                    return None
                content = self._read_yaml(version_path)
                return meta, content
        return None

    def list_resources(
        self,
        resource_type: str,
        *,
        status: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """List the latest version metadata for all resources of a type."""
        results: list[dict[str, Any]] = []
        for (rtype, _rname), versions in self._index.items():
            if rtype != resource_type:
                continue
            if not versions:
                continue
            latest = versions[-1]

            # Apply filters
            if tag and tag not in latest.get("tags", []):
                continue

            # Status filter requires reading the YAML content
            if status:
                res_dir = self._resource_dir(rtype, latest["resource_name"])
                current_path = res_dir / "current.yaml"
                if current_path.exists():
                    content = self._read_yaml(current_path)
                    if content.get("status") != status:
                        continue
                else:
                    continue

            results.append(latest)

        results.sort(key=lambda m: m.get("created_at", ""))
        return results

    def history(
        self, resource_type: str, resource_name: str
    ) -> list[dict[str, Any]]:
        """Get full version history for a resource (oldest first)."""
        key = (resource_type, resource_name)
        return list(self._index.get(key, []))

    # ------------------------------------------------------------------
    # Git integration
    # ------------------------------------------------------------------

    def _safe_git_sha(self) -> str | None:
        """Get current git SHA, or None if not in a git repo."""
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.store_dir.parent if self.store_dir.exists() else Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None
        sha = proc.stdout.strip()
        return sha or None

    def git_commit_version(self, meta: dict[str, Any]) -> str | None:
        """Create a git commit for a specific version. Returns commit SHA or None."""
        res_dir = self._resource_dir(meta["resource_type"], meta["resource_name"])
        version_path = res_dir / f"v{meta['version']}.yaml"
        current_path = res_dir / "current.yaml"

        project_root = self.store_dir.parent
        files_to_add = [
            str(version_path.relative_to(project_root)),
            str(current_path.relative_to(project_root)),
            str(self._ledger_path.relative_to(project_root)),
        ]

        try:
            for f in files_to_add:
                subprocess.run(
                    ["git", "add", f],
                    cwd=project_root,
                    capture_output=True,
                    check=True,
                )
            msg = (
                f"crucible: {meta['resource_type']} {meta['resource_name']} v{meta['version']}\n\n"
                f"{meta.get('summary', '')}\n\n"
                f"Agent: {meta.get('created_by', 'unknown')}\n"
                f"Version-Id: {meta['version_id']}"
            )
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=project_root,
                capture_output=True,
                check=True,
            )
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            commit_sha = proc.stdout.strip()
            meta["git_committed"] = True
            meta["commit_sha"] = commit_sha
            return commit_sha
        except Exception:
            return None
