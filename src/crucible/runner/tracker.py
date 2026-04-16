"""RunTracker: status sidecar and heartbeat writer for monitored experiment runs.

Each experiment gets three sidecar files in the logs directory:
  - <run_id>.txt          - captured stdout/stderr log
  - <run_id>.status.json  - live status (updated every heartbeat)
  - <run_id>.manifest.json - immutable run manifest (written at launch)

The status file is written atomically so external tools (CLI, fleet manager,
MCP server) can safely poll it without partial reads.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from crucible.core.io import atomic_write_json, _json_ready
from crucible.core.log import log_warn, utc_now_iso
from crucible.runner.fingerprint import code_fingerprint, safe_git_sha, safe_git_dirty


class RunTracker:
    """Maintains a status sidecar and run manifest for one experiment.

    Usage::

        tracker = RunTracker("exp_001", out_dir="logs", project_root=".")
        tracker.update(state="running", phase="launching", pid=12345)
        tracker.heartbeat("training", step=10, total_steps=100)
        tracker.finalize("completed", result={...})
    """

    def __init__(
        self,
        run_id: str,
        out_dir: str | Path = "logs",
        project_root: str | Path | None = None,
    ):
        self.run_id = run_id
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.out_dir / f"{run_id}.txt"
        self.status_path = self.out_dir / f"{run_id}.status.json"
        self.manifest_path = self.out_dir / f"{run_id}.manifest.json"

        # Load existing status or create fresh
        if self.status_path.exists():
            self._status = json.loads(
                self.status_path.read_text(encoding="utf-8")
            )
        else:
            self._status: dict[str, Any] = {
                "run_id": run_id,
                "state": "created",
                "phase": "created",
                "created_at": utc_now_iso(),
                "last_heartbeat_at": utc_now_iso(),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "log_path": str(self.log_path.resolve()),
                "status_path": str(self.status_path.resolve()),
                "manifest_path": str(self.manifest_path.resolve()),
            }
            atomic_write_json(self.status_path, self._status)

    def update(
        self,
        state: str | None = None,
        heartbeat: bool = True,
        **fields: Any,
    ) -> None:
        """Update the status sidecar with new fields.

        Args:
            state: New run state (e.g., "running", "retrying").
            heartbeat: Whether to bump last_heartbeat_at.
            **fields: Additional key-value pairs to merge in.
        """
        payload = dict(self._status)
        if state is not None:
            payload["state"] = state
        payload.update(_json_ready(fields))
        payload["updated_at"] = utc_now_iso()
        if heartbeat:
            payload["last_heartbeat_at"] = payload["updated_at"]
        self._status = payload
        atomic_write_json(self.status_path, payload)

    def heartbeat(self, phase: str, **fields: Any) -> None:
        """Shorthand for update with a phase change and heartbeat bump."""
        self.update(phase=phase, heartbeat=True, **fields)

    def finalize(self, state: str, **fields: Any) -> None:
        """Mark the run as finished with a terminal state."""
        self.update(state=state, ended_at=utc_now_iso(), **fields)

    def write_manifest(
        self,
        *,
        backend: str,
        script_path: str | Path,
        config: dict[str, Any],
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write (or update) the immutable run manifest.

        The manifest records everything needed to reproduce or audit a run:
        script path, config, git state, code fingerprint, etc.
        """
        extra = extra or {}
        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "backend": backend,
            "script_path": str(Path(script_path).resolve()),
            "project_root": str(self.project_root),
            "python": sys.executable,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "git_sha": safe_git_sha(self.project_root),
            "git_dirty": safe_git_dirty(self.project_root),
            "created_at": utc_now_iso(),
            "log_path": str(self.log_path.resolve()),
            "status_path": str(self.status_path.resolve()),
            "config": _json_ready(config),
            "tags": tags or [],
            "code_fingerprint": code_fingerprint(self.project_root),
            "parent_run_id": extra.pop("parent_run_id", None),
        }

        # Merge with existing manifest if re-writing (e.g., OOM retry)
        if self.manifest_path.exists():
            try:
                existing = json.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
                manifest = {**existing, **manifest}
            except (OSError, json.JSONDecodeError) as exc:
                log_warn(f"Failed to merge existing manifest {self.manifest_path}: {exc}")

        if extra:
            manifest.update(_json_ready(extra))
        atomic_write_json(self.manifest_path, manifest)

    @property
    def state(self) -> str:
        """Current run state."""
        return self._status.get("state", "unknown")

    @property
    def status(self) -> dict[str, Any]:
        """Full current status dict (read-only copy)."""
        return dict(self._status)
