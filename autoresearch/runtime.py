#!/usr/bin/env python3
"""Shared runtime utilities for monitored experiment runs."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def collect_public_attrs(obj: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        config[name] = _json_ready(value)
    return config


def safe_git_sha(project_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    sha = proc.stdout.strip()
    return sha or None


def safe_git_dirty(project_root: Path) -> bool | None:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


CODE_FILES = ("train_gpt.py", "autoresearch/torch_models.py")


def code_fingerprint(project_root: Path, extra_files: tuple[str, ...] = CODE_FILES) -> dict:
    """SHA-256 each key source file → combined fingerprint for dedup."""
    file_hashes = {}
    for rel in extra_files:
        p = project_root / rel
        if p.is_file():
            file_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    combined = hashlib.sha256(
        "|".join(f"{k}={v}" for k, v in sorted(file_hashes.items())).encode()
    ).hexdigest()[:16]
    return {"fingerprint": combined, "files": file_hashes}


class RunTracker:
    """Maintains a status sidecar and run manifest for one experiment."""

    def __init__(
        self,
        run_id: str,
        out_dir: str | Path = "logs",
        project_root: str | Path | None = None,
    ):
        self.run_id = run_id
        self.project_root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / f"{run_id}.txt"
        self.status_path = self.out_dir / f"{run_id}.status.json"
        self.manifest_path = self.out_dir / f"{run_id}.manifest.json"
        if self.status_path.exists():
            self._status = json.loads(self.status_path.read_text(encoding="utf-8"))
        else:
            self._status = {
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

    def update(self, state: str | None = None, heartbeat: bool = True, **fields: Any) -> None:
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
        self.update(phase=phase, heartbeat=True, **fields)

    def finalize(self, state: str, **fields: Any) -> None:
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
        extra = extra or {}
        manifest = {
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
        if self.manifest_path.exists():
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                manifest = {**existing, **manifest}
            except Exception:
                pass
        if extra:
            manifest.update(_json_ready(extra))
        atomic_write_json(self.manifest_path, manifest)


class WandbLogger:
    """Optional W&B logger that stays inert when WANDB_PROJECT is unset."""

    def __init__(self, run: Any | None = None, enabled: bool = False, error: str | None = None):
        self.run = run
        self.enabled = enabled
        self.error = error

    @property
    def url(self) -> str | None:
        if self.run is None:
            return None
        return getattr(self.run, "url", None)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        config: dict[str, Any],
        backend: str,
        tracker: RunTracker | None = None,
        job_type: str | None = None,
        tags: list[str] | None = None,
    ) -> "WandbLogger":
        project = os.environ.get("WANDB_PROJECT", "").strip()
        if not project:
            if tracker is not None:
                tracker.update(wandb={"enabled": False, "reason": "WANDB_PROJECT unset"}, heartbeat=False)
            return cls()
        try:
            import wandb  # type: ignore
        except ImportError:
            error = "wandb not installed"
            if tracker is not None:
                tracker.update(wandb={"enabled": False, "error": error}, heartbeat=False)
            return cls(error=error)

        env_tags = [tag.strip() for tag in os.environ.get("WANDB_TAGS", "").split(",") if tag.strip()]
        final_tags = list(dict.fromkeys([backend, *(tags or []), *env_tags]))
        kwargs = {
            "project": project,
            "entity": os.environ.get("WANDB_ENTITY") or None,
            "name": os.environ.get("WANDB_RUN_NAME", run_id),
            "group": os.environ.get("WANDB_RUN_GROUP") or None,
            "job_type": os.environ.get("WANDB_JOB_TYPE") or job_type,
            "tags": final_tags or None,
            "config": _json_ready(config),
            "mode": os.environ.get("WANDB_MODE") or None,
            "reinit": True,
        }
        run = wandb.init(**kwargs)
        logger = cls(run=run, enabled=True)
        if tracker is not None:
            tracker.update(
                wandb={
                    "enabled": True,
                    "url": logger.url,
                    "project": project,
                    "entity": os.environ.get("WANDB_ENTITY") or None,
                    "mode": os.environ.get("WANDB_MODE") or "online",
                },
                heartbeat=False,
            )
        return logger

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        payload = {k: v for k, v in _json_ready(metrics).items() if v is not None}
        if not payload:
            return
        if step is None:
            self.run.log(payload)
        else:
            self.run.log(payload, step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        for key, value in _json_ready(values).items():
            self.run.summary[key] = value

    def finish(self, exit_code: int = 0) -> None:
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        wandb.finish(exit_code=exit_code)
