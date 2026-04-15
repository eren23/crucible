"""File I/O utilities: atomic writes, JSONL operations, YAML helpers."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _json_ready(value: Any) -> Any:
    """Recursively convert a value to JSON-serializable form."""
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
    """Write JSON atomically via temp file + rename."""
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, returning a list of dicts. Returns [] if file missing."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file with file locking."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of records to a JSONL file (overwrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_yaml(path: Path) -> dict | list | None:
    """Read a YAML file, returning its parsed content.

    Returns ``None`` if the file does not exist.  Unlike
    :func:`yaml.safe_load`, this never returns raw strings or scalars —
    callers always get a ``dict``, ``list``, or ``None``.
    """
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, (dict, list)):
        return raw
    return None


def write_yaml(path: Path, data: Any, *, sort_keys: bool = False) -> None:
    """Write *data* as YAML, creating parent directories as needed.

    This is a simple (non-atomic) write suitable for files where
    partial writes are acceptable (e.g. versioned snapshots that are
    also tracked elsewhere).  For ledgers and registries use
    :func:`atomic_write_yaml`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=sort_keys,
        allow_unicode=True,
    )
    path.write_text(text, encoding="utf-8")


def atomic_write_yaml(path: Path, data: Any, *, sort_keys: bool = False) -> None:
    """Write YAML atomically via temp file + rename.

    Mirrors the pattern used by :func:`atomic_write_json` and
    :func:`atomic_write_jsonl`: write to a tempfile in the same
    directory, fsync, then ``os.replace`` into the target path.  This
    guarantees readers never see a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=sort_keys,
        allow_unicode=True,
    )
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def collect_public_attrs(obj: Any) -> dict[str, Any]:
    """Serialize an object's public, non-callable attributes to a dict."""
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
