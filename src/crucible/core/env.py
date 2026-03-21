"""Minimal .env loader with no external dependency."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILES = [".env", ".env.local"]


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_files(
    root: str | Path,
    *,
    filenames: list[str] | None = None,
    override: bool = False,
) -> list[Path]:
    """Load .env files from *root*, setting env vars that aren't already set.

    Returns list of paths that were actually loaded.
    """
    base = Path(root).resolve()
    names = filenames or DEFAULT_ENV_FILES
    loaded: list[Path] = []
    for name in names:
        path = base / name
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
        loaded.append(path)
    return loaded
