"""Code fingerprinting and git state tracking.

Provides SHA-256 fingerprints of key source files so that experiment
results can be associated with the exact code that produced them.
Also captures git HEAD and dirty state for audit trails.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def safe_git_sha(project_root: Path) -> str | None:
    """Return the current git HEAD SHA, or None if unavailable."""
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
    """Return True if the working tree has uncommitted changes, None on error."""
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


def safe_git_branch(project_root: Path) -> str | None:
    """Return the current git branch name, or None if unavailable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    branch = proc.stdout.strip()
    return branch or None


def code_fingerprint(
    project_root: Path,
    extra_files: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """SHA-256 fingerprint key source files for experiment deduplication.

    Args:
        project_root: Root of the project to scan.
        extra_files: Relative paths to hash.  If None, auto-discovers
            Python files in common locations (train*.py, src/**/*.py).

    Returns:
        Dict with "fingerprint" (combined hash) and "files" (per-file hashes).
    """
    if extra_files is not None:
        files_to_hash = list(extra_files)
    else:
        files_to_hash = _discover_files(project_root)

    file_hashes: dict[str, str] = {}
    for rel in files_to_hash:
        p = project_root / rel
        if p.is_file():
            file_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    combined = hashlib.sha256(
        "|".join(f"{k}={v}" for k, v in sorted(file_hashes.items())).encode()
    ).hexdigest()[:16]

    return {"fingerprint": combined, "files": file_hashes}


def _discover_files(project_root: Path) -> list[str]:
    """Auto-discover key source files to fingerprint.

    Looks for:
      - train*.py in project root
      - Any .py files in src/ directory (first level only)
    """
    found: list[str] = []

    # Training scripts in root
    for p in project_root.glob("train*.py"):
        found.append(p.relative_to(project_root).as_posix())

    # Source files one level deep in src/
    src_dir = project_root / "src"
    if src_dir.is_dir():
        for p in src_dir.glob("*.py"):
            found.append(p.relative_to(project_root).as_posix())

    return sorted(found)
