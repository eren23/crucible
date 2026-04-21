"""Code fingerprinting and git state tracking.

Provides SHA-256 fingerprints of key source files so that experiment
results can be associated with the exact code that produced them.
Also captures git HEAD and dirty state for audit trails.

This module lives in ``core/`` because it is used by fleet, runner,
and training — all of which need git identity without depending on
each other.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


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
    except (OSError, subprocess.SubprocessError):
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
    except (OSError, subprocess.SubprocessError):
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
    except (OSError, subprocess.SubprocessError):
        return None
    branch = proc.stdout.strip()
    return branch or None


def ensure_clean_commit(
    project_root: Path,
    *,
    auto_commit: bool = False,
) -> str:
    """Ensure working tree is committed.  Returns HEAD SHA.

    When *auto_commit* is True and the tree is dirty, creates a snapshot
    commit.  When False and dirty, raises RuntimeError.
    """
    dirty = safe_git_dirty(project_root)
    if dirty and not auto_commit:
        raise RuntimeError(
            "Working tree has uncommitted changes. Either commit them or "
            "set auto_commit_versions: true in crucible.yaml to auto-snapshot."
        )
    if dirty and auto_commit:
        from crucible.core.log import utc_now_iso

        subprocess.run(
            ["git", "add", "-A"],
            cwd=project_root,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"crucible: auto-snapshot {utc_now_iso()}"],
            cwd=project_root,
            capture_output=True,
            check=True,
        )
    sha = safe_git_sha(project_root)
    if sha is None:
        raise RuntimeError("Not in a git repository — cannot verify code identity.")
    return sha


def code_fingerprint(
    project_root: Path,
    extra_files: tuple[str, ...] | list[str] | None = None,
) -> dict[str, str | dict[str, str]]:
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
      - Any .py files in src/ directory (recursive)
    """
    found: list[str] = []

    # Training scripts in root
    for p in project_root.glob("train*.py"):
        found.append(p.relative_to(project_root).as_posix())

    # Source files recursively in src/
    src_dir = project_root / "src"
    if src_dir.is_dir():
        for p in src_dir.glob("**/*.py"):
            found.append(p.relative_to(project_root).as_posix())

    return sorted(found)


def build_run_manifest(project_root: Path) -> dict[str, Any]:
    """Build a cryptographic identity of the current code + data state.

    The manifest is stored in queue entries and verified at dispatch time
    to guarantee code consistency across experiments.
    """
    fp = code_fingerprint(project_root)

    # Tap versions
    tap_versions: dict[str, str] = {}
    hub_taps = Path.home() / ".crucible-hub" / "taps"
    if hub_taps.is_dir():
        for tap_dir in hub_taps.iterdir():
            if tap_dir.is_dir() and (tap_dir / ".git").exists():
                sha = safe_git_sha(tap_dir)
                if sha:
                    tap_versions[tap_dir.name] = sha

    # Data manifest checksum
    data_checksum: str | None = None
    manifest_path = project_root / "data" / "manifest.json"
    if manifest_path.is_file():
        data_checksum = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()[:16]

    # Data file checksums (HDF5 + other binary data).  Records SHA-256 of
    # each data file so we can detect dataset mutations even if
    # manifest.json stays the same.  Hashes are streamed in 1 MiB chunks
    # to avoid loading multi-GB files into memory.
    data_file_hashes: dict[str, str] = {}
    data_patterns = ("*.h5", "*.hdf5", "*.parquet", "*.jsonl")
    for tap_dir in (hub_taps.iterdir() if hub_taps.is_dir() else []):
        data_subdir = tap_dir / "data"
        if not data_subdir.is_dir():
            continue
        for pat in data_patterns:
            for data_file in data_subdir.glob(pat):
                rel = f"{tap_dir.name}/data/{data_file.name}"
                h = hashlib.sha256()
                try:
                    with data_file.open("rb") as fh:
                        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                            h.update(chunk)
                    data_file_hashes[rel] = h.hexdigest()[:16]
                except OSError:
                    continue

    return {
        "git_sha": safe_git_sha(project_root),
        "git_dirty": safe_git_dirty(project_root),
        "git_branch": safe_git_branch(project_root),
        "code_fingerprint": fp["fingerprint"],
        "code_files": fp["files"],
        "tap_versions": tap_versions,
        "data_manifest_checksum": data_checksum,
        "data_file_hashes": data_file_hashes,
    }
