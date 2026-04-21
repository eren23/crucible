"""Re-exports from ``crucible.core.fingerprint``.

The canonical location for fingerprinting utilities is
``crucible.core.fingerprint``. This module re-exports all public
symbols so that ``from crucible.runner.fingerprint import ...``
continues to resolve.
"""
from crucible.core.fingerprint import (  # noqa: F401  -- re-export
    build_run_manifest,
    code_fingerprint,
    ensure_clean_commit,
    safe_git_branch,
    safe_git_dirty,
    safe_git_sha,
    _discover_files,
)
