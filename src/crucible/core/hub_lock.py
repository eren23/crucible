"""Cross-process file lock for ``~/.crucible-hub/`` registries.

Two Crucible projects on the same machine share ``~/.crucible-hub/``. Any
read-filter-write sequence on a shared file (``installed.yaml``,
``taps.yaml``, ``hub.yaml``, the architecture registry, finding ledgers)
is racy without serialization: the second writer overwrites the first.

This module provides a single ``hub_lock()`` context manager that acquires
an advisory exclusive lock on ``{hub_dir}/.lock``. All callers that mutate
shared state should wrap their critical section in ``with hub_lock(hub):``.
``append_jsonl`` already uses per-file ``fcntl.flock`` for append-only
ledgers; this lock covers the broader read-filter-write pattern that
``write_jsonl`` cannot make atomic on its own.

The lock is process-wide, advisory, and POSIX-only (it degrades to a no-op
with a one-time warning where ``fcntl`` is unavailable — see
:func:`crucible.core.file_lock.file_lock`). Other tools that touch the hub
directory without acquiring this lock are not protected — that's an
accepted limitation; we control all writers.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from crucible.core.file_lock import file_lock

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_LOCK_FILENAME = ".lock"


class HubLockTimeout(RuntimeError):
    """Raised when ``hub_lock`` cannot acquire within the timeout."""


@contextmanager
def hub_lock(
    hub_dir: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    poll_interval: float = 0.1,
    lock_filename: str = DEFAULT_LOCK_FILENAME,
) -> Iterator[None]:
    """Acquire an exclusive advisory lock on ``{hub_dir}/{lock_filename}``.

    Blocks up to ``timeout`` seconds; raises :class:`HubLockTimeout` on
    contention. The lock is released when the context exits, including on
    exception. POSIX-only (no-op fallback where ``fcntl`` is unavailable).
    """
    lock_path = hub_dir / lock_filename
    with file_lock(
        lock_path,
        timeout=timeout,
        poll_interval=poll_interval,
        on_timeout=lambda _msg: HubLockTimeout(
            f"Could not acquire hub lock at {lock_path} within "
            f"{timeout:.1f}s — another Crucible process may be holding it."
        ),
    ):
        yield
