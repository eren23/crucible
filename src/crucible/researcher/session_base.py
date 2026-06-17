"""Shared base class for autonomous-loop session drivers (Phase 1 wrap-up).

Three session drivers in :mod:`crucible.researcher` (research / tree / harness)
duplicated ~150 LOC of lifecycle plumbing each: file paths, atomic YAML
save, JSONL event log, find_active, cancel, terminal-state guards,
doom-loop fingerprint windowing, and the Phase 1.8 budget refresh hook.
This module extracts the common surface so adding a fourth session type
later is ~50 LOC of subclass overrides rather than another copy of the
plumbing.

The base class is intentionally **not** an abstract ABC — subclasses
plug in via overriding class attributes (``SESSIONS_DIRNAME``,
``DOOM_LOOP_EXC``, ``LOCK_TIMEOUT_EXC``) plus a few class methods.
This keeps the inheritance shallow and the call sites readable.

Phase 1.8 propagation: the budget guard
(:meth:`_refresh_budget_and_maybe_cancel`) now lives on the base so
all three session types get it for free; the per-subclass wiring
just needs to (a) persist ``budget_usd`` in state_data and (b) call
the refresh at the right hook points (build_prompt + post-apply).
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import yaml

from crucible.core.config import ProjectConfig
from crucible.core.errors import ResearcherError
from crucible.core.file_lock import file_lock as _core_file_lock
from crucible.core.log import log_warn, utc_now_iso

_CREATE_LOCK_FILENAME = ".create.lock"
_DEFAULT_LOCK_TIMEOUT = 30.0
_DEFAULT_DOOM_LOOP_WINDOW = 5


class BudgetExceeded(ResearcherError):
    """Session's wall-clock × declared-pod-rate spend exceeded budget_usd.

    Used by all three session drivers (research / tree / harness). Session
    is automatically marked canceled with the reason captured in
    state_data.last_error.
    """


class BudgetCheckError(ResearcherError):
    """cost_tracker.compute_session_spend returned an unexpected shape.

    Required keys: ``spend_usd``, ``hours_elapsed``, ``hourly_rate``,
    ``active_pods``. If any are missing or non-numeric, this error is
    raised instead of letting a KeyError or TypeError leak out of a
    timer-context budget check. Raised before the session is canceled
    so the operator sees the cost-tracker bug, not a "budget exceeded"
    that's actually a bad spend dict.
    """


_REQUIRED_SPEND_KEYS = ("spend_usd", "hours_elapsed", "hourly_rate", "active_pods")


def _validate_spend_dict(spend: Any) -> None:
    """Raise BudgetCheckError if ``spend`` is not the expected shape.

    Expected: ``{spend_usd: number, hours_elapsed: number,
    hourly_rate: number, active_pods: int}``. Numeric values may be
    int or float; ``active_pods`` may be int or non-negative int.
    """
    if not isinstance(spend, dict):
        raise BudgetCheckError(
            f"compute_session_spend returned {type(spend).__name__}; "
            "expected dict."
        )
    missing = [k for k in _REQUIRED_SPEND_KEYS if k not in spend]
    if missing:
        raise BudgetCheckError(
            f"compute_session_spend missing keys: {missing}. "
            f"Got: {sorted(spend.keys())}."
        )
    for k in ("spend_usd", "hours_elapsed", "hourly_rate"):
        v = spend.get(k)
        if not isinstance(v, (int, float)):
            raise BudgetCheckError(
                f"compute_session_spend['{k}'] is {type(v).__name__}, "
                f"expected number. Value: {v!r}."
            )


def fingerprint_prompt(system: str | None, user: str | None) -> str:
    """SHA-256 of system+user, truncated to 16 hex chars.

    Shared helper used by every session's doom-loop detector. Stable
    across runs; same prompt always produces the same fingerprint.
    """
    combined = f"{system or ''}\n\n{user or ''}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


class SessionBase:
    """Common lifecycle for persisted autonomous-loop sessions.

    Subclass contract:
    - Override ``SESSIONS_DIRNAME`` (e.g. ``"autonomous_sessions"``).
    - Override ``NOT_FOUND_EXC`` and ``LOCK_TIMEOUT_EXC`` with the
      subclass's domain-specific exception types.
    - Override ``DOOM_LOOP_EXC`` if doom-loop detection is wired.
    - Provide ``create()`` classmethod that populates ``state_data``
      and saves before returning.
    - Provide ``find_active()`` classmethod (per-tree filter, per-project,
      etc. — varies by session type).
    """

    SCHEMA_VERSION = 1
    SESSIONS_DIRNAME: str = ""  # subclass must override
    DOOM_LOOP_WINDOW: int = _DEFAULT_DOOM_LOOP_WINDOW

    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_CANCELED = "canceled"
    STATUS_ERROR = "error"
    TERMINAL_STATUSES = (STATUS_DONE, STATUS_CANCELED, STATUS_ERROR)

    # Subclass overrides for domain-specific error types. Defaults are
    # the generic ResearcherError so unconfigured subclasses still
    # raise something sensible.
    NOT_FOUND_EXC: type = ResearcherError
    LOCK_TIMEOUT_EXC: type = ResearcherError
    DOOM_LOOP_EXC: type = ResearcherError

    def __init__(self, config: ProjectConfig, session_id: str) -> None:
        self.config = config
        self.session_id = session_id
        self.state_data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @classmethod
    def _sessions_dir(cls, config: ProjectConfig) -> Path:
        if not cls.SESSIONS_DIRNAME:
            raise NotImplementedError(
                f"{cls.__name__} must define SESSIONS_DIRNAME"
            )
        return Path(config.project_root) / ".crucible" / cls.SESSIONS_DIRNAME

    @classmethod
    def _create_lock_path(cls, config: ProjectConfig) -> Path:
        return cls._sessions_dir(config) / _CREATE_LOCK_FILENAME

    @property
    def yaml_path(self) -> Path:
        return self._sessions_dir(self.config) / f"{self.session_id}.yaml"

    @property
    def jsonl_path(self) -> Path:
        return self._sessions_dir(self.config) / f"{self.session_id}.jsonl"

    @property
    def lock_path(self) -> Path:
        return self._sessions_dir(self.config) / f"{self.session_id}.lock"

    # ------------------------------------------------------------------
    # Lock helper
    # ------------------------------------------------------------------

    @classmethod
    def _file_lock(cls, path: Path, *, timeout: float = _DEFAULT_LOCK_TIMEOUT) -> AbstractContextManager[None]:
        """Wrap :func:`crucible.core.file_lock.file_lock` with the
        subclass's domain-specific timeout exception.

        Classmethod so it's reachable from create() (which is itself a
        classmethod) as well as from instance methods.
        """
        return _core_file_lock(
            path,
            timeout=timeout,
            on_timeout=lambda msg: cls.LOCK_TIMEOUT_EXC(
                msg.replace("file lock", f"{cls.SESSIONS_DIRNAME} lock")
            ),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> SessionBase:
        if not self.yaml_path.exists():
            raise self.NOT_FOUND_EXC(
                f"Session {self.session_id!r} not found at {self.yaml_path}"
            )
        self.state_data = yaml.safe_load(
            self.yaml_path.read_text(encoding="utf-8")
        ) or {}
        return self

    def save(self) -> None:
        self.state_data["updated_at"] = utc_now_iso()
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.yaml_path.with_suffix(self.yaml_path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(self.state_data, sort_keys=False), encoding="utf-8"
        )
        os.replace(tmp, self.yaml_path)

    def append_event(self, event_type: str, **extra: Any) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": utc_now_iso(), "event": event_type, **extra}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Status / cancel
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.state_data.get("status") in self.TERMINAL_STATUSES

    def _mark_error(self, message: str) -> None:
        self.state_data["status"] = self.STATUS_ERROR
        self.state_data["last_error"] = message
        self.save()
        self.append_event("error", message=message)

    def cancel(self, reason: str = "") -> dict[str, Any]:
        with self._file_lock(self.lock_path):
            self.load()
            if self.is_terminal():
                return {
                    "session_id": self.session_id,
                    "session_status": self.state_data["status"],
                    "checkpoint_path": str(self.yaml_path),
                    "already_terminal": True,
                }
            self.state_data["status"] = self.STATUS_CANCELED
            self.state_data["last_error"] = reason or None
            self.save()
            self.append_event("canceled", reason=reason)
            return {
                "session_id": self.session_id,
                "session_status": self.STATUS_CANCELED,
                "checkpoint_path": str(self.yaml_path),
                "already_terminal": False,
            }

    # ------------------------------------------------------------------
    # Doom-loop detection
    # ------------------------------------------------------------------

    def _check_doom_loop(self, fingerprint: str, *, stage_label: str = "stage") -> None:
        """Append fingerprint to the recent window; raise DOOM_LOOP_EXC
        when the last ``DOOM_LOOP_WINDOW`` entries are all identical.

        Mutates ``state_data['recent_fingerprints']`` in place; callers
        should call ``save()`` after.
        """
        recent = list(self.state_data.get("recent_fingerprints") or [])
        recent.append(fingerprint)
        if len(recent) > self.DOOM_LOOP_WINDOW:
            recent = recent[-self.DOOM_LOOP_WINDOW :]
        self.state_data["recent_fingerprints"] = recent
        self.state_data["last_prompt_fingerprint"] = fingerprint
        if len(recent) >= self.DOOM_LOOP_WINDOW and len(set(recent)) == 1:
            msg = (
                f"Doom-loop detected: identical {stage_label} prompt fingerprint "
                f"{fingerprint} for {self.DOOM_LOOP_WINDOW} iterations."
            )
            self._mark_error(msg)
            raise self.DOOM_LOOP_EXC(msg)

    # ------------------------------------------------------------------
    # Phase 1.8 budget guard
    # ------------------------------------------------------------------

    def _refresh_budget_and_maybe_cancel(self) -> None:
        """Recompute spend; if over budget, mark session canceled + raise.

        All three session types (research / tree / harness) share this
        primitive — they just need to (a) persist ``budget_usd`` in
        ``state_data`` at create-time and (b) call this method at the
        right hook points (typically build-prompt start + post-apply).

        Skipped when ``budget_usd`` is None (no cap configured). Caller
        must hold the session lock if the cancellation must be atomic
        vs concurrent reads.
        """
        budget = self.state_data.get("budget_usd")
        if budget is None:
            return
        from crucible.runner.cost_tracker import compute_session_spend

        spend = compute_session_spend(
            self.config, self.state_data.get("created_at", "")
        )
        # Validate the spend dict shape before we read it. A malformed
        # response from compute_session_spend (missing keys, non-numeric
        # values, error sentinel dict) should raise a typed
        # BudgetCheckError so the operator sees a cost-tracker bug
        # instead of a confusing KeyError from inside a timer callback.
        _validate_spend_dict(spend)
        self.state_data["budget_spent_usd"] = spend["spend_usd"]
        self.state_data["last_budget_check"] = spend
        if spend["spend_usd"] >= float(budget):
            reason = (
                f"budget exceeded: spent ${spend['spend_usd']:.2f} "
                f"(${spend['hourly_rate']:.2f}/hr × {spend['hours_elapsed']:.2f}h) "
                f">= cap ${float(budget):.2f}"
            )
            self.state_data["status"] = self.STATUS_CANCELED
            self.state_data["last_error"] = reason
            self.save()
            self.append_event(
                "budget_exceeded",
                budget_usd=float(budget),
                spend_usd=spend["spend_usd"],
                hourly_rate=spend["hourly_rate"],
                hours_elapsed=spend["hours_elapsed"],
            )
            raise BudgetExceeded(reason)

    # ------------------------------------------------------------------
    # find_active helper (subclasses customize the filter)
    # ------------------------------------------------------------------

    @classmethod
    def _find_active_yamls(cls, config: ProjectConfig) -> list[tuple[str, str, dict[str, Any]]]:
        """Return ``[(updated_at, session_id, state_data), ...]`` of all
        non-terminal session YAMLs in this session type's directory.
        Subclasses' ``find_active`` typically filter this list further
        (e.g. by ``tree_name``).
        """
        sessions_dir = cls._sessions_dir(config)
        if not sessions_dir.exists():
            return []
        out: list[tuple[str, str, dict[str, Any]]] = []
        for p in sessions_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except (yaml.YAMLError, OSError) as exc:
                # A corrupted session yaml used to vanish silently from
                # the active-session list, which masked partial writes
                # and made debugging "router doesn't see my session"
                # bugs hard. Log it so the failure is visible.
                log_warn(
                    f"{cls.__name__}._find_active_yamls: skipping {p.name} — "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            status = data.get("status")
            if status in cls.TERMINAL_STATUSES:
                continue
            out.append((data.get("updated_at", ""), p.stem, data))
        out.sort(reverse=True)
        return out


__all__ = [
    "BudgetExceeded",
    "SessionBase",
    "fingerprint_prompt",
]
