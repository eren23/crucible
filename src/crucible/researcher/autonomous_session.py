"""Autonomous research loop — persisted session driver (Phase 1.1).

Implements the autonomous research loop as a stateful session that mediates
between Crucible's pure orchestrator-contract primitives (``request_prompt``
/ ``submit_response``) and an external LLM orchestrator. Crucible never
calls an LLM; the orchestrator passes responses back via ``submit``.

State machine per iteration:

    hypothesis → (submit) → reflection → (submit) → next iteration | done

Session state is persisted under
``.crucible/autonomous_sessions/{session_id}.yaml`` (snapshot) plus
``.crucible/autonomous_sessions/{session_id}.jsonl`` (append-only event
log), mirroring the SessionTracer and SearchTree patterns. Sessions
survive process restarts; an orchestrator that crashed between submits
can resume via ``status`` and continue the loop.

The session driver holds its own lease (a separate concern from
``ResearchState.write_lock``, which is a short-lived guard for state
file mutations). State writes are protected via ``state.write_lock`` in
``orchestrator_api.submit_response``; the session driver itself only
mutates session-local files.

Doom-loop detection: each iteration's stage prompt is fingerprinted
(sha256 of system+user). Repeated identical fingerprints across
iterations trip :func:`detect_doom_loop` and the session aborts.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError, ResearcherError, StaleSubmitError
from crucible.core.log import utc_now_iso
from crucible.researcher import orchestrator_api as oa
from crucible.researcher.state import ResearchState

_SESSIONS_DIRNAME = "autonomous_sessions"
_DOOM_LOOP_WINDOW = 5
_DEFAULT_BUDGET_HOURS_FALLBACK = 10.0
_DEFAULT_SESSION_LOCK_TIMEOUT = 30.0
_CREATE_LOCK_FILENAME = ".create.lock"


@contextmanager
def _file_lock(
    lock_path: Path,
    *,
    timeout: float = _DEFAULT_SESSION_LOCK_TIMEOUT,
    poll_interval: float = 0.1,
) -> Iterator[None]:
    """Acquire an exclusive advisory fcntl lock on ``lock_path``.

    POSIX-only; on platforms without ``fcntl`` (Windows), degrades to a
    no-op so callers still work but lose cross-process exclusivity.
    Raises :class:`AutonomousSessionError` on timeout.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise AutonomousSessionError(
                        f"Could not acquire session lock at {lock_path} within "
                        f"{timeout:.1f}s — another Crucible process may be holding it."
                    ) from exc
                time.sleep(poll_interval)
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)


class AutonomousSessionError(ResearcherError):
    """Autonomous research loop session lifecycle / lookup errors."""


class DoomLoopDetected(ResearcherError):
    """The autonomous loop produced N identical prompts in a row.

    Suggests the orchestrator is stuck — same hypothesis fingerprint after
    each iteration means reflection isn't moving beliefs forward. Session
    is marked errored and the orchestrator should re-seed or intervene.
    """


def _sessions_dir(project_root: Path) -> Path:
    return Path(project_root) / ".crucible" / _SESSIONS_DIRNAME


def _session_yaml_path(project_root: Path, session_id: str) -> Path:
    return _sessions_dir(project_root) / f"{session_id}.yaml"


def _session_jsonl_path(project_root: Path, session_id: str) -> Path:
    return _sessions_dir(project_root) / f"{session_id}.jsonl"


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _fingerprint(system: str | None, user: str | None) -> str:
    combined = f"{system or ''}\n\n{user or ''}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


class AutonomousSession:
    """A persisted autonomous research loop session.

    The session owns no LLM client — it brokers between the
    ``request_prompt`` / ``submit_response`` primitives in
    :mod:`crucible.researcher.orchestrator_api` and an external
    orchestrator. Only one session at a time is reachable per project; a
    second ``start`` while a non-terminal session exists returns the
    existing ``session_id`` rather than creating a new one (parallel
    autonomous loops in the same project would race over
    ``research_state.jsonl``).
    """

    SCHEMA_VERSION = 1
    STAGE_HYPOTHESIS = "hypothesis"
    STAGE_REFLECTION = "reflection"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_CANCELED = "canceled"
    STATUS_ERROR = "error"

    def __init__(self, config: ProjectConfig, session_id: str) -> None:
        self.config = config
        self.session_id = session_id
        self.state_data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def yaml_path(self) -> Path:
        return _session_yaml_path(self.config.project_root, self.session_id)

    @property
    def jsonl_path(self) -> Path:
        return _session_jsonl_path(self.config.project_root, self.session_id)

    @property
    def lock_path(self) -> Path:
        """Per-session advisory lock — serializes apply_response / cancel."""
        return _sessions_dir(self.config.project_root) / f"{self.session_id}.lock"

    def load(self) -> "AutonomousSession":
        if not self.yaml_path.exists():
            raise AutonomousSessionError(
                f"Session {self.session_id!r} not found at {self.yaml_path}"
            )
        self.state_data = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}
        return self

    def save(self) -> None:
        self.state_data["updated_at"] = utc_now_iso()
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.yaml_path.with_suffix(self.yaml_path.suffix + ".tmp")
        tmp.write_text(yaml.safe_dump(self.state_data, sort_keys=False), encoding="utf-8")
        os.replace(tmp, self.yaml_path)

    def append_event(self, event_type: str, **extra: Any) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": utc_now_iso(), "event": event_type, **extra}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def find_active(cls, config: ProjectConfig) -> "AutonomousSession | None":
        """Return the most recent non-terminal session for this project, if any."""
        sessions_dir = _sessions_dir(config.project_root)
        if not sessions_dir.exists():
            return None
        candidates: list[tuple[str, str]] = []  # (updated_at, session_id)
        for p in sessions_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except (yaml.YAMLError, OSError):
                continue
            status = data.get("status")
            if status in (cls.STATUS_DONE, cls.STATUS_CANCELED, cls.STATUS_ERROR):
                continue
            candidates.append((data.get("updated_at", ""), p.stem))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        sid = candidates[0][1]
        session = cls(config, sid)
        session.load()
        return session

    @classmethod
    def _create_lock_path(cls, config: ProjectConfig) -> Path:
        return _sessions_dir(config.project_root) / _CREATE_LOCK_FILENAME

    @classmethod
    def create(
        cls,
        config: ProjectConfig,
        *,
        iterations: int,
        tier: str,
        focus_family: str = "",
        budget_usd: float | None = None,
    ) -> "AutonomousSession":
        # Lock around find_active + write to prevent two concurrent
        # action_start calls from each creating distinct sessions.
        with _file_lock(cls._create_lock_path(config)):
            existing = cls.find_active(config)
            if existing is not None:
                # Idempotent: second start while session running returns the same one.
                return existing
            session_id = _new_session_id()
            session = cls(config, session_id)
            now = utc_now_iso()
            session.state_data = {
                "schema_version": cls.SCHEMA_VERSION,
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
                "status": cls.STATUS_RUNNING,
                "iterations_planned": int(iterations),
                "iterations_completed": 0,
                "current_iteration": 0,
                "current_stage": cls.STAGE_HYPOTHESIS,
                "tier": tier,
                "focus_family": focus_family,
                "budget_usd": budget_usd,
                "budget_spent_usd": 0.0,
                "last_state_snapshot": None,
                "last_prompt_fingerprint": None,
                "recent_fingerprints": [],
                "last_error": None,
            }
            session.save()
            session.append_event(
                "started",
                iterations_planned=iterations,
                tier=tier,
                focus_family=focus_family,
                budget_usd=budget_usd,
            )
            return session

    # ------------------------------------------------------------------
    # Driving
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.state_data.get("status") in (
            self.STATUS_DONE,
            self.STATUS_CANCELED,
            self.STATUS_ERROR,
        )

    def build_prompt(self, state: ResearchState) -> dict[str, Any]:
        """Build the prompt for the session's current stage."""
        stage = self.state_data["current_stage"]
        iteration = self.state_data["current_iteration"]
        focus_family = self.state_data.get("focus_family") or ""

        prompt = oa.request_prompt(
            stage=stage,
            config=self.config,
            state=state,
            focus_family=focus_family,
            iteration=iteration,
        )
        fingerprint = _fingerprint(prompt.get("system"), prompt.get("user"))

        # Doom-loop detection: same prompt fingerprint N times in a row.
        recent = list(self.state_data.get("recent_fingerprints") or [])
        recent.append(fingerprint)
        if len(recent) > _DOOM_LOOP_WINDOW:
            recent = recent[-_DOOM_LOOP_WINDOW:]
        if len(recent) >= _DOOM_LOOP_WINDOW and len(set(recent)) == 1:
            self._mark_error(
                f"Doom-loop detected: identical {stage} prompt fingerprint "
                f"{fingerprint} for {_DOOM_LOOP_WINDOW} iterations. "
                f"Belief/reflection updates aren't moving the prompt forward."
            )
            raise DoomLoopDetected(self.state_data["last_error"])

        self.state_data["recent_fingerprints"] = recent
        self.state_data["last_prompt_fingerprint"] = fingerprint
        self.state_data["last_state_snapshot"] = prompt.get("state_snapshot")
        self.save()
        self.append_event(
            "stage_prompted",
            iteration=iteration,
            stage=stage,
            fingerprint=fingerprint,
            state_snapshot=prompt.get("state_snapshot"),
        )
        return {
            "session_id": self.session_id,
            "iteration": iteration,
            "stage": stage,
            "system": prompt.get("system"),
            "user": prompt.get("user"),
            "schema": prompt.get("schema"),
            "state_snapshot": prompt.get("state_snapshot"),
            "session_status": self.state_data["status"],
            "iterations_planned": self.state_data["iterations_planned"],
            "iterations_completed": self.state_data["iterations_completed"],
        }

    def apply_response(
        self,
        state: ResearchState,
        response: Any,
        submitted_snapshot: dict | None,
    ) -> dict[str, Any]:
        """Apply an orchestrator-supplied response, advance stage.

        Holds a per-session advisory lock around the full check-apply-advance
        sequence so concurrent ``submit`` calls (e.g., orchestrator network
        retries) cannot double-advance the state machine. The inner
        ``oa.submit_response`` separately holds ``ResearchState.write_lock``
        for the research-state file; the two locks are nested in this
        consistent order (session → research-state) by every caller.
        """
        with _file_lock(self.lock_path):
            # Reload to see any state written by a peer that held the lock
            # before us.
            self.load()
            return self._apply_response_locked(state, response, submitted_snapshot)

    def _apply_response_locked(
        self,
        state: ResearchState,
        response: Any,
        submitted_snapshot: dict | None,
    ) -> dict[str, Any]:
        if self.is_terminal():
            raise AutonomousSessionError(
                f"Session {self.session_id} is {self.state_data['status']!r} — cannot submit."
            )
        stage = self.state_data["current_stage"]
        iteration = self.state_data["current_iteration"]

        try:
            apply_result = oa.submit_response(
                stage=stage,
                response=response,
                config=self.config,
                state=state,
                iteration=iteration,
                state_snapshot=submitted_snapshot,
            )
        except StaleSubmitError:
            self.append_event(
                "stage_submit_stale",
                iteration=iteration,
                stage=stage,
                submitted_snapshot=submitted_snapshot,
            )
            raise

        self.append_event(
            "stage_submitted",
            iteration=iteration,
            stage=stage,
            applied=apply_result.get("applied", False),
            summary=apply_result.get("summary", ""),
        )

        # Advance state machine.
        if stage == self.STAGE_HYPOTHESIS:
            self.state_data["current_stage"] = self.STAGE_REFLECTION
        elif stage == self.STAGE_REFLECTION:
            self.state_data["iterations_completed"] = iteration + 1
            next_iter = iteration + 1
            if next_iter >= self.state_data["iterations_planned"]:
                self.state_data["status"] = self.STATUS_DONE
                self.state_data["current_stage"] = "done"
                self.append_event("done", iterations_completed=next_iter)
            else:
                self.state_data["current_iteration"] = next_iter
                self.state_data["current_stage"] = self.STAGE_HYPOTHESIS
                # Note: recent_fingerprints is intentionally NOT reset across
                # iteration boundaries. A doom-loop is exactly "the same stage
                # prompt keeps coming back" — which by definition crosses
                # iterations. Reset would mask that signal.
                self.append_event("iteration_complete", iteration=iteration)
        else:
            raise AutonomousSessionError(f"Unknown stage {stage!r} on submit")

        self.save()
        return {
            "session_id": self.session_id,
            "stage_applied": stage,
            "next_stage": self.state_data["current_stage"],
            "iteration": iteration,
            "iterations_completed": self.state_data["iterations_completed"],
            "session_status": self.state_data["status"],
            "apply_result": apply_result,
        }

    def cancel(self, reason: str = "") -> dict[str, Any]:
        with _file_lock(self.lock_path):
            # Reload to see whether a concurrent submit already terminated us.
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

    def _mark_error(self, message: str) -> None:
        self.state_data["status"] = self.STATUS_ERROR
        self.state_data["last_error"] = message
        self.save()
        self.append_event("error", message=message)


# ---------------------------------------------------------------------------
# Action dispatch — used by the MCP tool wrapper and the CLI handler
# ---------------------------------------------------------------------------


def _load_state(config: ProjectConfig) -> ResearchState:
    state_path = config.project_root / config.research_state_file
    budget_hours = getattr(getattr(config, "researcher", None), "budget_hours", _DEFAULT_BUDGET_HOURS_FALLBACK)
    return ResearchState(state_path, budget_hours=budget_hours)


def action_start(
    config: ProjectConfig,
    *,
    iterations: int,
    tier: str = "proxy",
    focus_family: str = "",
    budget_usd: float | None = None,
) -> dict[str, Any]:
    """Start (or resume the active session of) an autonomous research loop."""
    if iterations < 1:
        raise CrucibleError("autonomous_research_loop: iterations must be >= 1")
    session = AutonomousSession.create(
        config,
        iterations=iterations,
        tier=tier,
        focus_family=focus_family,
        budget_usd=budget_usd,
    )
    state = _load_state(config)
    return session.build_prompt(state)


def action_submit(
    config: ProjectConfig,
    *,
    session_id: str,
    response: Any,
    state_snapshot: dict | None = None,
) -> dict[str, Any]:
    """Apply an orchestrator-supplied response and advance the session.

    If ``state_snapshot`` is None, falls back to the session's
    ``last_state_snapshot`` (set by the most recent ``build_prompt``). This
    keeps the Phase 1.2 stale-submit guard active by default even when the
    caller (CLI, simpler MCP client) doesn't track snapshots explicitly.
    Pass an explicit snapshot to override; the session's tracked one is
    the safe default but does not capture races where the orchestrator
    delayed long enough that the session itself didn't move.
    """
    session = AutonomousSession(config, session_id).load()
    if state_snapshot is None:
        state_snapshot = session.state_data.get("last_state_snapshot")
    state = _load_state(config)
    applied = session.apply_response(state, response, state_snapshot)

    if session.state_data["status"] == AutonomousSession.STATUS_DONE:
        return {
            **applied,
            "next_prompt": None,
            "session_status": AutonomousSession.STATUS_DONE,
        }

    # Build the next stage's prompt (reflection after hypothesis, or hypothesis
    # of the next iteration after reflection). DoomLoopDetected propagates to
    # the caller — the session has already been marked errored on disk by
    # build_prompt's internal _mark_error call.
    next_prompt = session.build_prompt(state)
    return {**applied, "next_prompt": next_prompt}


def action_status(config: ProjectConfig, *, session_id: str) -> dict[str, Any]:
    session = AutonomousSession(config, session_id).load()
    data = dict(session.state_data)
    data["yaml_path"] = str(session.yaml_path)
    data["jsonl_path"] = str(session.jsonl_path)
    return data


def action_cancel(
    config: ProjectConfig, *, session_id: str, reason: str = ""
) -> dict[str, Any]:
    session = AutonomousSession(config, session_id).load()
    return session.cancel(reason=reason)


__all__ = [
    "AutonomousSession",
    "AutonomousSessionError",
    "DoomLoopDetected",
    "action_start",
    "action_submit",
    "action_status",
    "action_cancel",
]
