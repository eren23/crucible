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

import uuid
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError, ResearcherError, StaleSubmitError
from crucible.core.log import utc_now_iso
from crucible.researcher import orchestrator_api as oa
from crucible.researcher.session_base import (
    BudgetExceeded,
    SessionBase,
)
from crucible.researcher.session_base import (
    fingerprint_prompt as _fingerprint,
)
from crucible.researcher.state import ResearchState

_SESSIONS_DIRNAME = "autonomous_sessions"
_DEFAULT_BUDGET_HOURS_FALLBACK = 10.0


class AutonomousSessionError(ResearcherError):
    """Autonomous research loop session lifecycle / lookup errors."""


class DoomLoopDetected(ResearcherError):
    """The autonomous loop produced N identical prompts in a row.

    Suggests the orchestrator is stuck — same hypothesis fingerprint after
    each iteration means reflection isn't moving beliefs forward. Session
    is marked errored and the orchestrator should re-seed or intervene.
    """


def _new_session_id() -> str:
    return str(uuid.uuid4())


class AutonomousSession(SessionBase):
    """A persisted autonomous research loop session.

    The session owns no LLM client — it brokers between the
    ``request_prompt`` / ``submit_response`` primitives in
    :mod:`crucible.researcher.orchestrator_api` and an external
    orchestrator. Only one session at a time is reachable per project; a
    second ``start`` while a non-terminal session exists returns the
    existing ``session_id`` rather than creating a new one (parallel
    autonomous loops in the same project would race over
    ``research_state.jsonl``).

    Inherits common lifecycle (paths, save/load/append_event,
    cancel, is_terminal, doom-loop fingerprint windowing, budget
    refresh) from :class:`SessionBase`.
    """

    SESSIONS_DIRNAME = _SESSIONS_DIRNAME
    NOT_FOUND_EXC = AutonomousSessionError
    LOCK_TIMEOUT_EXC = AutonomousSessionError
    DOOM_LOOP_EXC = DoomLoopDetected

    STAGE_HYPOTHESIS = "hypothesis"
    STAGE_REFLECTION = "reflection"

    @classmethod
    def find_active(cls, config: ProjectConfig) -> AutonomousSession | None:
        """Return the most recent non-terminal session for this project."""
        candidates = cls._find_active_yamls(config)
        if not candidates:
            return None
        _, sid, _ = candidates[0]
        return cls(config, sid).load()

    @classmethod
    def create(
        cls,
        config: ProjectConfig,
        *,
        iterations: int,
        tier: str,
        focus_family: str = "",
        budget_usd: float | None = None,
        with_literature: bool = False,
        literature_k: int = 5,
    ) -> AutonomousSession:
        # Lock around find_active + write to prevent two concurrent
        # action_start calls from each creating distinct sessions.
        with cls._file_lock(cls._create_lock_path(config)):
            existing = cls.find_active(config)
            if existing is not None:
                # Idempotent: second start while session running returns the same one.
                return existing

            # Phase 1.9: judge-separation enforcement at session start so
            # mis-separated judge panels (reward == eval model or family)
            # fail before any pod time is consumed. Mirrors the harness
            # session's identical check at create-time + HarnessOptimizer
            # .run_iteration's per-iteration assert.
            panel = getattr(config, "judges", None)
            if panel is not None and panel.is_configured():
                panel.assert_separated()
            session_id = _new_session_id()
            session = cls(config, session_id)
            now = utc_now_iso()
            # Capture project_name for literature search query fallback (Phase 1.7
            # review fix: the original fallback `focus_family or project_name or
            # "machine learning"` had project_name as a dead key because it was
            # never persisted in state_data).
            project_name = getattr(config, "name", "") or ""
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
                "project_name": project_name,
                "budget_usd": budget_usd,
                "budget_spent_usd": 0.0,
                "with_literature": bool(with_literature),
                "literature_k": int(literature_k),
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
    # Driving (is_terminal, cancel, _mark_error, _refresh_budget_and_maybe_cancel
    # are inherited from SessionBase)
    # ------------------------------------------------------------------

    def build_prompt(self, state: ResearchState) -> dict[str, Any]:
        """Build the prompt for the session's current stage."""
        # Phase 1.8: refresh budget before each prompt build. Auto-cancels
        # + raises BudgetExceeded if the cap is hit.
        self._refresh_budget_and_maybe_cancel()

        stage = self.state_data["current_stage"]
        iteration = self.state_data["current_iteration"]
        focus_family = self.state_data.get("focus_family") or ""

        # Phase 1.7: literature pre-injection (opt-in). When the session
        # was started with with_literature=True, fetch top-K HF Papers via
        # researcher.literature and pass as literature_context. Best-
        # effort: network failures return empty string, never block.
        literature_context = ""
        if self.state_data.get("with_literature") and stage == self.STAGE_HYPOTHESIS:
            try:
                from crucible.researcher import literature as _lit
                query = focus_family or self.state_data.get("project_name") or "machine learning"
                papers = _lit.search_papers(query, limit=int(self.state_data.get("literature_k", 5)))
                literature_context = _lit.format_literature_context(papers, max_papers=int(self.state_data.get("literature_k", 5)))
            except Exception:
                # Best-effort. literature module already swallows network
                # errors; this is a belt-and-suspenders catch-all.
                literature_context = ""

        prompt = oa.request_prompt(
            stage=stage,
            config=self.config,
            state=state,
            focus_family=focus_family,
            iteration=iteration,
            literature_context=literature_context,
        )
        fingerprint = _fingerprint(prompt.get("system"), prompt.get("user"))

        # Doom-loop detection via shared base class. Raises DOOM_LOOP_EXC
        # (DoomLoopDetected for this subclass) when the recent window is
        # fully identical.
        self._check_doom_loop(fingerprint, stage_label=stage)

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
        submitted_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Apply an orchestrator-supplied response, advance stage.

        Holds a per-session advisory lock around the full check-apply-advance
        sequence so concurrent ``submit`` calls (e.g., orchestrator network
        retries) cannot double-advance the state machine. The inner
        ``oa.submit_response`` separately holds ``ResearchState.write_lock``
        for the research-state file; the two locks are nested in this
        consistent order (session → research-state) by every caller.
        """
        with self._file_lock(self.lock_path):
            # Reload to see any state written by a peer that held the lock
            # before us.
            self.load()
            return self._apply_response_locked(state, response, submitted_snapshot)

    def _apply_response_locked(
        self,
        state: ResearchState,
        response: Any,
        submitted_snapshot: dict[str, Any] | None,
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

        # Phase 1.8: post-submit budget check (matches the docstring claim
        # "build AND submit"). Runs while we still hold the session lock so
        # the cancel-on-overrun mutation is atomic vs. concurrent reads.
        # If the session was already advanced to STATUS_DONE above, skip —
        # _refresh would no-op on the is_terminal check next time anyway.
        if self.state_data.get("status") == self.STATUS_RUNNING:
            self._refresh_budget_and_maybe_cancel()

        return {
            "session_id": self.session_id,
            "stage_applied": stage,
            "next_stage": self.state_data["current_stage"],
            "iteration": iteration,
            "iterations_completed": self.state_data["iterations_completed"],
            "session_status": self.state_data["status"],
            "apply_result": apply_result,
        }


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
    with_literature: bool = False,
    literature_k: int = 5,
) -> dict[str, Any]:
    """Start (or resume the active session of) an autonomous research loop.

    Set ``with_literature=True`` to inject HuggingFace Papers as context
    into each hypothesis prompt (best-effort; network failures degrade
    silently). ``literature_k`` controls the number of papers (default 5).
    """
    if iterations < 1:
        raise CrucibleError("autonomous_research_loop: iterations must be >= 1")
    session = AutonomousSession.create(
        config,
        iterations=iterations,
        tier=tier,
        focus_family=focus_family,
        budget_usd=budget_usd,
        with_literature=with_literature,
        literature_k=literature_k,
    )
    state = _load_state(config)
    return session.build_prompt(state)


def action_submit(
    config: ProjectConfig,
    *,
    session_id: str,
    response: Any,
    state_snapshot: dict[str, Any] | None = None,
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
    "BudgetExceeded",
    "DoomLoopDetected",
    "action_start",
    "action_submit",
    "action_status",
    "action_cancel",
]
