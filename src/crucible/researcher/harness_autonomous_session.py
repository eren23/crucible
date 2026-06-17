"""Harness-optimization autonomous loop — persisted session driver (Phase 1.5).

Wraps :class:`crucible.researcher.harness_optimizer.HarnessOptimizer` as
an orchestrator-contract session. Each iteration asks the orchestrator
to propose harness candidates (Python class implementations matching a
domain spec), validates them, benchmarks them via the fleet
(fire-and-forget enqueue), then waits for the orchestrator to drive
fleet dispatch + collect + sync before continuing.

State machine:

    propose → (submit) → benchmark (enqueue) → external_dispatch
    → (continue) → next iteration's propose | done

Like tree_autonomous_session, the session itself never calls an LLM.
Between submits the orchestrator drives the fleet (dispatch + collect
+ tree_sync_results). The session driver only manages lifecycle, locks,
and judge-separation enforcement at start.

Judge-separation: the existing ``HarnessOptimizer.run_iteration``
already calls ``panel.assert_separated()`` when ``config.judges`` is
configured. This session driver mirrors the call at session start so
mis-separated judge panels fail before any pod time is consumed.
"""
from __future__ import annotations

import uuid
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError, ResearcherError
from crucible.core.log import utc_now_iso
from crucible.researcher.session_base import (
    SessionBase,
)
from crucible.researcher.session_base import (
    fingerprint_prompt as _fingerprint,
)

_SESSIONS_DIRNAME = "harness_autonomous_sessions"


class HarnessAutonomousSessionError(ResearcherError):
    """Harness autonomous session lifecycle / lookup errors."""


class HarnessDoomLoopDetected(ResearcherError):
    """The harness loop produced N identical proposal prompts in a row."""


def _make_optimizer(
    config: ProjectConfig,
    *,
    domain_spec: str,
    tree_name: str,
    n_candidates: int,
    dry_run: bool = False,
):
    """Construct a HarnessOptimizer with the supplied params."""
    from crucible.researcher.harness_optimizer import HarnessOptimizer

    return HarnessOptimizer(
        config=config,
        domain_spec=domain_spec,
        tree_name=tree_name,
        n_candidates=n_candidates,
        dry_run=dry_run,
    )


class HarnessAutonomousSession(SessionBase):
    """Persisted harness-optimization autonomous loop session.

    Inherits common lifecycle from :class:`SessionBase`.
    """

    SESSIONS_DIRNAME = _SESSIONS_DIRNAME
    NOT_FOUND_EXC = HarnessAutonomousSessionError
    LOCK_TIMEOUT_EXC = HarnessAutonomousSessionError
    DOOM_LOOP_EXC = HarnessDoomLoopDetected

    STAGE_PROPOSAL = "proposal"
    STAGE_BENCHMARK_WAIT = "benchmark_wait"

    @classmethod
    def find_active(
        cls, config: ProjectConfig, tree_name: str
    ) -> HarnessAutonomousSession | None:
        for _ts, sid, data in cls._find_active_yamls(config):
            if data.get("tree_name") == tree_name:
                return cls(config, sid).load()
        return None

    @classmethod
    def create(
        cls,
        config: ProjectConfig,
        *,
        domain_spec: str,
        tree_name: str,
        iterations: int,
        n_candidates: int = 3,
        dry_run: bool = False,
        budget_usd: float | None = None,
    ) -> HarnessAutonomousSession:
        with cls._file_lock(cls._create_lock_path(config)):
            existing = cls.find_active(config, tree_name)
            if existing is not None:
                return existing

            # Enforce judge separation BEFORE creating the session — mirrors
            # HarnessOptimizer.run_iteration's check (harness_optimizer.py:407-409).
            panel = getattr(config, "judges", None)
            if panel is not None and panel.is_configured():
                panel.assert_separated()

            # Validate the domain spec + tree by constructing the optimizer
            # once. Surfaces config errors before session-id is minted.
            _make_optimizer(
                config,
                domain_spec=domain_spec,
                tree_name=tree_name,
                n_candidates=n_candidates,
                dry_run=dry_run,
            )

            session_id = str(uuid.uuid4())
            session = cls(config, session_id)
            now = utc_now_iso()
            session.state_data = {
                "schema_version": cls.SCHEMA_VERSION,
                "session_id": session_id,
                "tree_name": tree_name,
                "domain_spec": domain_spec,
                "n_candidates": int(n_candidates),
                "dry_run": bool(dry_run),
                "created_at": now,
                "updated_at": now,
                "status": cls.STATUS_RUNNING,
                "stage": cls.STAGE_PROPOSAL,
                "iterations_planned": int(iterations),
                "iterations_completed": 0,
                "current_iteration": 0,
                "budget_usd": budget_usd,
                "budget_spent_usd": 0.0,
                "last_prompt_fingerprint": None,
                "recent_fingerprints": [],
                "last_pending_node_ids": [],
                "last_error": None,
            }
            session.save()
            session.append_event(
                "started",
                tree_name=tree_name,
                domain_spec=domain_spec,
                iterations_planned=iterations,
                n_candidates=n_candidates,
                budget_usd=budget_usd,
            )
            return session

    # ------------------------------------------------------------------
    # Driving (is_terminal, cancel, _mark_error, _refresh_budget,
    # doom-loop helpers inherited from SessionBase)
    # ------------------------------------------------------------------

    def _build_optimizer(self):
        return _make_optimizer(
            self.config,
            domain_spec=self.state_data["domain_spec"],
            tree_name=self.state_data["tree_name"],
            n_candidates=int(self.state_data["n_candidates"]),
            dry_run=bool(self.state_data.get("dry_run", False)),
        )

    def build_proposal_prompt(self) -> dict[str, Any]:
        """Build the next proposal prompt — used by start and continue."""
        if self.is_terminal():
            raise HarnessAutonomousSessionError(
                f"Session {self.session_id} is {self.state_data['status']!r}."
            )
        if self.state_data["stage"] == self.STAGE_BENCHMARK_WAIT:
            # Caller must drive fleet ops before proposing again.
            optimizer = self._build_optimizer()
            pending = optimizer.tree.get_pending_nodes()
            if pending:
                self.append_event(
                    "awaiting_external",
                    pending_count=len(pending),
                )
                return {
                    "session_id": self.session_id,
                    "session_status": self.state_data["status"],
                    "next_action": "external_dispatch",
                    "message": (
                        f"{len(pending)} pending node(s) from the last benchmark "
                        "await dispatch + collect. Run dispatch_experiments → "
                        "collect_results → tree_sync_results, then call "
                        "action='continue' to advance to the next proposal."
                    ),
                    "pending_node_ids": [n["node_id"] for n in pending],
                }
            # Pending drained — advance to next proposal stage.
            self.state_data["stage"] = self.STAGE_PROPOSAL
            self.state_data["last_pending_node_ids"] = []
            self.save()

        # Proposal stage: build prompt via HarnessOptimizer's private builder.
        optimizer = self._build_optimizer()
        system, user = optimizer._build_proposal_prompt(int(self.state_data["n_candidates"]))
        fingerprint = _fingerprint(system, user)

        # Doom-loop detection via shared base class.
        self._check_doom_loop(fingerprint, stage_label="proposal")

        # Phase 1.8: refresh budget before returning the prompt.
        self._refresh_budget_and_maybe_cancel()

        self.save()
        self.append_event(
            "proposal_prompted",
            iteration=self.state_data["current_iteration"],
            fingerprint=fingerprint,
        )
        return {
            "session_id": self.session_id,
            "iteration": self.state_data["current_iteration"],
            "stage": self.STAGE_PROPOSAL,
            "system": system,
            "user": user,
            "session_status": self.state_data["status"],
            "iterations_planned": self.state_data["iterations_planned"],
            "iterations_completed": self.state_data["iterations_completed"],
        }

    def apply_response(self, raw_response: str) -> dict[str, Any]:
        """Parse + validate + benchmark orchestrator-supplied candidates.

        Holds the per-session lock AND ``optimizer.tree.write_lock`` around
        the benchmark step so concurrent processes mutating the same tree
        (e.g., a peer harness session or a tree_auto_expand call) cannot
        lose updates via current_tree.yaml's last-writer-wins rewrite.

        Note on benchmark semantics: ``HarnessOptimizer.benchmark`` does NOT
        just enqueue — it calls ``fleet.enqueue`` then ``fleet.dispatch``
        and falls back to local synchronous execution on fleet error
        (harness_optimizer.py:312, 343). So submit can do real work
        synchronously when running locally. The orchestrator's role
        between submit and continue is to drive collect_results and
        tree_sync_results so pending nodes get their metrics.
        """
        with self._file_lock(self.lock_path):
            self.load()
            if self.is_terminal():
                raise HarnessAutonomousSessionError(
                    f"Session {self.session_id} is {self.state_data['status']!r} — cannot submit."
                )
            if self.state_data["stage"] != self.STAGE_PROPOSAL:
                raise HarnessAutonomousSessionError(
                    f"Session is in stage {self.state_data['stage']!r}; "
                    "submit only valid in 'proposal' stage."
                )

            optimizer = self._build_optimizer()
            try:
                proposed = optimizer._parse_candidates(raw_response or "")
            except Exception as exc:
                raise CrucibleError(
                    f"harness_autonomous_loop submit: could not parse orchestrator "
                    f"response as harness candidates: {exc}"
                ) from exc

            # validate_candidates is read-only on the tree, but the subsequent
            # benchmark mutates: add_root, store_candidate, eventual record_result.
            # Hold tree.write_lock so the full check-then-mutate sequence is
            # atomic w.r.t. peer mutations on the same tree.
            with optimizer.tree.write_lock():
                valid = optimizer.validate_candidates(list(proposed))
                node_ids = optimizer.benchmark(valid) if valid else []

            self.state_data["iterations_completed"] += 1
            next_iter = self.state_data["current_iteration"] + 1
            self.state_data["current_iteration"] = next_iter
            self.state_data["last_pending_node_ids"] = list(node_ids)

            self.append_event(
                "proposal_submitted",
                iteration=next_iter - 1,
                proposed=len(proposed),
                validated=len(valid),
                benchmarked=len(node_ids),
            )

            if next_iter >= self.state_data["iterations_planned"]:
                # Done with iterations — but the last batch's benchmark may
                # still have pending nodes whose results haven't been
                # collected yet. Surface the pending list in the response so
                # the orchestrator knows to run collect_results +
                # tree_sync_results to finalize results.
                self.state_data["status"] = self.STATUS_DONE
                self.state_data["stage"] = "done"
                self.save()
                self.append_event(
                    "done",
                    iterations_completed=self.state_data["iterations_completed"],
                )
                pending = [n["node_id"] for n in optimizer.tree.get_pending_nodes()]
                return {
                    "session_id": self.session_id,
                    "session_status": self.STATUS_DONE,
                    "iteration": next_iter - 1,
                    "proposed": [c.get("name") for c in proposed],
                    "validated": [c.get("name") for c in valid],
                    "benchmarked_node_ids": node_ids,
                    "pending_node_ids": pending,
                    "message": (
                        f"Session done. {len(pending)} node(s) still pending — "
                        "run collect_results + tree_sync_results to finalize metrics."
                    ) if pending else "Session done. No pending nodes.",
                    "next_prompt": None,
                }

            # Move to benchmark_wait stage. The orchestrator drives fleet ops,
            # then calls continue to advance to the next proposal.
            self.state_data["stage"] = self.STAGE_BENCHMARK_WAIT
            self.save()
            # Budget check fires after the synchronous benchmark step. The
            # benchmark call above can dispatch (and locally execute) real
            # pod work, so wall-clock cost may have just jumped. Mirrors the
            # research-loop pattern (autonomous_session.py:_apply_response_locked).
            if self.state_data.get("status") == self.STATUS_RUNNING:
                self._refresh_budget_and_maybe_cancel()
            return {
                "session_id": self.session_id,
                "session_status": self.STATUS_RUNNING,
                "iteration": next_iter - 1,
                "proposed": [c.get("name") for c in proposed],
                "validated": [c.get("name") for c in valid],
                "benchmarked_node_ids": node_ids,
                "next_action": "external_dispatch",
                "message": (
                    f"{len(node_ids)} candidate(s) enqueued/dispatched. Run "
                    "collect_results + tree_sync_results, then call "
                    "action='continue' to advance to the next proposal."
                ),
            }


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def action_start(
    config: ProjectConfig,
    *,
    domain_spec: str,
    tree_name: str,
    iterations: int,
    n_candidates: int = 3,
    dry_run: bool = False,
    budget_usd: float | None = None,
) -> dict[str, Any]:
    if iterations < 1:
        raise CrucibleError("harness_autonomous_loop: iterations must be >= 1")
    if n_candidates < 1:
        raise CrucibleError("harness_autonomous_loop: n_candidates must be >= 1")
    session = HarnessAutonomousSession.create(
        config,
        domain_spec=domain_spec,
        tree_name=tree_name,
        iterations=iterations,
        n_candidates=n_candidates,
        dry_run=dry_run,
        budget_usd=budget_usd,
    )
    return session.build_proposal_prompt()


def action_submit(
    config: ProjectConfig, *, session_id: str, response: str
) -> dict[str, Any]:
    session = HarnessAutonomousSession(config, session_id).load()
    return session.apply_response(response)


def action_continue(config: ProjectConfig, *, session_id: str) -> dict[str, Any]:
    session = HarnessAutonomousSession(config, session_id).load()
    if session.is_terminal():
        raise HarnessAutonomousSessionError(
            f"Session {session.session_id} is {session.state_data['status']!r} "
            "— continue is only valid for running sessions."
        )
    return session.build_proposal_prompt()


def action_status(config: ProjectConfig, *, session_id: str) -> dict[str, Any]:
    session = HarnessAutonomousSession(config, session_id).load()
    data = dict(session.state_data)
    data["yaml_path"] = str(session.yaml_path)
    data["jsonl_path"] = str(session.jsonl_path)
    return data


def action_cancel(
    config: ProjectConfig, *, session_id: str, reason: str = ""
) -> dict[str, Any]:
    session = HarnessAutonomousSession(config, session_id).load()
    return session.cancel(reason=reason)


__all__ = [
    "HarnessAutonomousSession",
    "HarnessAutonomousSessionError",
    "HarnessDoomLoopDetected",
    "action_start",
    "action_submit",
    "action_continue",
    "action_status",
    "action_cancel",
]
