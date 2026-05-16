"""Tree-search autonomous loop — persisted session driver (Phase 1.4b).

Mirrors :mod:`crucible.researcher.autonomous_session` but for tree
search. Each iteration selects one expandable node via the tree's
selection policy, hands the orchestrator a ``tree_auto_expand``-style
prompt for that node, then applies the response under
``SearchTree.write_lock`` (Phase 1.4a). Sessions persist under
``.crucible/tree_autonomous_sessions/{session_id}.yaml`` plus a JSONL
event log, surviving process restarts.

State machine (single-stage per iteration):

    select expandable node → build prompt → (submit) → expand under lock
    → select next | done if iterations exhausted or no expandable nodes

Between submits the orchestrator drives the fleet:
``tree_enqueue_pending`` → ``dispatch_experiments`` → ``collect_results``
→ ``tree_sync_results``. The session itself doesn't move pods.

Doom-loop detector: same prompt fingerprint for 5 iterations in a row
marks the session errored, mirroring the autonomous_research_loop guard.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError, ResearcherError, StaleSubmitError
from crucible.core.log import utc_now_iso
from crucible.researcher.search_tree import SearchTree
from crucible.researcher.session_base import (
    BudgetExceeded,
    SessionBase,
    fingerprint_prompt as _fingerprint,
)

_SESSIONS_DIRNAME = "tree_autonomous_sessions"


class TreeAutonomousSessionError(ResearcherError):
    """Tree autonomous session lifecycle / lookup errors."""


class TreeDoomLoopDetected(ResearcherError):
    """The tree loop produced N identical expansion prompts in a row."""


def _tree_dir(config: ProjectConfig, tree_name: str) -> Path:
    return Path(config.project_root) / ".crucible" / "search_trees" / tree_name


class TreeAutonomousSession(SessionBase):
    """Persisted tree-search autonomous loop session.

    Inherits common lifecycle (paths, save/load/append_event, cancel,
    is_terminal, doom-loop, budget refresh) from :class:`SessionBase`.
    """

    SESSIONS_DIRNAME = _SESSIONS_DIRNAME
    NOT_FOUND_EXC = TreeAutonomousSessionError
    LOCK_TIMEOUT_EXC = TreeAutonomousSessionError
    DOOM_LOOP_EXC = TreeDoomLoopDetected

    @classmethod
    def find_active(
        cls, config: ProjectConfig, tree_name: str
    ) -> "TreeAutonomousSession | None":
        """Return the most recent non-terminal session for this tree, if any."""
        for _ts, sid, data in cls._find_active_yamls(config):
            if data.get("tree_name") == tree_name:
                return cls(config, sid).load()
        return None

    @classmethod
    def create(
        cls,
        config: ProjectConfig,
        *,
        tree_name: str,
        iterations: int,
        n_children: int = 3,
        budget_usd: float | None = None,
    ) -> "TreeAutonomousSession":
        """Create a session (or return the existing one for this tree)."""
        with cls._file_lock(cls._create_lock_path(config)):
            existing = cls.find_active(config, tree_name)
            if existing is not None:
                return existing

            # Phase 1.9: judge-separation enforcement at session start.
            # Tree GRPO + auto-expand can ride on LM-as-judge loops;
            # mis-separated panels would let reward and eval collapse to
            # the same model. Fail-fast before pod time.
            panel = getattr(config, "judges", None)
            if panel is not None and panel.is_configured():
                panel.assert_separated()

            session_id = str(uuid.uuid4())
            session = cls(config, session_id)
            now = utc_now_iso()
            session.state_data = {
                "schema_version": cls.SCHEMA_VERSION,
                "session_id": session_id,
                "tree_name": tree_name,
                "created_at": now,
                "updated_at": now,
                "status": cls.STATUS_RUNNING,
                "iterations_planned": int(iterations),
                "iterations_completed": 0,
                "current_iteration": 0,
                "n_children": int(n_children),
                "current_node_id": None,
                "budget_usd": budget_usd,
                "budget_spent_usd": 0.0,
                "last_tree_snapshot": None,
                "last_prompt_fingerprint": None,
                "recent_fingerprints": [],
                "last_error": None,
            }
            session.save()
            session.append_event(
                "started",
                tree_name=tree_name,
                iterations_planned=iterations,
                n_children=n_children,
                budget_usd=budget_usd,
            )
            return session

    # ------------------------------------------------------------------
    # Driving (is_terminal, cancel, _mark_error, _refresh_budget,
    # doom-loop helpers inherited from SessionBase)
    # ------------------------------------------------------------------

    def _select_next_expandable(self, tree: SearchTree) -> str | None:
        """Pick the next expandable node (best-metric among completed-but-
        not-yet-expanded). ``tree.select_next`` returns pending-execution
        nodes which is a different concept — expansion targets completed
        nodes that can spawn children."""
        expandable = tree.get_expandable_nodes()
        if not expandable:
            return None
        # Greedy on metric: best result first. minimize direction handled by
        # comparing with infinity for missing metric.
        minimize = tree.meta.get("metric_direction", "minimize") == "minimize"
        worst = float("inf") if minimize else float("-inf")

        def metric_key(n: dict[str, Any]) -> float:
            m = n.get("result_metric")
            if m is None:
                return worst
            return m if minimize else -m

        expandable.sort(key=metric_key)
        return expandable[0]["node_id"]

    def _build_prompt_for_node(self, tree: SearchTree, node_id: str) -> dict[str, Any]:
        """Delegate to tree_auto_expand's prompt builder."""
        from crucible.mcp.tools import (
            TREE_AUTO_EXPAND_SCHEMA,
            TREE_AUTO_EXPAND_SYSTEM_PROMPT,
            _tree_auto_expand_build_user_prompt,
        )

        n_children = int(self.state_data["n_children"])
        user_prompt = _tree_auto_expand_build_user_prompt(
            tree, node_id, n_children, extra_context=""
        )
        return {
            "system": TREE_AUTO_EXPAND_SYSTEM_PROMPT,
            "user": user_prompt,
            "schema": TREE_AUTO_EXPAND_SCHEMA,
        }

    def build_next_prompt(self) -> dict[str, Any]:
        """Build the next prompt — used by start and post-submit."""
        tree = SearchTree.load(_tree_dir(self.config, self.state_data["tree_name"]))
        node_id = self._select_next_expandable(tree)
        if node_id is None:
            # Nothing expandable yet — could be waiting on results.
            pending = tree.get_pending_nodes()
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
                        f"{len(pending)} pending node(s) await dispatch + collect. "
                        "Run tree_enqueue_pending → dispatch_experiments → "
                        "collect_results → tree_sync_results, then call action='continue' "
                        "to re-check for expandable nodes."
                    ),
                    "pending_node_ids": [n["node_id"] for n in pending],
                }
            # No expandable, no pending — tree is exhausted for this policy.
            self.state_data["status"] = self.STATUS_DONE
            self.state_data["current_node_id"] = None
            self.save()
            self.append_event("done", reason="no_expandable_nodes")
            return {
                "session_id": self.session_id,
                "session_status": self.STATUS_DONE,
                "message": "Tree has no more expandable nodes.",
            }

        prompt = self._build_prompt_for_node(tree, node_id)
        tree_snapshot = tree.snapshot(node_id=node_id)
        fingerprint = _fingerprint(prompt["system"], prompt["user"])

        # Doom-loop detection via shared base class — raises
        # TreeDoomLoopDetected when the recent window is fully identical.
        self._check_doom_loop(fingerprint, stage_label="expansion")

        # Phase 1.8: refresh budget before returning the prompt. Auto-
        # cancels + raises BudgetExceeded if the cap is hit.
        self._refresh_budget_and_maybe_cancel()

        self.state_data["last_tree_snapshot"] = tree_snapshot
        self.state_data["current_node_id"] = node_id
        self.save()
        self.append_event(
            "node_prompted",
            iteration=self.state_data["current_iteration"],
            node_id=node_id,
            fingerprint=fingerprint,
        )
        return {
            "session_id": self.session_id,
            "iteration": self.state_data["current_iteration"],
            "node_id": node_id,
            "system": prompt["system"],
            "user": prompt["user"],
            "schema": prompt["schema"],
            "tree_snapshot": tree_snapshot,
            "session_status": self.state_data["status"],
            "iterations_planned": self.state_data["iterations_planned"],
            "iterations_completed": self.state_data["iterations_completed"],
        }

    def apply_response(self, response: Any, submitted_snapshot: dict | None) -> dict[str, Any]:
        """Apply orchestrator response under per-session + tree locks."""
        with self._file_lock(self.lock_path):
            self.load()
            if self.is_terminal():
                raise TreeAutonomousSessionError(
                    f"Session {self.session_id} is {self.state_data['status']!r} — cannot submit."
                )
            node_id = self.state_data.get("current_node_id")
            if node_id is None:
                raise TreeAutonomousSessionError(
                    f"Session {self.session_id} has no current node — call build_next_prompt first."
                )

            tree = SearchTree.load(_tree_dir(self.config, self.state_data["tree_name"]))
            # Apply under the tree's write_lock so the snapshot check + expand
            # are atomic w.r.t. peer mutations.
            with tree.write_lock():
                if submitted_snapshot is not None:
                    current = tree.snapshot(node_id=node_id)
                    if current != submitted_snapshot:
                        raise StaleSubmitError(
                            f"Tree advanced since prompt was issued. "
                            f"submitted={submitted_snapshot} current={current}. "
                            "Re-call status/build_next_prompt and retry."
                        )

                from crucible.mcp.tools import _tree_auto_expand_parse_response
                children_specs = _tree_auto_expand_parse_response(response)
                for spec in children_specs:
                    spec["generation_method"] = "llm_auto_expand"
                new_ids = tree.expand_node(node_id, children_specs)

            self.state_data["iterations_completed"] += 1
            next_iter = self.state_data["current_iteration"] + 1
            self.state_data["current_node_id"] = None
            self.append_event(
                "node_expanded",
                iteration=self.state_data["current_iteration"],
                node_id=node_id,
                new_node_ids=new_ids,
            )

            # Keep current_iteration synced with iterations_completed even on
            # terminal — confusing otherwise: a 1-iteration session would end
            # with iterations_completed=1 but current_iteration=0 (Codex review).
            self.state_data["current_iteration"] = next_iter
            if next_iter >= self.state_data["iterations_planned"]:
                self.state_data["status"] = self.STATUS_DONE
                self.save()
                self.append_event(
                    "done",
                    iterations_completed=self.state_data["iterations_completed"],
                )
                return {
                    "session_id": self.session_id,
                    "session_status": self.STATUS_DONE,
                    "node_id": node_id,
                    "new_node_ids": new_ids,
                    "next_prompt": None,
                }

            self.save()
            # Budget check fires after a successful expand. Although
            # tree-search submit doesn't dispatch synchronously (the
            # orchestrator drives dispatch between iterations), the
            # wall-clock-since-start cost may have jumped while the
            # orchestrator was deliberating. Mirrors the research-loop
            # post-apply check for defense-in-depth.
            if self.state_data.get("status") == self.STATUS_RUNNING:
                self._refresh_budget_and_maybe_cancel()
            return {
                "session_id": self.session_id,
                "session_status": self.STATUS_RUNNING,
                "node_id": node_id,
                "new_node_ids": new_ids,
                "iteration": self.state_data["current_iteration"],
            }


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def action_start(
    config: ProjectConfig,
    *,
    tree_name: str,
    iterations: int,
    n_children: int = 3,
    budget_usd: float | None = None,
) -> dict[str, Any]:
    """Start (or resume) a tree autonomous loop session."""
    if iterations < 1:
        raise CrucibleError("tree_autonomous_loop: iterations must be >= 1")
    if n_children < 1:
        raise CrucibleError("tree_autonomous_loop: n_children must be >= 1")
    session = TreeAutonomousSession.create(
        config,
        tree_name=tree_name,
        iterations=iterations,
        n_children=n_children,
        budget_usd=budget_usd,
    )
    return session.build_next_prompt()


def action_submit(
    config: ProjectConfig,
    *,
    session_id: str,
    response: Any,
    tree_snapshot: dict | None = None,
) -> dict[str, Any]:
    """Apply orchestrator response, advance, return next prompt or done."""
    session = TreeAutonomousSession(config, session_id).load()
    if tree_snapshot is None:
        tree_snapshot = session.state_data.get("last_tree_snapshot")
    applied = session.apply_response(response, tree_snapshot)

    if session.state_data["status"] == TreeAutonomousSession.STATUS_DONE:
        return applied

    next_prompt = session.build_next_prompt()
    return {**applied, "next_prompt": next_prompt}


def action_status(config: ProjectConfig, *, session_id: str) -> dict[str, Any]:
    session = TreeAutonomousSession(config, session_id).load()
    data = dict(session.state_data)
    data["yaml_path"] = str(session.yaml_path)
    data["jsonl_path"] = str(session.jsonl_path)
    return data


def action_continue(config: ProjectConfig, *, session_id: str) -> dict[str, Any]:
    """Re-trigger build_next_prompt without applying a response.

    Used after the orchestrator drove external fleet ops in response to
    ``next_action='external_dispatch'``: once new results land in the tree
    via ``tree_sync_results``, calling ``continue`` re-checks for expandable
    nodes and returns the next prompt (or the same external_dispatch hint
    if nothing has advanced, or done if exhausted).
    """
    session = TreeAutonomousSession(config, session_id).load()
    if session.is_terminal():
        raise TreeAutonomousSessionError(
            f"Session {session.session_id} is {session.state_data['status']!r} "
            "— continue is only valid for running sessions."
        )
    return session.build_next_prompt()


def action_cancel(
    config: ProjectConfig, *, session_id: str, reason: str = ""
) -> dict[str, Any]:
    session = TreeAutonomousSession(config, session_id).load()
    return session.cancel(reason=reason)


__all__ = [
    "TreeAutonomousSession",
    "TreeAutonomousSessionError",
    "TreeDoomLoopDetected",
    "action_start",
    "action_submit",
    "action_status",
    "action_cancel",
    "action_continue",
]
