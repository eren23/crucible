"""Tree search over experiments.

Manages a persistent search tree where each node represents an experiment
configuration. Supports expansion (adding child experiments), result recording,
pruning, and selection policies (UCB1, greedy, epsilon-greedy, agent-directed).

Storage layout under tree_dir (.crucible/search_trees/{name}/):
    tree.yaml           # SearchTreeMeta
    nodes.jsonl          # Append-only event ledger
    current_tree.yaml    # Full snapshot of all nodes (rebuilt on each mutation)
"""
from __future__ import annotations

import math
import random
import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from crucible.core.errors import SearchTreeError
from crucible.core.io import append_jsonl, read_jsonl, read_yaml
from crucible.core.log import log_warn, utc_now_iso


class SearchTree:
    """A persistent tree over experiment configurations."""

    def __init__(self, tree_dir: Path) -> None:
        self.tree_dir = Path(tree_dir)
        self._meta_path = self.tree_dir / "tree.yaml"
        self._nodes_path = self.tree_dir / "nodes.jsonl"
        self._snapshot_path = self.tree_dir / "current_tree.yaml"
        self.meta: dict[str, Any] = {}
        self.nodes: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        tree_dir: Path,
        name: str,
        description: str = "",
        roots: list[dict[str, Any]] | None = None,
        expansion_policy: str = "agent_directed",
        pruning_policy: str = "agent_directed",
        expansion_config: dict[str, Any] | None = None,
        pruning_config: dict[str, Any] | None = None,
        primary_metric: str = "val_bpb",
        metric_direction: str = "minimize",
        max_depth: int = 10,
        max_nodes: int = 500,
        max_expansions_per_node: int = 5,
        metrics: list[dict[str, str]] | None = None,
        candidate_store_dir: str | None = None,
    ) -> "SearchTree":
        """Create a new search tree on disk.

        ``metrics`` enables multi-metric Pareto-frontier tracking. When unset,
        behavior is single-metric and falls back to ``primary_metric`` /
        ``metric_direction`` for backward compatibility. Each entry must have
        a ``name`` and a ``direction`` of ``"minimize"`` or ``"maximize"``.

        ``candidate_store_dir`` is a path (absolute or relative to
        ``tree_dir``) used by :meth:`store_candidate` to persist candidate
        source code files. When unset, defaults to ``"candidates"`` inside
        ``tree_dir``.
        """
        tree_dir = Path(tree_dir)
        if tree_dir.exists() and (tree_dir / "tree.yaml").exists():
            raise SearchTreeError(f"Search tree already exists at {tree_dir}")

        tree_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()

        # Derive metrics list from (primary_metric, metric_direction) when
        # callers didn't provide a multi-metric list.
        resolved_metrics = metrics if metrics else [
            {"name": primary_metric, "direction": metric_direction}
        ]
        for m in resolved_metrics:
            if "name" not in m or "direction" not in m:
                raise SearchTreeError(
                    f"metrics entry missing name/direction: {m!r}"
                )
            if m["direction"] not in ("minimize", "maximize"):
                raise SearchTreeError(
                    f"metrics direction must be 'minimize' or 'maximize', got {m['direction']!r}"
                )

        tree = cls(tree_dir)
        tree.meta = {
            "name": name,
            "description": description,
            "root_node_ids": [],
            "expansion_policy": expansion_policy,
            "pruning_policy": pruning_policy,
            "expansion_config": expansion_config or {},
            "pruning_config": pruning_config or {},
            "primary_metric": primary_metric,
            "metric_direction": metric_direction,
            "metrics": resolved_metrics,
            "candidate_store_dir": candidate_store_dir or "candidates",
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "max_expansions_per_node": max_expansions_per_node,
            "status": "active",
            "total_nodes": 0,
            "completed_nodes": 0,
            "pruned_nodes": 0,
            "best_node_id": None,
            "best_metric": None,
            "frontier_node_ids": [],
            "created_at": now,
            "updated_at": now,
        }

        tree._save_meta()

        # Add root nodes if provided
        if roots:
            for root in roots:
                tree.add_root(
                    name=root.get("name", f"root-{len(tree.meta['root_node_ids'])}"),
                    config=root.get("config", {}),
                    hypothesis=root.get("hypothesis", ""),
                    rationale=root.get("rationale", ""),
                    tags=root.get("tags", []),
                )

        return tree

    @classmethod
    def load(cls, tree_dir: Path) -> "SearchTree":
        """Load an existing search tree from disk."""
        tree_dir = Path(tree_dir)
        tree = cls(tree_dir)
        tree._load_meta()
        tree._load_nodes()
        return tree

    # ------------------------------------------------------------------
    # Root / expansion
    # ------------------------------------------------------------------

    def add_root(
        self,
        name: str,
        config: dict[str, str],
        hypothesis: str = "",
        rationale: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Add a root node to the tree. Returns node_id."""
        if self.meta["total_nodes"] >= self.meta["max_nodes"]:
            raise SearchTreeError(
                f"Tree '{self.meta['name']}' has reached max_nodes ({self.meta['max_nodes']})"
            )

        node_id = self._make_node_id()
        now = utc_now_iso()

        node: dict[str, Any] = {
            "node_id": node_id,
            "tree_name": self.meta["name"],
            "parent_node_id": None,
            "children": [],
            "depth": 0,
            "experiment_name": name,
            "run_id": None,
            "config": dict(config),
            "status": "pending",
            "result_metric": None,
            "result": None,
            "hypothesis": hypothesis,
            "rationale": rationale,
            "generation_method": "manual",
            "priority_score": 1.0,
            "visit_count": 0,
            "created_at": now,
            "completed_at": None,
            "pruned_at": None,
            "prune_reason": None,
            "tags": tags or [],
        }

        self.nodes[node_id] = node
        self.meta["root_node_ids"].append(node_id)
        self.meta["total_nodes"] += 1
        self.meta["updated_at"] = now

        self._append_node_event("add_root", node)
        self._save_meta()
        self._save_snapshot()

        return node_id

    def expand_node(
        self, parent_id: str, children: list[dict[str, Any]]
    ) -> list[str]:
        """Add children to a node. Merges parent config with child overrides.

        Each child dict should have: name, config (overrides), hypothesis, rationale.
        Optional: tags, generation_method, priority_score.
        Returns list of new node IDs.
        """
        parent = self.get_node(parent_id)
        if parent is None:
            raise SearchTreeError(f"Parent node '{parent_id}' not found")

        if parent["status"] == "pruned":
            raise SearchTreeError(f"Cannot expand pruned node '{parent_id}'")

        current_children_count = len(parent["children"])
        max_per_node = self.meta["max_expansions_per_node"]
        if current_children_count + len(children) > max_per_node:
            raise SearchTreeError(
                f"Node '{parent_id}' already has {current_children_count} children; "
                f"adding {len(children)} would exceed max_expansions_per_node ({max_per_node})"
            )

        remaining_capacity = self.meta["max_nodes"] - self.meta["total_nodes"]
        if len(children) > remaining_capacity:
            raise SearchTreeError(
                f"Adding {len(children)} children would exceed max_nodes "
                f"({self.meta['max_nodes']}); only {remaining_capacity} slots remaining"
            )

        child_depth = parent["depth"] + 1
        if child_depth > self.meta["max_depth"]:
            raise SearchTreeError(
                f"Child depth {child_depth} exceeds max_depth ({self.meta['max_depth']})"
            )

        now = utc_now_iso()
        new_ids: list[str] = []

        for child_spec in children:
            node_id = self._make_node_id()
            # Merge parent config with child overrides
            merged_config = dict(parent["config"])
            merged_config.update(child_spec.get("config", {}))
            # Set PARENT_RUN_ID if parent has a run_id
            if parent.get("run_id"):
                merged_config["PARENT_RUN_ID"] = parent["run_id"]

            node: dict[str, Any] = {
                "node_id": node_id,
                "tree_name": self.meta["name"],
                "parent_node_id": parent_id,
                "children": [],
                "depth": child_depth,
                "experiment_name": child_spec.get("name", f"expand-{node_id[:8]}"),
                "run_id": None,
                "config": merged_config,
                "status": "pending",
                "result_metric": None,
                "result": None,
                "hypothesis": child_spec.get("hypothesis", ""),
                "rationale": child_spec.get("rationale", ""),
                "generation_method": child_spec.get("generation_method", "manual"),
                "priority_score": child_spec.get("priority_score", 1.0),
                "visit_count": 0,
                "created_at": now,
                "completed_at": None,
                "pruned_at": None,
                "prune_reason": None,
                "tags": child_spec.get("tags", []),
                "group_advantage": child_spec.get("group_advantage"),
            }

            self.nodes[node_id] = node
            parent["children"].append(node_id)
            self.meta["total_nodes"] += 1
            new_ids.append(node_id)

            self._append_node_event("expand", node)

        self.meta["updated_at"] = now
        self._save_meta()
        self._save_snapshot()

        return new_ids

    # ------------------------------------------------------------------
    # Result recording
    # ------------------------------------------------------------------

    def record_result(self, node_id: str, result: dict[str, Any]) -> None:
        """Record a completed experiment result for a node.

        Updates status, backpropagates visit counts, and updates best
        tracking (single-metric) and the Pareto frontier (multi-metric).
        The result dict should contain all metric keys listed in tree meta.
        """
        node = self.get_node(node_id)
        if node is None:
            raise SearchTreeError(f"Node '{node_id}' not found")

        if node["status"] == "pruned":
            raise SearchTreeError(f"Cannot record result for pruned node '{node_id}'")

        now = utc_now_iso()
        metric_key = self.meta["primary_metric"]
        metric_val = result.get(metric_key)

        # Collect all tracked metric values (multi-metric Pareto support).
        metric_values: dict[str, float] = {}
        for m in self._get_metrics():
            v = result.get(m["name"])
            if v is not None:
                try:
                    metric_values[m["name"]] = float(v)
                except (TypeError, ValueError):
                    # Non-numeric metric values are skipped silently so the
                    # tree stays usable even when a candidate reports garbage.
                    continue

        node["status"] = "completed"
        node["result"] = dict(result)
        node["result_metric"] = float(metric_val) if metric_val is not None else None
        node["result_metrics"] = metric_values
        node["completed_at"] = now
        if result.get("run_id"):
            node["run_id"] = result["run_id"]

        self.meta["completed_nodes"] += 1
        self.meta["updated_at"] = now

        # Update best tracking (primary metric)
        if metric_val is not None:
            self._update_best(node_id, float(metric_val))

        # Recompute Pareto frontier across all tracked metrics
        self._update_frontier()

        # Backpropagate visit counts
        self._propagate_visits(node_id)

        # Check auto-prune if configured
        self._auto_prune_check(node_id)

        self._append_node_event("record_result", node)
        self._save_meta()
        self._save_snapshot()

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_node(self, node_id: str, reason: str = "") -> None:
        """Prune a single node."""
        node = self.get_node(node_id)
        if node is None:
            raise SearchTreeError(f"Node '{node_id}' not found")

        if node["status"] == "pruned":
            return  # already pruned, idempotent

        now = utc_now_iso()
        node["status"] = "pruned"
        node["pruned_at"] = now
        node["prune_reason"] = reason
        self.meta["pruned_nodes"] += 1
        self.meta["updated_at"] = now

        self._append_node_event("prune", node)
        self._save_meta()
        self._save_snapshot()

    def prune_branch(self, node_id: str, reason: str = "") -> int:
        """Recursively prune a node and all its descendants. Returns count pruned."""
        node = self.get_node(node_id)
        if node is None:
            raise SearchTreeError(f"Node '{node_id}' not found")

        count = 0
        to_prune = [node_id]
        now = utc_now_iso()

        while to_prune:
            nid = to_prune.pop()
            n = self.nodes.get(nid)
            if n is None or n["status"] == "pruned":
                continue
            n["status"] = "pruned"
            n["pruned_at"] = now
            n["prune_reason"] = reason
            self.meta["pruned_nodes"] += 1
            count += 1
            self._append_node_event("prune_branch", n)
            to_prune.extend(n.get("children", []))

        self.meta["updated_at"] = now
        self._save_meta()
        self._save_snapshot()

        return count

    # ------------------------------------------------------------------
    # Selection policies
    # ------------------------------------------------------------------

    def select_next(self, n: int = 1) -> list[str]:
        """Select the next n nodes to run based on the expansion policy.

        Policies:
        - ucb1: UCB1 bandit score
        - greedy: best metric among expandable
        - epsilon_greedy: epsilon random, else best
        - agent_directed: returns empty (agent calls expand_node directly)
        """
        policy = self.meta["expansion_policy"]

        if policy == "agent_directed":
            return []

        pending = self.get_pending_nodes()
        if not pending:
            return []

        if policy == "ucb1":
            return self._select_ucb1(pending, n)
        elif policy == "greedy":
            return self._select_greedy(pending, n)
        elif policy == "epsilon_greedy":
            return self._select_epsilon_greedy(pending, n)
        elif policy == "pareto":
            return self._select_pareto(pending, n)
        else:
            # Unknown policy falls back to priority score
            pending.sort(key=lambda nd: nd.get("priority_score", 0), reverse=True)
            return [nd["node_id"] for nd in pending[:n]]

    def _select_ucb1(self, candidates: list[dict], n: int) -> list[str]:
        """UCB1: score = mean_metric + C * sqrt(ln(total_visits) / visits)."""
        ec = self.meta.get("expansion_config", {})
        c_param = float(ec.get("ucb_c", 1.41))
        minimize = self.meta["metric_direction"] == "minimize"

        total_visits = sum(nd.get("visit_count", 0) for nd in self.nodes.values())
        total_visits = max(total_visits, 1)

        scored: list[tuple[float, str]] = []
        for nd in candidates:
            visits = nd.get("visit_count", 0)
            priority = nd.get("priority_score", 1.0)

            if visits == 0:
                # Unvisited nodes get high exploration bonus
                score = float("inf")
            else:
                # For pending nodes use parent's metric or priority
                parent_metric = self._get_parent_metric(nd)
                if parent_metric is not None:
                    mean = -parent_metric if minimize else parent_metric
                else:
                    mean = priority
                exploration = c_param * math.sqrt(math.log(total_visits) / visits)
                score = mean + exploration

            scored.append((score, nd["node_id"]))

        scored.sort(reverse=True)
        return [nid for _, nid in scored[:n]]

    def _select_greedy(self, candidates: list[dict], n: int) -> list[str]:
        """Select by best parent metric or priority score."""
        minimize = self.meta["metric_direction"] == "minimize"

        def sort_key(nd: dict) -> float:
            parent_metric = self._get_parent_metric(nd)
            if parent_metric is not None:
                return -parent_metric if minimize else parent_metric
            return nd.get("priority_score", 0)

        candidates.sort(key=sort_key, reverse=True)
        return [nd["node_id"] for nd in candidates[:n]]

    def _select_epsilon_greedy(self, candidates: list[dict], n: int) -> list[str]:
        """Epsilon-greedy: epsilon fraction random, else greedy."""
        ec = self.meta.get("expansion_config", {})
        epsilon = float(ec.get("epsilon", 0.1))

        selected: list[str] = []
        remaining = list(candidates)

        for _ in range(min(n, len(remaining))):
            if not remaining:
                break
            if random.random() < epsilon:
                choice = random.choice(remaining)
            else:
                # Greedy pick
                minimize = self.meta["metric_direction"] == "minimize"

                def sort_key(nd: dict) -> float:
                    parent_metric = self._get_parent_metric(nd)
                    if parent_metric is not None:
                        return -parent_metric if minimize else parent_metric
                    return nd.get("priority_score", 0)

                remaining.sort(key=sort_key, reverse=True)
                choice = remaining[0]

            selected.append(choice["node_id"])
            remaining.remove(choice)

        return selected

    def _get_parent_metric(self, node: dict) -> float | None:
        """Get the result metric of a node's parent, if available."""
        parent_id = node.get("parent_node_id")
        if parent_id is None:
            return None
        parent = self.nodes.get(parent_id)
        if parent is None:
            return None
        return parent.get("result_metric")

    def _select_pareto(self, candidates: list[dict], n: int) -> list[str]:
        """Prefer pending children whose completed parent is on the Pareto frontier.

        Unvisited candidates with frontier ancestry bubble to the top. Ties
        are broken by priority_score, then by visit_count (lowest first).
        """
        frontier_ids = set(self.meta.get("frontier_node_ids") or [])
        scored: list[tuple[int, float, int, str]] = []
        for nd in candidates:
            parent_id = nd.get("parent_node_id")
            # Higher primary key when parent sits on the Pareto frontier.
            on_frontier = 1 if parent_id in frontier_ids else 0
            priority = float(nd.get("priority_score", 0.0))
            visits = int(nd.get("visit_count", 0))
            scored.append((on_frontier, priority, -visits, nd["node_id"]))
        # Sort by (on_frontier desc, priority desc, -visits desc -> visits asc)
        scored.sort(reverse=True)
        return [nid for _, _, _, nid in scored[:n]]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get a single node by ID."""
        return self.nodes.get(node_id)

    def get_expandable_nodes(self) -> list[dict[str, Any]]:
        """Get completed, non-pruned nodes with room for more children."""
        max_per = self.meta["max_expansions_per_node"]
        result = []
        for node in self.nodes.values():
            if node["status"] == "completed" and len(node["children"]) < max_per:
                result.append(node)
        return result

    def get_pending_nodes(self) -> list[dict[str, Any]]:
        """Get all nodes with status 'pending'."""
        return [n for n in self.nodes.values() if n["status"] == "pending"]

    def get_frontier(self) -> list[dict[str, Any]]:
        """Get leaf nodes (no children) that are not pruned."""
        return [
            n for n in self.nodes.values()
            if not n["children"] and n["status"] != "pruned"
        ]

    def get_best_path(self) -> list[dict[str, Any]]:
        """Get the path from root to best node."""
        best_id = self.meta.get("best_node_id")
        if best_id is None:
            return []
        return self.get_ancestry(best_id)

    def get_ancestry(self, node_id: str) -> list[dict[str, Any]]:
        """Get the path from root to this node (inclusive, root first)."""
        path: list[dict[str, Any]] = []
        current_id: str | None = node_id
        visited: set[str] = set()

        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            node = self.nodes.get(current_id)
            if node is None:
                break
            path.append(node)
            current_id = node.get("parent_node_id")

        path.reverse()
        return path

    def get_siblings(self, node_id: str) -> list[dict[str, Any]]:
        """Get sibling nodes (same parent, excluding self)."""
        node = self.get_node(node_id)
        if node is None:
            return []
        parent_id = node.get("parent_node_id")
        if parent_id is None:
            # Root node: siblings are other roots
            return [
                self.nodes[rid]
                for rid in self.meta["root_node_ids"]
                if rid != node_id and rid in self.nodes
            ]
        parent = self.nodes.get(parent_id)
        if parent is None:
            return []
        return [
            self.nodes[cid]
            for cid in parent["children"]
            if cid != node_id and cid in self.nodes
        ]

    # ------------------------------------------------------------------
    # Multi-metric Pareto frontier
    # ------------------------------------------------------------------

    def _get_metrics(self) -> list[dict[str, str]]:
        """Return the list of tracked metrics.

        Trees created before multi-metric support store only
        ``primary_metric`` / ``metric_direction``; reconstruct the
        ``metrics`` list from those fields when absent.
        """
        metrics = self.meta.get("metrics")
        if metrics:
            return list(metrics)
        return [
            {
                "name": self.meta.get("primary_metric", "val_bpb"),
                "direction": self.meta.get("metric_direction", "minimize"),
            }
        ]

    def pareto_nodes(self) -> list[str]:
        """Return node IDs on the current N-dimensional Pareto frontier.

        Only completed, non-pruned nodes with values for every tracked metric
        are considered. Returns an empty list when no tree results exist or
        when no node carries a complete metric vector.
        """
        from crucible.analysis.leaderboard import pareto_frontier_nd

        metrics = self._get_metrics()
        directions = [m["direction"] for m in metrics]
        names = [m["name"] for m in metrics]

        points: list[list[float]] = []
        node_ids: list[str] = []
        for nid, node in self.nodes.items():
            if node.get("status") != "completed":
                continue
            values = node.get("result_metrics") or {}
            # Single-metric nodes: populate vector from result_metric.
            if not values and node.get("result_metric") is not None and len(names) == 1:
                values = {names[0]: node["result_metric"]}
            if any(n not in values for n in names):
                continue
            points.append([float(values[n]) for n in names])
            node_ids.append(nid)

        if not points:
            return []

        frontier_idx = pareto_frontier_nd(points, directions, ids=node_ids)
        return [node_ids[i] for i in frontier_idx]

    def frontier_summary(self) -> dict[str, Any]:
        """Return a snapshot of the Pareto frontier (useful for iteration logs).

        Contains:
            - ``frontier_node_ids``: IDs on the current frontier
            - ``frontier_size``: count of frontier nodes
            - ``dominated_count``: completed nodes dominated by someone
            - ``metrics``: [{name, direction}] in use
            - ``best_per_metric``: {metric: {node_id, value}} winner per axis
            - ``hypervolume``: 2D HV when exactly two metrics are tracked
        """
        from crucible.analysis.leaderboard import hypervolume_2d

        metrics = self._get_metrics()
        names = [m["name"] for m in metrics]
        directions = [m["direction"] for m in metrics]

        frontier_ids = self.pareto_nodes()

        completed = [
            n for n in self.nodes.values() if n.get("status") == "completed"
        ]
        dominated_count = max(0, len(completed) - len(frontier_ids))

        best_per_metric: dict[str, Any] = {}
        for name, direction in zip(names, directions):
            best_node_id = None
            best_val: float | None = None
            for node in completed:
                values = node.get("result_metrics") or {}
                if name not in values and node.get("result_metric") is not None and len(names) == 1:
                    # Single-metric tree: use result_metric as the sole value.
                    v = node["result_metric"]
                else:
                    v = values.get(name)
                if v is None:
                    continue
                try:
                    vf = float(v)
                except (TypeError, ValueError):
                    continue
                if best_val is None:
                    best_val = vf
                    best_node_id = node["node_id"]
                elif direction == "minimize" and vf < best_val:
                    best_val = vf
                    best_node_id = node["node_id"]
                elif direction == "maximize" and vf > best_val:
                    best_val = vf
                    best_node_id = node["node_id"]
            if best_node_id is not None:
                best_per_metric[name] = {"node_id": best_node_id, "value": best_val}

        hv: float | None = None
        if len(names) == 2 and frontier_ids:
            pts: list[list[float]] = []
            for nid in frontier_ids:
                node = self.nodes.get(nid)
                if not node:
                    continue
                values = node.get("result_metrics") or {}
                pts.append([float(values[n]) for n in names])
            if pts:
                hv = hypervolume_2d(pts, directions)

        return {
            "frontier_node_ids": frontier_ids,
            "frontier_size": len(frontier_ids),
            "dominated_count": dominated_count,
            "metrics": metrics,
            "best_per_metric": best_per_metric,
            "hypervolume": hv,
        }

    def _update_frontier(self) -> None:
        """Recompute the cached frontier_node_ids after a result is recorded.

        Narrow catch: only structural errors from the frontier computation
        (bad metric values, inconsistent shapes) are tolerated. Anything
        else — including OSError during snapshot writes — should propagate.
        """
        try:
            self.meta["frontier_node_ids"] = self.pareto_nodes()
        except (ValueError, TypeError) as exc:
            # Metric values can be missing or non-numeric for in-flight nodes.
            # Logging here would be noisy — the frontier will refresh on the
            # next record_result() once data is clean.
            log_warn(f"Frontier recompute skipped: {exc}")

    # ------------------------------------------------------------------
    # Code-as-candidate storage
    # ------------------------------------------------------------------

    def _candidate_dir(self) -> Path:
        raw = self.meta.get("candidate_store_dir") or "candidates"
        p = Path(raw)
        if not p.is_absolute():
            p = self.tree_dir / p
        return p

    _SAFE_CANDIDATE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    _MAX_CANDIDATE_SIZE_BYTES = 256 * 1024  # 256 KB per-candidate source cap

    def store_candidate(self, node_id: str, code: str) -> Path:
        """Persist candidate source code for a node.

        The file is written to ``{candidate_store_dir}/{node_id}.py``. The
        tree node's config is augmented with ``HARNESS_CANDIDATE_ID`` and
        ``HARNESS_CANDIDATES_DIR`` so the runner can locate the code at
        dispatch time (parallel to how ``MODEL_FAMILY`` points to
        architecture code on disk).

        Hardens against filesystem misuse: validates the node_id against a
        safe character set, rejects symlink-escaped destinations, and caps
        source size.
        """
        node = self.get_node(node_id)
        if node is None:
            raise SearchTreeError(f"Node '{node_id}' not found")
        if not self._SAFE_CANDIDATE_ID_RE.match(node_id):
            raise SearchTreeError(
                f"Unsafe candidate id {node_id!r}; must match [a-zA-Z0-9][a-zA-Z0-9_-]*"
            )
        if len(code.encode("utf-8")) > self._MAX_CANDIDATE_SIZE_BYTES:
            raise SearchTreeError(
                f"Candidate source exceeds {self._MAX_CANDIDATE_SIZE_BYTES} bytes"
            )

        cdir = self._candidate_dir().resolve()
        cdir.mkdir(parents=True, exist_ok=True)
        path = (cdir / f"{node_id}.py").resolve()
        try:
            path.relative_to(cdir)
        except ValueError as exc:
            raise SearchTreeError(
                f"Candidate path {path} escapes candidate store {cdir}"
            ) from exc

        path.write_text(code, encoding="utf-8")
        cfg = node.setdefault("config", {})
        cfg["HARNESS_CANDIDATE_ID"] = node_id
        cfg["HARNESS_CANDIDATES_DIR"] = str(cdir)
        self._append_node_event("store_candidate", node)
        self._save_snapshot()
        return path

    def load_candidate(self, node_id: str) -> str:
        """Read back candidate source code for a node. Raises if missing."""
        if not self._SAFE_CANDIDATE_ID_RE.match(node_id):
            raise SearchTreeError(
                f"Unsafe candidate id {node_id!r}; must match [a-zA-Z0-9][a-zA-Z0-9_-]*"
            )
        cdir = self._candidate_dir().resolve()
        path = (cdir / f"{node_id}.py").resolve()
        try:
            path.relative_to(cdir)
        except ValueError as exc:
            raise SearchTreeError(
                f"Candidate path {path} escapes candidate store {cdir}"
            ) from exc
        if not path.exists():
            raise SearchTreeError(f"No candidate code at {path}")
        return path.read_text(encoding="utf-8")

    def get_tree_summary(self) -> dict[str, Any]:
        """Get a summary of the tree state."""
        status_counts: dict[str, int] = {}
        depth_counts: dict[int, int] = {}
        for node in self.nodes.values():
            status = node.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            depth = node.get("depth", 0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        return {
            "name": self.meta.get("name"),
            "description": self.meta.get("description", ""),
            "status": self.meta.get("status"),
            "expansion_policy": self.meta.get("expansion_policy"),
            "pruning_policy": self.meta.get("pruning_policy"),
            "primary_metric": self.meta.get("primary_metric"),
            "metric_direction": self.meta.get("metric_direction"),
            "total_nodes": self.meta.get("total_nodes", 0),
            "completed_nodes": self.meta.get("completed_nodes", 0),
            "pruned_nodes": self.meta.get("pruned_nodes", 0),
            "best_node_id": self.meta.get("best_node_id"),
            "best_metric": self.meta.get("best_metric"),
            "metrics": self._get_metrics(),
            "frontier_size": len(self.meta.get("frontier_node_ids") or []),
            "max_depth": self.meta.get("max_depth"),
            "max_nodes": self.meta.get("max_nodes"),
            "status_breakdown": status_counts,
            "depth_breakdown": depth_counts,
            "root_count": len(self.meta.get("root_node_ids", [])),
            "created_at": self.meta.get("created_at"),
            "updated_at": self.meta.get("updated_at"),
        }

    # ------------------------------------------------------------------
    # ASCII rendering
    # ------------------------------------------------------------------

    def render_ascii(self, max_depth: int | None = None) -> str:
        """Render the tree as an ASCII diagram."""
        lines: list[str] = []
        root_ids = self.meta.get("root_node_ids", [])

        for i, root_id in enumerate(root_ids):
            is_last_root = i == len(root_ids) - 1
            self._render_node(root_id, "", is_last_root, lines, max_depth)

        return "\n".join(lines) if lines else "(empty tree)"

    def _render_node(
        self,
        node_id: str,
        prefix: str,
        is_last: bool,
        lines: list[str],
        max_depth: int | None,
    ) -> None:
        """Recursively render a node and its children."""
        node = self.nodes.get(node_id)
        if node is None:
            return

        if max_depth is not None and node["depth"] > max_depth:
            return

        connector = "└── " if is_last else "├── "
        status_icon = {
            "pending": "○",
            "queued": "◎",
            "running": "●",
            "completed": "✓",
            "failed": "✗",
            "pruned": "✂",
        }.get(node["status"], "?")

        metric_str = ""
        if node.get("result_metric") is not None:
            metric_str = f" [{self.meta['primary_metric']}={node['result_metric']:.4f}]"

        label = f"{status_icon} {node['experiment_name']}{metric_str}"

        if prefix == "" and node["depth"] == 0:
            lines.append(label)
        else:
            lines.append(f"{prefix}{connector}{label}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        children = node.get("children", [])
        for j, child_id in enumerate(children):
            is_last_child = j == len(children) - 1
            self._render_node(child_id, child_prefix, is_last_child, lines, max_depth)

    # ------------------------------------------------------------------
    # Persistence (private)
    # ------------------------------------------------------------------

    def _save_meta(self) -> None:
        """Write tree metadata to tree.yaml."""
        self.tree_dir.mkdir(parents=True, exist_ok=True)
        text = yaml.dump(
            self.meta, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
        self._meta_path.write_text(text, encoding="utf-8")

    def _save_snapshot(self) -> None:
        """Write a full snapshot of all nodes to current_tree.yaml."""
        snapshot = {
            "meta": self.meta,
            "nodes": {nid: dict(n) for nid, n in self.nodes.items()},
        }
        text = yaml.dump(
            snapshot, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
        self._snapshot_path.write_text(text, encoding="utf-8")

    def _append_node_event(self, event_type: str, node: dict[str, Any]) -> None:
        """Append an event to the JSONL ledger."""
        event = {
            "ts": utc_now_iso(),
            "event": event_type,
            "node_id": node["node_id"],
            "data": dict(node),
        }
        append_jsonl(self._nodes_path, event)

    def _load_meta(self) -> None:
        """Load tree metadata from tree.yaml."""
        if not self._meta_path.exists():
            raise SearchTreeError(f"No search tree found at {self.tree_dir}")
        raw = read_yaml(self._meta_path)
        if not isinstance(raw, dict):
            raise SearchTreeError(f"Invalid tree metadata in {self._meta_path}")
        self.meta = raw

    def _load_nodes(self) -> None:
        """Load nodes from the snapshot file, falling back to the JSONL ledger."""
        self.nodes.clear()

        # Try snapshot first (faster)
        raw = read_yaml(self._snapshot_path)
        if isinstance(raw, dict) and "nodes" in raw:
            self.nodes = raw["nodes"]
            return

        # Fall back to rebuilding from JSONL ledger
        for event in read_jsonl(self._nodes_path):
            node_data = event.get("data", {})
            node_id = node_data.get("node_id")
            if node_id:
                self.nodes[node_id] = node_data

    def _propagate_visits(self, node_id: str) -> None:
        """Backpropagate visit count from node to root."""
        current_id: str | None = node_id
        visited: set[str] = set()

        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            node = self.nodes.get(current_id)
            if node is None:
                break
            node["visit_count"] = node.get("visit_count", 0) + 1
            current_id = node.get("parent_node_id")

    def _auto_prune_check(self, node_id: str) -> None:
        """Check if any auto-pruning rules should fire after a result is recorded."""
        policy = self.meta.get("pruning_policy", "agent_directed")
        if policy == "agent_directed":
            return

        pc = self.meta.get("pruning_config", {})
        if not pc:
            return

        # Threshold-based pruning: prune siblings worse than threshold
        threshold = pc.get("metric_threshold")
        if threshold is not None:
            node = self.nodes.get(node_id)
            if node and node.get("result_metric") is not None:
                metric = node["result_metric"]
                minimize = self.meta["metric_direction"] == "minimize"
                if minimize and metric > float(threshold):
                    self.prune_node(node_id, f"metric {metric} > threshold {threshold}")
                elif not minimize and metric < float(threshold):
                    self.prune_node(node_id, f"metric {metric} < threshold {threshold}")

    def _update_best(self, node_id: str, metric: float) -> None:
        """Update best_node_id and best_metric if this node is better."""
        minimize = self.meta["metric_direction"] == "minimize"
        current_best = self.meta.get("best_metric")

        if current_best is None:
            self.meta["best_node_id"] = node_id
            self.meta["best_metric"] = metric
        elif minimize and metric < current_best:
            self.meta["best_node_id"] = node_id
            self.meta["best_metric"] = metric
        elif not minimize and metric > current_best:
            self.meta["best_node_id"] = node_id
            self.meta["best_metric"] = metric

    @staticmethod
    def _make_node_id() -> str:
        """Generate a unique node ID."""
        return uuid.uuid4().hex[:12]
