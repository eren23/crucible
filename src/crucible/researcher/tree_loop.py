"""Tree-search-based autonomous research loop.

Wraps :class:`SearchTree` with LLM-driven hypothesis generation and
fleet queue integration so that a single ``run_iteration`` call selects
promising nodes, generates children, and enqueues them for execution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.errors import ResearcherError
from crucible.core.log import utc_now_iso
from crucible.researcher.search_tree import SearchTree


class TreeSearchResearcher:
    """Autonomous research loop organized as tree search."""

    def __init__(
        self,
        config: ProjectConfig,
        tree_name: str,
        tree_dir: Path | None = None,
        max_iterations: int = 10,
        n_children: int = 3,
        llm: Any = None,
    ) -> None:
        self.config = config
        self.tree_name = tree_name
        self.max_iterations = max_iterations
        self.n_children = n_children
        self._iteration = 0

        # Resolve tree directory
        if tree_dir is None:
            tree_dir = config.project_root / ".crucible" / "search_trees" / tree_name

        self.tree_dir = Path(tree_dir)

        # Load or create the tree
        if (self.tree_dir / "tree.yaml").exists():
            self.tree = SearchTree.load(self.tree_dir)
        else:
            self.tree = SearchTree.create(
                tree_dir=self.tree_dir,
                name=tree_name,
                description=f"Tree search: {tree_name}",
                primary_metric=config.metrics.primary,
                metric_direction="minimize",
            )

        # LLM client (lazy import if not provided)
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Lazy-load the LLM client on first access."""
        if self._llm is None:
            from crucible.researcher.llm_client import AnthropicClient
            self._llm = AnthropicClient(model=self.config.researcher.model)
        return self._llm

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def run_iteration(self) -> dict[str, Any]:
        """Run one iteration: select expandable nodes, generate children, enqueue pending.

        Returns a summary dict describing what was done.
        """
        self._iteration += 1
        summary: dict[str, Any] = {
            "iteration": self._iteration,
            "expanded_nodes": [],
            "new_children": [],
            "enqueued": [],
        }

        # 1. Find nodes to expand
        expandable = self.tree.get_expandable_nodes()
        if not expandable:
            # Check if there are pending nodes to enqueue instead
            pending = self.tree.get_pending_nodes()
            if pending:
                summary["enqueued"] = [n["node_id"] for n in pending]
            summary["message"] = (
                "No expandable nodes (need completed results before expanding)"
                if not pending
                else f"No expandable nodes, but {len(pending)} pending nodes ready"
            )
            return summary

        # 2. For each expandable node, generate children
        for node in expandable:
            node_id = node["node_id"]
            children_specs = self.generate_children(node_id, n_children=self.n_children)
            if not children_specs:
                continue

            new_ids = self.tree.expand_node(node_id, children_specs)
            summary["expanded_nodes"].append(node_id)
            summary["new_children"].extend(new_ids)

        # 3. Report pending nodes (callers can enqueue them via fleet)
        pending = self.tree.get_pending_nodes()
        summary["enqueued"] = [n["node_id"] for n in pending]
        summary["message"] = (
            f"Expanded {len(summary['expanded_nodes'])} nodes, "
            f"created {len(summary['new_children'])} children, "
            f"{len(summary['enqueued'])} pending"
        )

        return summary

    # ------------------------------------------------------------------
    # Child generation
    # ------------------------------------------------------------------

    def generate_children(
        self, node_id: str, n_children: int = 3
    ) -> list[dict[str, Any]]:
        """Generate child configs for a node using LLM with tree context.

        Returns a list of child specs compatible with
        :meth:`SearchTree.expand_node`.
        """
        from crucible.researcher.hypothesis import generate_tree_hypotheses

        return generate_tree_hypotheses(
            tree=self.tree,
            node_id=node_id,
            llm=self.llm,
            n_children=n_children,
        )

    # ------------------------------------------------------------------
    # Result sync and pruning
    # ------------------------------------------------------------------

    def sync_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Match experiment results back to tree nodes and record them.

        *results* is a list of dicts, each with at minimum ``name`` and the
        primary metric key.  We match by ``experiment_name``.

        Returns a summary of what was synced.
        """
        synced: list[str] = []
        pending = self.tree.get_pending_nodes()

        # Build lookup by experiment name
        pending_by_name: dict[str, dict[str, Any]] = {}
        for node in pending:
            pending_by_name[node["experiment_name"]] = node

        for result in results:
            name = result.get("name", "")
            if name in pending_by_name:
                node = pending_by_name[name]
                self.tree.record_result(node["node_id"], result)
                synced.append(node["node_id"])

        return {
            "synced_count": len(synced),
            "synced_node_ids": synced,
        }

    def auto_prune(self, threshold: float | None = None) -> dict[str, Any]:
        """Prune branches whose metrics are significantly worse than best.

        If *threshold* is ``None``, uses the tree's pruning_config threshold
        or skips pruning.

        Returns summary of what was pruned.
        """
        if threshold is None:
            pc = self.tree.meta.get("pruning_config", {})
            threshold = pc.get("metric_threshold")
            if threshold is None:
                return {"pruned_count": 0, "message": "No threshold configured"}
            threshold = float(threshold)

        minimize = self.tree.meta["metric_direction"] == "minimize"
        pruned: list[str] = []

        for node in list(self.tree.nodes.values()):
            if node["status"] != "completed":
                continue
            metric = node.get("result_metric")
            if metric is None:
                continue

            should_prune = (
                (minimize and metric > threshold)
                or (not minimize and metric < threshold)
            )
            if should_prune:
                self.tree.prune_branch(
                    node["node_id"],
                    reason=f"metric {metric} {'>' if minimize else '<'} threshold {threshold}",
                )
                pruned.append(node["node_id"])

        return {
            "pruned_count": len(pruned),
            "pruned_node_ids": pruned,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current tree status summary."""
        summary = self.tree.get_tree_summary()
        summary["iteration"] = self._iteration
        summary["max_iterations"] = self.max_iterations
        summary["pending_nodes"] = len(self.tree.get_pending_nodes())
        summary["expandable_nodes"] = len(self.tree.get_expandable_nodes())
        summary["frontier_nodes"] = len(self.tree.get_frontier())
        return summary

    def get_pending_configs(self) -> list[dict[str, Any]]:
        """Return configs for all pending nodes, ready for fleet enqueue.

        Each item has ``name`` and ``config`` keys matching the format
        expected by ``FleetManager.enqueue``.
        """
        return [
            {
                "name": node["experiment_name"],
                "config": dict(node["config"]),
                "node_id": node["node_id"],
                "tags": node.get("tags", []) + [f"tree:{self.tree_name}"],
            }
            for node in self.tree.get_pending_nodes()
        ]
