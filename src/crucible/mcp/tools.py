"""MCP tool implementations for Crucible fleet operations."""
from __future__ import annotations

import json
from typing import Any

from crucible.core.config import ProjectConfig, load_config
from crucible.core.io import read_jsonl
from crucible.core.log import utc_now_iso


def _get_config() -> ProjectConfig:
    return load_config()


def get_fleet_status(args: dict[str, Any]) -> dict[str, Any]:
    """Node inventory, health summary, and current assignments."""
    config = _get_config()
    try:
        from crucible.fleet.inventory import load_nodes, summarize_nodes

        nodes = load_nodes(config.project_root / config.nodes_file)
        summary = summarize_nodes(nodes)
        node_details = [
            {
                "name": n.get("name"),
                "node_id": n.get("node_id"),
                "state": n.get("state"),
                "gpu": n.get("gpu"),
                "ssh_host": n.get("ssh_host"),
                "env_ready": n.get("env_ready", False),
                "dataset_ready": n.get("dataset_ready", False),
            }
            for n in nodes
        ]
        return {"summary": summary, "nodes": node_details}
    except Exception as exc:
        return {"error": str(exc), "nodes": []}


def get_leaderboard(args: dict[str, Any]) -> dict[str, Any]:
    """Top N experiment results sorted by primary metric."""
    config = _get_config()
    top_n = args.get("top_n", 20)
    try:
        from crucible.analysis.leaderboard import leaderboard
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        top = leaderboard(results, top_n=top_n)
        entries = []
        for i, r in enumerate(top, 1):
            res = r.get("result", {})
            entries.append(
                {
                    "rank": i,
                    "name": r.get("name", ""),
                    "val_bpb": res.get("val_bpb"),
                    "val_loss": res.get("val_loss"),
                    "steps_completed": res.get("steps_completed"),
                    "model_bytes": r.get("model_bytes"),
                }
            )
        return {"total_completed": len(results), "top": entries}
    except Exception as exc:
        return {"error": str(exc), "top": []}


def get_queue_status(args: dict[str, Any]) -> dict[str, Any]:
    """Fleet queue state: counts of queued, running, and completed experiments."""
    config = _get_config()
    try:
        from crucible.fleet.queue import load_queue, summarize_queue

        rows = load_queue(config.project_root / config.fleet_results_file)
        summary = summarize_queue(rows)
        return {"total": len(rows), "summary": summary}
    except Exception as exc:
        return {"error": str(exc)}


def enqueue_experiment(args: dict[str, Any]) -> dict[str, Any]:
    """Add an experiment configuration to the fleet queue."""
    config = _get_config()
    try:
        from crucible.fleet.queue import enqueue_experiments

        experiment = {
            "name": args["name"],
            "config": args["config"],
            "tier": args.get("tier", "proxy"),
            "backend": args.get("backend", "torch"),
            "tags": args.get("tags", []),
        }
        added = enqueue_experiments(
            [experiment],
            queue_path=config.project_root / "fleet_queue.jsonl",
            limit=1,
        )
        if added:
            return {"status": "enqueued", "run_id": added[0]["run_id"], "item": added[0]}
        return {"status": "skipped", "reason": "Experiment with same name and tier already exists."}
    except Exception as exc:
        return {"error": str(exc)}


def get_experiment_result(args: dict[str, Any]) -> dict[str, Any]:
    """Get the result for a specific experiment run_id."""
    config = _get_config()
    run_id = args["run_id"]
    try:
        from crucible.analysis.results import load_all_results

        for row in load_all_results(config):
            if row.get("id") == run_id or row.get("run_id") == run_id:
                return {"found": True, "result": row}
        return {"found": False, "run_id": run_id}
    except Exception as exc:
        return {"error": str(exc)}


def provision_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Create N new compute nodes."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        new_nodes = fleet.provision(
            count=args.get("count", 2),
            name_prefix=args.get("name_prefix", "crucible"),
        )
        return {
            "created": len(new_nodes),
            "new_nodes": [{"name": n.get("name"), "node_id": n.get("node_id")} for n in new_nodes],
        }
    except Exception as exc:
        return {"error": str(exc)}


def destroy_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Tear down tracked nodes."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        node_names = args.get("node_names") or None
        fleet.destroy(node_names=node_names)
        return {"destroyed": node_names or "all", "status": "ok"}
    except Exception as exc:
        return {"error": str(exc)}


def sync_code(args: dict[str, Any]) -> dict[str, Any]:
    """Push local code to nodes via rsync."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        # Simple sync implementation
        from crucible.fleet.inventory import load_nodes
        from crucible.fleet.sync import sync_repo

        nodes = load_nodes(config.project_root / config.nodes_file)
        node_names = args.get("node_names")
        selected = set(node_names) if node_names else None
        synced = []
        errors = []
        for node in nodes:
            if selected and node["name"] not in selected:
                continue
            try:
                sync_repo(node, config.project_root, config.sync_excludes)
                synced.append(node["name"])
            except Exception as exc:
                errors.append({"node": node["name"], "error": str(exc)})
        return {"synced": synced, "errors": errors}
    except Exception as exc:
        return {"error": str(exc)}


def get_research_state(args: dict[str, Any]) -> dict[str, Any]:
    """Current research state: hypotheses, beliefs, and budget info."""
    config = _get_config()
    state_path = config.project_root / "research_state.jsonl"
    if not state_path.exists():
        return {"available": False}
    try:
        from crucible.researcher.state import ResearchState

        state = ResearchState(state_path)
        return {
            "available": True,
            "hypotheses_count": len(state.hypotheses),
            "history_count": len(state.history),
            "beliefs": state.beliefs,
            "budget_remaining": state.budget_remaining,
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_sensitivity(args: dict[str, Any]) -> dict[str, Any]:
    """Parameter sensitivity analysis."""
    config = _get_config()
    try:
        from crucible.analysis.leaderboard import sensitivity_analysis
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        sens = sensitivity_analysis(results)
        return {"parameters": {k: v for k, v in list(sens.items())[:20]}}
    except Exception as exc:
        return {"error": str(exc)}


# Dispatch table
TOOL_DISPATCH: dict[str, Any] = {
    "get_fleet_status": get_fleet_status,
    "get_leaderboard": get_leaderboard,
    "get_queue_status": get_queue_status,
    "enqueue_experiment": enqueue_experiment,
    "get_experiment_result": get_experiment_result,
    "provision_nodes": provision_nodes,
    "destroy_nodes": destroy_nodes,
    "sync_code": sync_code,
    "get_research_state": get_research_state,
    "get_sensitivity": get_sensitivity,
}
