"""MCP tool implementations for Crucible fleet operations."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from crucible.core.config import ProjectConfig, load_config
from crucible.core.errors import CrucibleError
from crucible.core.experiment_contract import (
    contract_metadata,
    resolve_wandb_settings,
    validate_experiment_contract,
)
from crucible.core.io import read_jsonl
from crucible.core.log import utc_now_iso


def _get_config() -> ProjectConfig:
    return load_config()


def _get_hub_store() -> Any:
    """Return a HubStore honoring project config hub_dir overrides."""
    from crucible.core.hub import HubStore

    config = _get_config()
    hub_dir = HubStore.resolve_hub_dir(config_hub_dir=config.hub_dir)
    return HubStore(hub_dir=hub_dir)


def _queue_contract_fields(config: ProjectConfig) -> dict[str, Any]:
    metadata = validate_experiment_contract(
        config,
        action="MCP experiment enqueue",
        execution_mode="remote",
    )
    return {
        "execution_provider": metadata["execution_provider"],
        "contract_status": metadata["contract_status"],
        "wandb": metadata["wandb"],
    }


def _project_contract_env(config: ProjectConfig, spec: Any) -> dict[str, str]:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in getattr(spec, "env_set", {}).items()})
    wandb = resolve_wandb_settings(config, env=env)
    validate_experiment_contract(
        config,
        action=f"external project {spec.name}",
        execution_mode="remote",
        env=env,
    )
    merged: dict[str, str] = {}
    if wandb["project"]:
        merged["WANDB_PROJECT"] = wandb["project"]
    if wandb["entity"]:
        merged["WANDB_ENTITY"] = str(wandb["entity"])
    if wandb["mode"]:
        merged["WANDB_MODE"] = wandb["mode"]
    return merged


def _probe_node_metrics(node: dict[str, Any]) -> dict[str, Any]:
    """SSH to a node and collect GPU/memory/disk metrics."""
    from crucible.fleet.sync import remote_exec

    name = node.get("name", "unknown")
    if not node.get("ssh_host"):
        return {"node": name, "error": "no SSH host"}
    try:
        cmd = (
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
            "--format=csv,noheader,nounits 2>/dev/null; echo '---SEP---'; "
            "free -m 2>/dev/null | grep Mem; echo '---SEP---'; "
            "df -BM /workspace 2>/dev/null | tail -1"
        )
        proc = remote_exec(node, cmd, check=False)
        parts = (proc.stdout or "").split("---SEP---")

        result: dict[str, Any] = {"node": name}

        # Parse nvidia-smi
        if len(parts) >= 1 and parts[0].strip():
            gpu_parts = parts[0].strip().split(",")
            if len(gpu_parts) >= 4:
                result["gpu_utilization_pct"] = int(gpu_parts[0].strip())
                result["gpu_memory_used_mb"] = int(gpu_parts[1].strip())
                result["gpu_memory_total_mb"] = int(gpu_parts[2].strip())
                result["gpu_temperature_c"] = int(gpu_parts[3].strip())

        # Parse free
        if len(parts) >= 2 and parts[1].strip():
            mem_parts = parts[1].strip().split()
            if len(mem_parts) >= 3:
                result["ram_total_mb"] = int(mem_parts[1])
                result["ram_used_mb"] = int(mem_parts[2])

        # Parse df
        if len(parts) >= 3 and parts[2].strip():
            disk_parts = parts[2].strip().split()
            if len(disk_parts) >= 5:
                result["disk_used_pct"] = disk_parts[4]

        return result
    except Exception as exc:
        return {"node": name, "error": str(exc)}


def get_fleet_status(args: dict[str, Any]) -> dict[str, Any]:
    """Node inventory, health summary, current assignments, and optional live metrics."""
    config = _get_config()
    include_metrics = args.get("include_metrics", False)
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
        result: dict[str, Any] = {"summary": summary, "nodes": node_details}
        active_project_runs = [
            {
                "run_id": row.get("run_id"),
                "launch_id": row.get("launch_id"),
                "project": row.get("project"),
                "variant_name": row.get("variant_name"),
                "node_name": row.get("node_name") or row.get("remote_node"),
                "status": row.get("status"),
                "updated_at": row.get("updated_at"),
            }
            for row in _load_project_runs()
            if row.get("status") in _PROJECT_ACTIVE_STATUSES
        ]
        if active_project_runs:
            result["active_project_runs"] = active_project_runs

        if include_metrics and nodes:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            ssh_nodes = [n for n in nodes if n.get("ssh_host")]
            metrics = []
            with ThreadPoolExecutor(max_workers=min(8, len(ssh_nodes) or 1)) as pool:
                futures = {pool.submit(_probe_node_metrics, n): n for n in ssh_nodes}
                for fut in as_completed(futures, timeout=30):
                    try:
                        metrics.append(fut.result(timeout=10))
                    except Exception:
                        node = futures[fut]
                        metrics.append({"node": node.get("name", "?"), "error": "probe timeout"})
            result["metrics"] = metrics

        return result
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}", "nodes": []}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}", "nodes": []}


def get_leaderboard(args: dict[str, Any]) -> dict[str, Any]:
    """Top N experiment results sorted by primary metric."""
    config = _get_config()
    top_n = args.get("top_n", 20)
    primary = config.metrics.primary
    secondary = config.metrics.secondary or ""
    try:
        from crucible.analysis.leaderboard import leaderboard
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        top = leaderboard(results, top_n=top_n, cfg=config)
        entries = []
        for i, r in enumerate(top, 1):
            res = r.get("result", {})
            entry: dict[str, Any] = {
                "rank": i,
                "name": r.get("name", ""),
                "primary_metric": primary,
                primary: res.get(primary),
                "steps_completed": res.get("steps_completed"),
                "model_bytes": r.get("model_bytes"),
                "contract_status": r.get("contract_status", "legacy_missing_contract"),
            }
            if secondary:
                entry[secondary] = res.get(secondary)
            entries.append(entry)
        return {"total_completed": len(results), "primary_metric": primary, "top": entries}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}", "top": []}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}", "top": []}


def get_queue_status(args: dict[str, Any]) -> dict[str, Any]:
    """Fleet queue state: counts of queued, running, and completed experiments."""
    config = _get_config()
    try:
        from crucible.fleet.queue import load_queue, summarize_queue

        rows = load_queue(config.project_root / "fleet_queue.jsonl")
        summary = summarize_queue(rows)
        return {"total": len(rows), "summary": summary}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def enqueue_experiment(args: dict[str, Any]) -> dict[str, Any]:
    """Add an experiment configuration to the fleet queue."""
    config = _get_config()
    try:
        from crucible.fleet.queue import enqueue_experiments
        from crucible.runner.fingerprint import build_run_manifest

        contract = _queue_contract_fields(config)
        manifest = build_run_manifest(config.project_root)
        experiment = {
            "name": args["name"],
            "config": args["config"],
            "tier": args.get("tier", "proxy"),
            "backend": args.get("backend", "torch"),
            "tags": args.get("tags", []),
            "run_manifest": manifest,
            **contract,
        }
        added = enqueue_experiments(
            config.project_root / "fleet_queue.jsonl",
            [experiment],
            limit=1,
        )
        if added:
            return {"status": "enqueued", "run_id": added[0]["run_id"], "item": added[0]}
        return {"status": "skipped", "reason": "Experiment with same name and tier already exists."}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def get_experiment_result(args: dict[str, Any]) -> dict[str, Any]:
    """Get the result for a specific experiment run_id."""
    config = _get_config()
    run_id = args["run_id"]
    try:
        from crucible.analysis.results import merged_results

        for row in merged_results(config):
            if row.get("id") == run_id or row.get("run_id") == run_id:
                row = dict(row)
                row["contract_status"] = row.get("contract_status", "legacy_missing_contract")
                return {"found": True, "result": row}
        return {"found": False, "run_id": run_id}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def provision_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Create N new compute nodes."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        kwargs: dict[str, Any] = {}
        if args.get("network_volume_id"):
            kwargs["network_volume_id"] = args["network_volume_id"]
        if args.get("template_id"):
            kwargs["template_id"] = args["template_id"]
        if args.get("gpu_count"):
            kwargs["gpu_count"] = args["gpu_count"]
        new_nodes = fleet.provision(
            count=args.get("count", 2),
            name_prefix=args.get("name_prefix", "crucible"),
            **kwargs,
        )
        return {
            "created": len(new_nodes),
            "new_nodes": [{"name": n.get("name"), "node_id": n.get("node_id")} for n in new_nodes],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def destroy_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Tear down nodes. Supports names, pod IDs, or destroy-all."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        node_names = args.get("node_names") or None
        pod_ids = args.get("pod_ids") or None
        selected = set(node_names) if node_names else None

        destroyed_names: list[str] = []
        destroyed_ids: list[str] = []

        # Mode 1: Direct pod ID destruction (bypasses inventory entirely)
        if pod_ids:
            if config.provider.type.lower() == "runpod":
                from crucible.fleet.providers.runpod import RunPodProvider
                provider = fleet.provider
                if isinstance(provider, RunPodProvider):
                    destroyed_ids = provider.destroy_pods_by_id(pod_ids)
                    # Also remove from inventory if present
                    from crucible.fleet.inventory import load_nodes_if_exists, save_nodes
                    nodes = load_nodes_if_exists(fleet.nodes_file)
                    nodes = [n for n in nodes if (n.get("pod_id") or n.get("node_id")) not in set(pod_ids)]
                    save_nodes(fleet.nodes_file, nodes)
            return {"destroyed_pod_ids": destroyed_ids, "status": "ok"}

        # Mode 2: Destroy by name or all
        fleet.destroy(selected_names=selected)
        destroyed_names = list(selected) if selected else ["all"]

        # Also clean up orphaned pods via RunPod API
        orphan_destroyed: list[str] = []
        if config.provider.type.lower() == "runpod":
            from crucible.fleet.providers.runpod import RunPodProvider
            provider = fleet.provider
            if isinstance(provider, RunPodProvider):
                if not node_names:
                    # No names: destroy ALL pods in the account
                    orphan_destroyed = provider.destroy_all_pods()
                else:
                    # Names specified: find orphans with matching names
                    from crucible.fleet.inventory import load_nodes_if_exists
                    remaining = load_nodes_if_exists(fleet.nodes_file)
                    tracked_ids = {n.get("pod_id") or n.get("node_id") for n in remaining}
                    for pod in provider.list_all_pods():
                        pod_name = str(pod.get("name") or "")
                        pod_id = str(pod.get("id") or "")
                        if pod_name in selected and pod_id not in tracked_ids:
                            orphan_destroyed.extend(provider.destroy_pods_by_id([pod_id]))

        result: dict[str, Any] = {
            "destroyed": destroyed_names if node_names else "all",
            "status": "ok",
        }
        if orphan_destroyed:
            result["orphan_pods_destroyed"] = orphan_destroyed
        return result
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def cleanup_orphans(args: dict[str, Any]) -> dict[str, Any]:
    """List and optionally destroy pods on the provider that aren't in local inventory.

    REQUIRES: Provider supports pod listing (RunPod does; SSH does not).
    RETURNS: {orphans: [{name, pod_id}], destroyed: [pod_id, ...], total_orphans: int}
    NEXT: If destroy=False, review the list and re-run with destroy=True, or
          use destroy_nodes with pod_ids to destroy specific ones.
    """
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        result = fleet.cleanup_orphans(destroy=bool(args.get("destroy", False)))
        return {
            "orphans": result["orphans"],
            "destroyed": result["destroyed"],
            "total_orphans": len(result["orphans"]),
            "status": "ok",
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def stop_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Stop running pods to save cost.  Disk and bootstrap state are preserved."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        node_names = args.get("node_names") or None
        selected = set(node_names) if node_names else None
        updated = fleet.stop(selected_names=selected)
        stopped = [n["name"] for n in updated if n.get("state") == "stopped"]
        return {"stopped": stopped, "status": "ok"}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def start_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Start stopped pods and wait for SSH readiness."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        node_names = args.get("node_names") or None
        selected = set(node_names) if node_names else None
        updated = fleet.start(selected_names=selected)
        started = [n["name"] for n in updated if n.get("state") in {"ready", "new", "running"}]
        return {"started": started, "status": "ok"}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def runpod_list_volumes(args: dict[str, Any]) -> dict[str, Any]:
    """List RunPod network volumes."""
    _get_config()  # ensure .env files are loaded (RUNPOD_API_KEY)
    try:
        from crucible.fleet.providers.runpod import runpod_list_network_volumes

        volumes = runpod_list_network_volumes()
        return {"volumes": volumes, "count": len(volumes)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def runpod_create_volume(args: dict[str, Any]) -> dict[str, Any]:
    """Create a persistent RunPod network volume."""
    _get_config()  # ensure .env files are loaded (RUNPOD_API_KEY)
    try:
        from crucible.fleet.providers.runpod import runpod_create_network_volume

        volume = runpod_create_network_volume(
            name=args["name"],
            size_gb=args.get("size_gb", 100),
            datacenter_id=args.get("datacenter_id", "US-GA-1"),
        )
        return {"volume": volume, "status": "ok"}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def runpod_delete_volume(args: dict[str, Any]) -> dict[str, Any]:
    """Delete a RunPod network volume."""
    _get_config()  # ensure .env files are loaded (RUNPOD_API_KEY)
    try:
        from crucible.fleet.providers.runpod import runpod_delete_network_volume

        runpod_delete_network_volume(args["volume_id"])
        return {"deleted": args["volume_id"], "status": "ok"}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def runpod_gpu_availability(args: dict[str, Any]) -> dict[str, Any]:
    """List available GPU types with pricing."""
    _get_config()  # ensure .env files are loaded (RUNPOD_API_KEY)
    try:
        from crucible.fleet.providers.runpod import runpod_list_gpu_types

        gpu_types = runpod_list_gpu_types(
            gpu_count=args.get("gpu_count", 1),
            secure_cloud=args.get("secure_cloud"),
        )
        return {"gpu_types": gpu_types, "count": len(gpu_types)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def runpod_list_templates_tool(args: dict[str, Any]) -> dict[str, Any]:
    """List user's RunPod pod templates."""
    _get_config()  # ensure .env files are loaded (RUNPOD_API_KEY)
    try:
        from crucible.fleet.providers.runpod import runpod_list_templates

        templates = runpod_list_templates()
        return {"templates": templates, "count": len(templates)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def runpod_create_template_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Create a RunPod pod template."""
    _get_config()  # ensure .env files are loaded (RUNPOD_API_KEY)
    try:
        from crucible.fleet.providers.runpod import runpod_create_template

        template = runpod_create_template(
            name=args["name"],
            image=args["image"],
            container_disk_gb=args.get("container_disk_gb", 20),
            volume_gb=args.get("volume_gb", 40),
            ports=args.get("ports", "22/tcp,8888/http"),
            env=args.get("env"),
        )
        return {"template": template, "status": "ok"}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def sync_code(args: dict[str, Any]) -> dict[str, Any]:
    """Push local code to nodes via rsync (Crucible repo + taps)."""
    config = _get_config()
    try:
        from crucible.fleet.bootstrap import _materialize_global_architectures
        from crucible.fleet.manager import FleetManager

        _materialize_global_architectures(config.project_root)
        fleet = FleetManager(config)
        # Simple sync implementation
        from crucible.fleet.inventory import load_nodes
        from crucible.fleet.sync import sync_repo, sync_taps

        nodes = load_nodes(config.project_root / config.nodes_file)
        node_names = args.get("node_names")
        selected = set(node_names) if node_names else None
        synced = []
        errors = []
        for node in nodes:
            if selected and node["name"] not in selected:
                continue
            try:
                sync_repo(node, project_root=config.project_root, sync_excludes=config.sync_excludes)
                # Also rsync ~/.crucible-hub/taps/ so tap architectures, launchers,
                # and data files are pushed in the same operation. Idempotent,
                # no-op if the user has no taps installed.
                sync_taps(node)
                synced.append(node["name"])
            except Exception as exc:
                errors.append({"node": node["name"], "error": str(exc)})
        return {"synced": synced, "errors": errors}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def fleet_refresh(args: dict[str, Any]) -> dict[str, Any]:
    """Refresh node states from the cloud provider API (updates SSH hosts, GPU info, state)."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(config)
        nodes = fm.refresh()
        return {
            "refreshed": len(nodes),
            "nodes": [
                {
                    "name": n.get("name"),
                    "state": n.get("state"),
                    "ssh_host": n.get("ssh_host"),
                    "ssh_port": n.get("ssh_port", 22),
                    "gpu": n.get("gpu"),
                    "env_ready": n.get("env_ready", False),
                    "dataset_ready": n.get("dataset_ready", False),
                }
                for n in nodes
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def bootstrap_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Bootstrap fleet nodes: sync code, install deps, download data. Run after provision_nodes."""
    config = _get_config()
    try:
        # Precondition: nodes with SSH hosts must exist
        from crucible.fleet.inventory import load_nodes_if_exists
        nodes_check = load_nodes_if_exists(config.project_root / config.nodes_file)
        ssh_nodes = [n for n in nodes_check if n.get("ssh_host")]
        if not ssh_nodes:
            return {"error": "No nodes with SSH connectivity found. Run fleet_refresh first (wait ~60s after provision_nodes).", "total": 0, "bootstrapped": 0, "nodes": []}

        from crucible.fleet.manager import FleetManager

        fm = FleetManager(config)
        train_shards = args.get("train_shards", 1)
        skip_install = args.get("skip_install", False)
        skip_data = args.get("skip_data", False)
        node_names = args.get("node_names")
        selected = set(node_names) if node_names else None

        nodes = fm.bootstrap(
            train_shards=train_shards,
            skip_install=skip_install,
            skip_data=skip_data,
            selected_names=selected,
        )
        bootstrapped = [n for n in nodes if n.get("state") == "bootstrapped"]
        return {
            "total": len(nodes),
            "bootstrapped": len(bootstrapped),
            "nodes": [
                {"name": n.get("name"), "state": n.get("state"), "env_ready": n.get("env_ready"), "dataset_ready": n.get("dataset_ready")}
                for n in nodes
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def dispatch_experiments(args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch queued experiments to idle bootstrapped nodes. Run after bootstrap_nodes + enqueue."""
    config = _get_config()
    try:
        validate_experiment_contract(
            config,
            action="MCP dispatch_experiments",
            execution_mode="remote",
        )
        # Precondition: bootstrapped nodes must exist
        from crucible.fleet.inventory import load_nodes_if_exists
        nodes_check = load_nodes_if_exists(config.project_root / config.nodes_file)
        bootstrapped = [n for n in nodes_check if n.get("env_ready")]
        if not bootstrapped:
            return {"error": "No bootstrapped nodes found. Run bootstrap_nodes first (after provision_nodes + fleet_refresh).", "dispatched": 0, "assignments": []}

        from crucible.fleet.manager import FleetManager

        fm = FleetManager(config)
        max_assignments = args.get("max_assignments", 8)
        assignments = fm.dispatch(max_assignments=max_assignments)
        return {
            "dispatched": len(assignments),
            "assignments": [
                {"node": a.get("assigned_node", a.get("assigned_pod", "")), "experiment": a.get("experiment_name", "")}
                for a in assignments
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def collect_results(args: dict[str, Any]) -> dict[str, Any]:
    """Collect experiment results from all fleet nodes via rsync and merge into fleet results."""
    config = _get_config()
    try:
        # Precondition: nodes must exist
        from crucible.fleet.inventory import load_nodes_if_exists
        nodes_check = load_nodes_if_exists(config.project_root / config.nodes_file)
        if not nodes_check:
            return {"error": "No fleet nodes found. Run provision_nodes + fleet_refresh first.", "collected": False, "total_results": 0, "completed": 0}

        from crucible.fleet.manager import FleetManager
        from crucible.analysis.results import merged_results
        from crucible.fleet.queue import load_queue, save_queue, reconcile_queue_with_results
        from crucible.core.io import read_jsonl

        fm = FleetManager(config)
        fm.collect()
        results = merged_results(config)
        completed = [r for r in results if r.get("status") == "completed"]

        # Reconcile queue: mark finished experiments so nodes become available
        queue_path = config.project_root / "fleet_queue.jsonl"
        result_index = {r["id"]: r for r in results if "id" in r}
        queue = reconcile_queue_with_results(load_queue(queue_path), result_index)
        save_queue(queue_path, queue)

        return {
            "collected": True,
            "total_results": len(results),
            "completed": len(completed),
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def get_research_state(args: dict[str, Any]) -> dict[str, Any]:
    """Current research state: hypotheses, beliefs, and budget info."""
    config = _get_config()
    state_path = config.project_root / config.research_state_file
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
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def get_sensitivity(args: dict[str, Any]) -> dict[str, Any]:
    """Parameter sensitivity analysis."""
    config = _get_config()
    try:
        from crucible.analysis.leaderboard import sensitivity_analysis
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        sens = sensitivity_analysis(results, cfg=config)
        return {"parameters": {k: v for k, v in list(sens.items())[:20]}}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Design tools
# ---------------------------------------------------------------------------


def _get_store():
    """Lazy-load a VersionStore from project config."""
    from crucible.core.store import VersionStore

    config = _get_config()
    store_dir = config.project_root / config.store_dir
    return VersionStore(store_dir)


def design_browse_experiments(args: dict[str, Any]) -> dict[str, Any]:
    """Browse experiments with filtering across local, project, and fleet sources."""
    config = _get_config()
    try:
        from crucible.analysis.results import merged_results

        results = merged_results(config)
        primary = config.metrics.primary

        # Apply filters
        name_pattern = args.get("name_pattern", "")
        family = args.get("family", "")
        tag = args.get("tag", "")
        metric_below = args.get("metric_below")
        metric_above = args.get("metric_above")
        config_filter = args.get("config_filter", {})
        limit = args.get("limit", 50)
        sort_by = args.get("sort_by", "metric")

        filtered = []
        for r in results:
            if name_pattern and name_pattern not in r.get("name", ""):
                continue
            if family and r.get("config", {}).get("MODEL_FAMILY", "") != family:
                continue
            if tag and tag not in r.get("tags", []):
                continue
            metric_val = r.get("result", {}).get(primary)
            if metric_below is not None and isinstance(metric_val, (int, float)) and metric_val >= metric_below:
                continue
            if metric_above is not None and isinstance(metric_val, (int, float)) and metric_val <= metric_above:
                continue
            if config_filter:
                exp_config = r.get("config", {})
                if not all(exp_config.get(k) == v for k, v in config_filter.items()):
                    continue
            filtered.append(r)

        # Sort
        if sort_by == "metric":
            filtered.sort(key=lambda r: r.get("result", {}).get(primary, float("inf")))
        elif sort_by == "name":
            filtered.sort(key=lambda r: r.get("name", ""))
        elif sort_by == "timestamp":
            filtered.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        # Trim and format
        trimmed = []
        for r in filtered[:limit]:
            trimmed.append({
                "name": r.get("name"),
                "config": r.get("config", {}),
                primary: r.get("result", {}).get(primary),
                "model_bytes": r.get("model_bytes"),
                "tags": r.get("tags", []),
                "status": r.get("status"),
                "timestamp": r.get("timestamp"),
                "project": r.get("project"),
                "launcher": r.get("launcher"),
                "remote_node": r.get("remote_node"),
            })
        return {"total_matched": len(filtered), "experiments": trimmed}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def design_compare_experiments(args: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side comparison of 2-5 experiments."""
    config = _get_config()
    names = args.get("experiment_names", [])
    if len(names) < 2 or len(names) > 5:
        return {"error": "Provide 2-5 experiment names."}
    try:
        from crucible.analysis.results import merged_results

        all_results = merged_results(config)
        by_name = {r["name"]: r for r in all_results if r.get("name") in names}
        missing = [n for n in names if n not in by_name]
        if missing:
            return {"error": f"Experiments not found: {missing}"}

        primary = config.metrics.primary
        experiments = [by_name[n] for n in names]

        # Config diffs: find keys that differ
        all_keys = set()
        for exp in experiments:
            all_keys.update(exp.get("config", {}).keys())

        config_diff = {}
        for key in sorted(all_keys):
            values = [exp.get("config", {}).get(key, "<not set>") for exp in experiments]
            if len(set(values)) > 1:
                config_diff[key] = dict(zip(names, values))

        # Metric comparison
        metric_comparison = {}
        for name in names:
            exp = by_name[name]
            res = exp.get("result", {})
            metric_comparison[name] = {
                primary: res.get(primary),
                "model_bytes": exp.get("model_bytes"),
                "status": exp.get("status"),
            }

        return {
            "experiments": names,
            "config_diff": config_diff,
            "config_same_keys": sorted(all_keys - set(config_diff.keys())),
            "metrics": metric_comparison,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def design_generate_hypotheses(args: dict[str, Any]) -> dict[str, Any]:
    """Generate LLM-driven experiment hypotheses with agent-provided context."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY not set. Export it or add to .env file."}
    config = _get_config()
    try:
        from crucible.researcher.analysis import build_analysis
        from crucible.researcher.hypothesis import generate_hypotheses
        from crucible.researcher.llm_client import AnthropicClient
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)

        analysis = build_analysis(config, state)
        extra_context = args.get("extra_context", "")
        if extra_context:
            analysis += f"\n\n## Additional Agent Context\n{extra_context}"

        focus = args.get("focus_family", "")
        if focus:
            analysis += f"\n\n## Focus Area\nFocus on the '{focus}' model family."

        program_path = config.project_root / config.researcher.program_file
        program_text = program_path.read_text(encoding="utf-8") if program_path.exists() else ""

        llm = AnthropicClient(model=config.researcher.model)
        hypotheses = generate_hypotheses(analysis, program_text, state, llm, iteration=0)
        state.save()

        max_hyp = args.get("max_hypotheses", 5)
        return {
            "hypotheses": hypotheses[:max_hyp],
            "total_generated": len(hypotheses),
            "analysis_summary": analysis[:500] + "..." if len(analysis) > 500 else analysis,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def design_batch_from_hypotheses(args: dict[str, Any]) -> dict[str, Any]:
    """Convert hypotheses to an executable experiment batch."""
    config = _get_config()
    try:
        from crucible.researcher.batch_design import design_batch
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)

        hypotheses = args.get("hypotheses", [])
        tier = args.get("tier", "proxy")
        backend = args.get("backend", "torch")
        include_baseline = args.get("include_baseline", True)

        baseline_config = None
        if include_baseline:
            try:
                from crucible.analysis.leaderboard import leaderboard
                from crucible.analysis.results import completed_results

                results = completed_results(config)
                top = leaderboard(results, top_n=1, cfg=config)
                if top:
                    baseline_config = top[0].get("config", {})
            except ImportError:
                pass

        batch = design_batch(
            hypotheses, state, tier, backend, iteration=0,
            baseline_config=baseline_config if include_baseline else None,
        )
        return {"batch": batch, "batch_size": len(batch)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def design_enqueue_batch(args: dict[str, Any]) -> dict[str, Any]:
    """Enqueue a batch of experiment configs to the fleet queue."""
    config = _get_config()
    try:
        from crucible.fleet.queue import enqueue_experiments
        from crucible.runner.fingerprint import build_run_manifest

        contract = _queue_contract_fields(config)
        manifest = build_run_manifest(config.project_root)
        batch = args.get("batch", [])
        wave_name = args.get("wave_name", "")
        if not wave_name:
            wave_name = f"agent_{utc_now_iso()[:19].replace(':', '').replace('-', '')}"

        experiments = []
        for exp in batch:
            exp.setdefault("wave", wave_name)
            experiments.append({**exp, "run_manifest": manifest, **contract})

        added = enqueue_experiments(
            config.project_root / "fleet_queue.jsonl",
            experiments,
            limit=0,
        )
        return {
            "enqueued": len(added),
            "wave_name": wave_name,
            "run_ids": [item["run_id"] for item in added],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Context tools
# ---------------------------------------------------------------------------


def context_get_analysis(args: dict[str, Any]) -> dict[str, Any]:
    """Full structured analysis of experiment results."""
    config = _get_config()
    try:
        from crucible.researcher.analysis import build_analysis_structured
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)
        result = build_analysis_structured(config, state)

        # Inject hub findings if available
        try:
            from crucible.core.hub import HubStore

            hub = HubStore()
            if hub.initialized:
                active_track = hub.get_active_track() or config.active_track
                if active_track:
                    hub_findings = hub.load_context_for_track(
                        active_track, include_global=True, max_findings=20,
                    )
                    result["hub_findings"] = hub_findings
        except Exception:
            pass  # Hub is optional

        return result
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def context_push_finding(args: dict[str, Any]) -> dict[str, Any]:
    """Record a research finding in the context store."""
    config = _get_config()
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)

        entry = state.add_finding(
            finding=args["finding"],
            category=args.get("category", "observation"),
            source_experiments=args.get("source_experiments", []),
            confidence=args.get("confidence", 0.7),
            created_by=args.get("created_by", "mcp-agent"),
        )
        state.save()

        return {"status": "recorded", "entry": entry}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def context_get_findings(args: dict[str, Any]) -> dict[str, Any]:
    """Query accumulated research findings."""
    config = _get_config()
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)

        category = args.get("category", "")
        limit = args.get("limit", 50)
        findings = state.get_findings(category=category or None, limit=limit)
        return {"findings": findings, "total": len(findings)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Version tools
# ---------------------------------------------------------------------------


def version_save_design(args: dict[str, Any]) -> dict[str, Any]:
    """Save or update a versioned experiment design.

    Supports partial updates: if a design already exists, only fields
    present in args are overwritten. Unspecified fields keep their
    previous values.
    """
    try:
        import re

        store = _get_store()
        name = args["name"]

        # Validate name is a slug
        if not re.match(r'^[a-z0-9][a-z0-9_-]*$', name):
            return {"error": f"Invalid design name '{name}'. Use lowercase letters, numbers, hyphens, underscores. Must start with letter or number."}

        # Start from existing content if this is an update
        current = store.get_current("experiment_design", name)
        if current is not None:
            _, prev_content = current
            content = dict(prev_content)
        else:
            # Defaults for new designs
            content = {
                "name": name,
                "description": "",
                "hypothesis": "",
                "config": {},
                "base_preset": "proxy",
                "backend": "torch",
                "tags": [],
                "family": "",
                "status": "draft",
                "linked_run_ids": [],
                "parent_design": None,
                "rationale": "",
            }

        # Override only fields explicitly provided in args
        # (name is always set, linked_run_ids is managed internally)
        _UPDATABLE_FIELDS = [
            "description", "hypothesis", "config", "base_preset", "backend",
            "tags", "family", "status", "parent_design", "rationale",
        ]
        for field in _UPDATABLE_FIELDS:
            if field in args:
                content[field] = args[field]
        content["name"] = name  # always ensure name matches

        meta = store.create(
            "experiment_design",
            args["name"],
            content,
            summary=args.get("summary", f"Design: {args['name']}"),
            created_by=args.get("created_by", "mcp-agent"),
            tags=args.get("tags", []),
        )

        # Auto-commit if configured
        config = _get_config()
        if config.auto_commit_versions:
            store.git_commit_version(meta)

        return {"status": "saved", "version_meta": meta}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_list_designs(args: dict[str, Any]) -> dict[str, Any]:
    """List all versioned experiment designs."""
    try:
        store = _get_store()

        status_filter = args.get("status_filter")
        tag_filter = args.get("tag_filter")

        resources = store.list_resources(
            "experiment_design",
            status=status_filter,
            tag=tag_filter,
        )

        # Enrich with current content status
        designs = []
        for meta in resources:
            entry = dict(meta)
            current = store.get_current("experiment_design", meta["resource_name"])
            if current:
                _, content = current
                entry["design_status"] = content.get("status", "unknown")
                entry["family"] = content.get("family", "")
                entry["base_preset"] = content.get("base_preset", "")
            designs.append(entry)

        return {"designs": designs, "total": len(designs)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_diff(args: dict[str, Any]) -> dict[str, Any]:
    """Compare two versions of a design."""
    try:
        store = _get_store()

        resource_name = args["resource_name"]
        va = args["version_a"]
        vb = args["version_b"]

        result_a = store.get_version_number("experiment_design", resource_name, va)
        result_b = store.get_version_number("experiment_design", resource_name, vb)

        if result_a is None:
            return {"error": f"Version {va} not found for {resource_name}"}
        if result_b is None:
            return {"error": f"Version {vb} not found for {resource_name}"}

        meta_a, content_a = result_a
        meta_b, content_b = result_b

        # Compute diffs
        all_keys = set(content_a.keys()) | set(content_b.keys())
        changes = {}
        for key in sorted(all_keys):
            val_a = content_a.get(key)
            val_b = content_b.get(key)
            if val_a != val_b:
                changes[key] = {f"v{va}": val_a, f"v{vb}": val_b}

        return {
            "resource_name": resource_name,
            "version_a": {"version": va, "created_at": meta_a.get("created_at"), "summary": meta_a.get("summary")},
            "version_b": {"version": vb, "created_at": meta_b.get("created_at"), "summary": meta_b.get("summary")},
            "changes": changes,
            "unchanged_keys": sorted(all_keys - set(changes.keys())),
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_get_design(args: dict[str, Any]) -> dict[str, Any]:
    """Get full content and metadata for a versioned design."""
    try:
        store = _get_store()
        design_name = args["design_name"]
        version = args.get("version")

        if version is not None:
            result = store.get_version_number("experiment_design", design_name, version)
        else:
            result = store.get_current("experiment_design", design_name)

        if result is None:
            return {"error": f"Design '{design_name}' not found" + (f" at version {version}" if version else "")}

        meta, content = result
        return {"version_meta": meta, "design": content}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_run_design(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a versioned design by converting it to an ExperimentConfig and enqueuing."""
    config = _get_config()
    try:
        from crucible.fleet.queue import enqueue_experiments
        from crucible.runner.design import design_to_experiment_config
        from crucible.runner.fingerprint import build_run_manifest

        contract = _queue_contract_fields(config)
        manifest = build_run_manifest(config.project_root)
        store = _get_store()
        design_name = args["design_name"]

        result = store.get_current("experiment_design", design_name)
        if result is None:
            return {"error": f"Design '{design_name}' not found"}

        meta, content = result

        # Apply tier/backend overrides if provided
        if "tier" in args:
            content["base_preset"] = args["tier"]
        if "backend" in args:
            content["backend"] = args["backend"]

        exp_config = design_to_experiment_config(content, meta)

        # Enqueue the experiment
        experiment = {
            "name": exp_config["name"],
            "config": exp_config["config"],
            "tier": exp_config.get("tier", "proxy"),
            "backend": exp_config.get("backend", "torch"),
            "tags": exp_config.get("tags", []),
            "run_manifest": manifest,
            **contract,
        }
        added = enqueue_experiments(
            config.project_root / "fleet_queue.jsonl",
            [experiment],
            limit=1,
        )

        if not added:
            return {"error": "Experiment already enqueued (duplicate name+tier)."}

        run_id = added[0]["run_id"]

        # Update design status to "running" and link run_id
        content["status"] = "running"
        linked = list(content.get("linked_run_ids", []))
        linked.append(run_id)
        content["linked_run_ids"] = linked

        new_meta = store.create(
            "experiment_design", design_name, content,
            summary=f"Running as {run_id}",
            created_by=meta.get("created_by", "mcp-agent"),
            tags=meta.get("tags", []),
        )

        if config.auto_commit_versions:
            store.git_commit_version(new_meta)

        return {
            "status": "enqueued",
            "run_id": run_id,
            "version_meta": new_meta,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_link_result(args: dict[str, Any]) -> dict[str, Any]:
    """Link a completed experiment run_id back to a design."""
    try:
        from crucible.runner.design import link_result_to_design

        store = _get_store()
        design_name = args["design_name"]
        run_id = args["run_id"]

        new_meta = link_result_to_design(store, design_name, run_id)
        if new_meta is None:
            return {"error": f"Design '{design_name}' not found"}

        config = _get_config()
        if config.auto_commit_versions:
            store.git_commit_version(new_meta)

        return {"status": "linked", "version_meta": new_meta}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Note tools
# ---------------------------------------------------------------------------


def _get_note_store():
    """Lazy-load a NoteStore from project config."""
    from crucible.runner.notes import NoteStore

    config = _get_config()
    store_dir = config.project_root / config.store_dir
    return NoteStore(store_dir)


def note_add(args: dict[str, Any]) -> dict[str, Any]:
    """Attach a note to an experiment run."""
    try:
        store = _get_note_store()
        entry = store.add(
            run_id=args["run_id"],
            body=args["text"],
            stage=args.get("stage", ""),
            tags=args.get("tags", []),
            confidence=args.get("confidence"),
            created_by=args.get("created_by", "mcp-agent"),
        )
        return {"status": "added", "note": entry}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def note_get(args: dict[str, Any]) -> dict[str, Any]:
    """Get all notes for a run."""
    try:
        store = _get_note_store()
        run_id = args["run_id"]
        stage = args.get("stage", "")
        entries = store.get_for_run(run_id)
        if stage:
            entries = [e for e in entries if e.get("stage") == stage]
        return {"run_id": run_id, "notes": entries, "total": len(entries)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def note_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search notes across runs."""
    try:
        store = _get_note_store()
        entries = store.search(
            query=args.get("query", ""),
            tags=args.get("tags"),
            stage=args.get("stage", ""),
            run_id=args.get("run_id", ""),
            limit=args.get("limit", 50),
        )
        return {"notes": entries, "total": len(entries)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# W&B tools
# ---------------------------------------------------------------------------


def wandb_log_image(args: dict[str, Any]) -> dict[str, Any]:
    """Log an image file to a W&B run."""
    config = _get_config()
    run_id = args["run_id"]
    image_path = args["image_path"]
    caption = args.get("caption", "")
    key = args.get("key", "image")

    try:
        from crucible.runner.wandb_logger import _resolve_wandb_url

        wandb_url = _resolve_wandb_url(run_id, config)
        if not wandb_url:
            return {"error": f"No W&B URL found for run {run_id}"}

        try:
            import wandb  # type: ignore
        except ImportError:
            return {"error": "wandb not installed"}

        # Parse entity/project/run_id from URL
        parts = wandb_url.rstrip("/").split("/")
        runs_idx = parts.index("runs")
        wb_run_id = parts[runs_idx + 1]
        project = parts[runs_idx - 1]
        entity = parts[runs_idx - 2]

        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{wb_run_id}")
        run.upload_file(str(image_path))
        return {"status": "uploaded", "run_id": run_id, "wandb_url": wandb_url, "image_path": image_path}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def wandb_get_url(args: dict[str, Any]) -> dict[str, Any]:
    """Get W&B dashboard URL for a Crucible run."""
    config = _get_config()
    run_id = args["run_id"]
    try:
        from crucible.runner.wandb_logger import _resolve_wandb_url

        url = _resolve_wandb_url(run_id, config)
        if url:
            return {"run_id": run_id, "wandb_url": url}
        return {"run_id": run_id, "wandb_url": None, "reason": "No W&B URL found in status sidecar or results."}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def wandb_annotate(args: dict[str, Any]) -> dict[str, Any]:
    """Push note/finding to W&B run summary."""
    config = _get_config()
    run_id = args["run_id"]
    text = args["text"]
    annotation_type = args.get("annotation_type", "note")

    try:
        from crucible.runner.wandb_logger import _resolve_wandb_url, wandb_annotate_finished_run

        wandb_url = _resolve_wandb_url(run_id, config)
        if not wandb_url:
            return {"error": f"No W&B URL found for run {run_id}"}

        if annotation_type == "finding":
            ok = wandb_annotate_finished_run(wandb_url, findings=[text])
        else:
            ok = wandb_annotate_finished_run(wandb_url, notes=[text])

        if ok:
            return {"status": "annotated", "run_id": run_id, "annotation_type": annotation_type, "wandb_url": wandb_url}
        return {"error": f"Failed to annotate W&B run for {run_id}. wandb may not be installed or the run may be inaccessible."}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Hub tools
# ---------------------------------------------------------------------------


def _get_hub():
    """Lazy-load a HubStore, returning None if not initialized."""
    from crucible.core.hub import HubStore

    hub = HubStore()
    if not hub.initialized:
        return None
    return hub


def hub_status(args: dict[str, Any]) -> dict[str, Any]:
    """Hub info, active track, linked projects."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"initialized": False, "message": "Hub not initialized. Run hub init first."}

        active_track = hub.get_active_track()
        projects = hub.list_projects()
        tracks = hub.list_tracks()

        return {
            "initialized": True,
            "hub_dir": str(hub.hub_dir),
            "active_track": active_track,
            "projects": projects,
            "tracks_count": len(tracks),
            "tracks": [
                {"name": t.get("name"), "description": t.get("description", ""), "active": t.get("active", True)}
                for t in tracks
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def hub_sync(args: dict[str, Any]) -> dict[str, Any]:
    """Git sync the hub (push/pull/both)."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        remote = args.get("remote")
        result = hub.sync(remote=remote)
        return {"status": "synced", **result}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def track_create(args: dict[str, Any]) -> dict[str, Any]:
    """Create a new research track."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        name = args["name"]
        description = args.get("description", "")
        tags = args.get("tags", [])

        track = hub.create_track(name, description=description, tags=tags)
        return {"status": "created", "track": track}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def track_list(args: dict[str, Any]) -> dict[str, Any]:
    """List all research tracks."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        tracks = hub.list_tracks()
        active = hub.get_active_track()
        return {
            "active_track": active,
            "tracks": tracks,
            "total": len(tracks),
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def track_switch(args: dict[str, Any]) -> dict[str, Any]:
    """Switch the active research track."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        name = args["name"]
        hub.activate_track(name)
        return {"status": "switched", "active_track": name}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def hub_findings_query(args: dict[str, Any]) -> dict[str, Any]:
    """Query findings across hub scopes."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        scope = args.get("scope", "global")
        track = args.get("track")
        status = args.get("status")
        tags = args.get("tags")
        limit = args.get("limit", 50)

        findings = hub.list_findings(scope, track=track, status=status, tags=tags)
        return {"findings": findings[:limit], "total": len(findings)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def _research_finding_to_hub_finding(finding: dict[str, Any], config: Any) -> dict[str, Any]:
    """Convert a ResearchState finding to hub-compatible Finding format."""
    from crucible.core.finding import make_finding_id

    return {
        "id": make_finding_id(finding.get("finding", "untitled")[:40], "project"),
        "title": finding.get("finding", "")[:80],
        "body": finding.get("finding", ""),
        "scope": "project",
        "status": "active",
        "confidence": finding.get("confidence", 0.5),
        "tags": [],
        "category": finding.get("category", "observation"),
        "source_project": config.name,
        "source_experiments": finding.get("source_experiments", []),
        "created_by": finding.get("created_by", "unknown"),
        "created_at": finding.get("ts", ""),
    }


def finding_promote(args: dict[str, Any]) -> dict[str, Any]:
    """Promote a finding from one scope to another. Supports project→track→global."""
    config = _get_config()
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        from_scope = args["from_scope"]
        to_scope = args["to_scope"]
        from_track = args.get("from_track")
        to_track = args.get("to_track")

        # Confidence threshold check
        from crucible.core.finding import PROMOTION_RULES

        rule = PROMOTION_RULES.get((from_scope, to_scope), {})
        min_conf = rule.get("min_confidence", 0.0)

        if from_scope == "project":
            # Promote from ResearchState → Hub
            finding_index = args.get("finding_index")
            if finding_index is None:
                finding_id = args.get("finding_id", "")
                return {"error": "finding_index is required when promoting from project scope."}

            from crucible.researcher.state import ResearchState

            state_path = config.project_root / config.research_state_file
            state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)

            if finding_index < 0 or finding_index >= len(state.findings):
                return {"error": f"Finding index {finding_index} out of range ({len(state.findings)} findings)."}

            finding = state.findings[finding_index]
            if finding.get("confidence", 0) < min_conf:
                return {"error": f"Confidence {finding.get('confidence', 0):.2f} below threshold {min_conf:.2f} for {from_scope}→{to_scope}."}

            hub_finding = _research_finding_to_hub_finding(finding, config)
            promoted = hub.store_finding(hub_finding, to_scope, track=to_track)
            return {"status": "promoted", "finding": promoted}
        else:
            # Hub-to-hub promotion (track→global)
            finding_id = args["finding_id"]
            existing = hub.get_finding(finding_id, from_scope, track=from_track)
            if existing is None:
                return {"error": f"Finding '{finding_id}' not found in {from_scope} scope."}

            if existing.get("confidence", 0) < min_conf:
                return {"error": f"Confidence {existing.get('confidence', 0):.2f} below threshold {min_conf:.2f} for {from_scope}→{to_scope}."}

            promoted = hub.promote_finding(
                finding_id, from_scope, to_scope,
                from_track=from_track, to_track=to_track,
            )
            return {"status": "promoted", "finding": promoted}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Briefing tools
# ---------------------------------------------------------------------------


def get_research_briefing(args: dict[str, Any]) -> dict[str, Any]:
    """Comprehensive session orientation: project state, leaderboard, hypotheses, findings, notes, and suggested next steps."""
    config = _get_config()
    try:
        from crucible.researcher.briefing import build_briefing

        track = args.get("track")
        if track:
            original = config.active_track
            config.active_track = track
            try:
                return build_briefing(config)
            finally:
                config.active_track = original
        return build_briefing(config)
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def annotate_run(args: dict[str, Any]) -> dict[str, Any]:
    """Bidirectional link: attach a finding to a run and record the run in the finding's source_experiments."""
    config = _get_config()
    run_id = args["run_id"]
    finding_index = args["finding_index"]

    try:
        from crucible.researcher.state import ResearchState
        from crucible.runner.notes import NoteStore

        state_path = config.project_root / config.research_state_file
        if not state_path.exists():
            return {"error": "No research state found. Record findings first with context_push_finding."}

        state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)

        if finding_index < 0 or finding_index >= len(state.findings):
            return {
                "error": f"Finding index {finding_index} out of range. "
                f"There are {len(state.findings)} findings (0-indexed)."
            }

        finding = state.findings[finding_index]

        # Add run_id to the finding's source_experiments
        source_exps = finding.get("source_experiments", [])
        if run_id not in source_exps:
            source_exps.append(run_id)
            finding["source_experiments"] = source_exps
            state.save()

        # Add a note to the run referencing this finding
        store_dir = config.project_root / config.store_dir
        note_store = NoteStore(store_dir)
        finding_text = finding.get("finding", "")
        category = finding.get("category", "observation")
        note_body = (
            f"Linked to finding [{category}]: {finding_text}\n\n"
            f"(finding_index={finding_index}, confidence={finding.get('confidence', '?')})"
        )
        note_entry = note_store.add(
            run_id=run_id,
            body=note_body,
            stage="post-run",
            tags=["annotate_run", category],
            created_by="mcp-agent",
        )

        return {
            "status": "annotated",
            "run_id": run_id,
            "finding_index": finding_index,
            "finding_text": finding_text[:200],
            "note": note_entry,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Literature search tools
# ---------------------------------------------------------------------------


def research_literature_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search AI research papers on HuggingFace."""
    from crucible.researcher.literature import (
        format_literature_context,
        search_papers,
        suggest_queries,
    )

    query = args.get("query", "")
    auto = args.get("auto", False)
    limit = args.get("limit", 10)

    if auto and not query:
        # Auto-generate queries from research state
        config = _get_config()
        beliefs: list[str] = []
        findings: list[dict[str, Any]] = []
        program_text = ""
        try:
            from crucible.researcher.state import ResearchState

            state_path = config.project_root / config.research_state_file
            if state_path.exists():
                state = ResearchState(state_path, budget_hours=0)
                beliefs = state.beliefs
                findings = state.findings
        except Exception:
            pass
        try:
            prog_path = config.project_root / config.researcher.program_file
            if prog_path.exists():
                program_text = prog_path.read_text(encoding="utf-8")
        except Exception:
            pass
        queries = suggest_queries(program_text, beliefs, findings)
        all_papers: list[dict[str, Any]] = []
        seen: set[str] = set()
        for q in queries:
            for p in search_papers(q, limit=5):
                if p["id"] not in seen:
                    seen.add(p["id"])
                    all_papers.append(p)
        papers = all_papers[:limit]
        query_used: Any = queries
    else:
        papers = search_papers(query, limit=limit)
        query_used = query

    return {
        "papers": papers,
        "query_used": query_used,
        "literature_context": format_literature_context(papers),
        "count": len(papers),
    }


# ---------------------------------------------------------------------------
# Model extensibility tools
# ---------------------------------------------------------------------------


def model_list_families(args: dict[str, Any]) -> dict[str, Any]:
    """List all registered model architecture families."""
    try:
        detailed = args.get("detailed", False)
        if detailed:
            from crucible.models.registry import list_families_detailed
            families = list_families_detailed()
            # Enrich with kind: "spec" if a .yaml spec exists, else "code"
            config = _get_config()
            specs_dir = config.project_root / "src" / "crucible" / "models" / "specs"
            arch_dir = config.project_root / config.store_dir / "architectures"
            global_meta: dict[str, dict[str, Any]] = {}
            try:
                hub = _get_hub_store()
                global_meta = {entry["name"]: entry for entry in hub.list_architectures()}
            except Exception:
                global_meta = {}
            for entry in families:
                name = entry["name"]
                if entry.get("source") == "global" and name in global_meta:
                    entry["kind"] = global_meta[name].get("kind", "code")
                    continue
                has_spec = (specs_dir / f"{name}.yaml").exists() or (arch_dir / f"{name}.yaml").exists()
                entry["kind"] = "spec" if has_spec else "code"
            return {"families": families}
        from crucible.models.registry import list_families
        return {"families": list_families()}
    except ImportError:
        # torch not installed — return known built-in families
        return {"families": ["baseline", "convloop", "looped", "memory", "prefix_memory"]}


def model_list_activations(args: dict[str, Any]) -> dict[str, Any]:
    """List all available activation functions."""
    try:
        from crucible.models.components.mlp import ACTIVATIONS
        return {"activations": sorted(ACTIVATIONS.keys())}
    except ImportError:
        return {"activations": [
            "elu03_sq", "gelu_sq", "leaky01_sq", "leaky02_sq", "leaky08_sq",
            "log1p_relu_sq", "mish_sq", "relu_sq", "x_absx",
        ]}


def model_list_components(args: dict[str, Any]) -> dict[str, Any]:
    """List all available model components."""
    try:
        from crucible.models import components
        return {"components": components.__all__}
    except ImportError:
        return {"components": [
            "RMSNorm", "CastedLinear", "Rotary", "CausalSelfAttention", "Block",
            "MLP", "SmearGate", "BigramHash", "TrigramHash", "TokenMerger",
            "BatchedLinearLoRA", "BatchedTTTLoRA", "MoELayer",
        ]}


def model_get_config_schema(args: dict[str, Any]) -> dict[str, Any]:
    """Get accepted parameters for a model family."""
    family = args["family"]
    try:
        from crucible.models.registry import get_family_schema, list_families
        if family not in list_families():
            return {"error": f"Unknown family: {family}. Available: {list_families()}"}
        return {"family": family, "parameters": get_family_schema(family)}
    except Exception as exc:
        return {"error": f"Failed to get schema: {exc}"}


def model_validate_config(args: dict[str, Any]) -> dict[str, Any]:
    """Pre-flight validation of experiment config."""
    config = args.get("config", {})
    warnings: list[str] = []
    errors: list[str] = []
    family = config.get("MODEL_FAMILY", "baseline")
    try:
        from crucible.models.registry import list_families
        if family not in list_families():
            errors.append(f"Unknown MODEL_FAMILY: {family}")
    except Exception as exc:
        warnings.append(f"Could not validate MODEL_FAMILY: {exc}")
    activation = config.get("ACTIVATION", "relu_sq")
    try:
        from crucible.models.components.mlp import ACTIVATIONS
        if activation not in ACTIVATIONS:
            errors.append(f"Unknown ACTIVATION: {activation}. Available: {sorted(ACTIVATIONS.keys())}")
    except ImportError:
        warnings.append("torch not installed; cannot validate ACTIVATION")
    # Validate spec-based families: check that template variables have values or defaults
    try:
        spec_dict = _load_spec_dict(family)
        if spec_dict is not None:
            import re
            var_pattern = re.compile(r"\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")
            missing_vars: list[str] = []
            _scan_spec_vars(spec_dict, var_pattern, config, missing_vars)
            if missing_vars:
                warnings.append(
                    f"Spec-based family {family!r} has template variables without defaults "
                    f"or config values: {missing_vars}"
                )
    except Exception:
        pass  # Don't fail validation for spec introspection errors
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def model_add_architecture(args: dict[str, Any]) -> dict[str, Any]:
    """Write and register a new architecture family at runtime."""
    name = args["name"]
    code = args["code"]
    scope = args.get("scope", "local")
    if "register_model" not in code:
        return {"error": (
            "Code must call register_model() to register the family. Example:\n\n"
            "from crucible.models.registry import register_model\n\n"
            "def _build(args):\n"
            "    return MyArchitecture(args)\n\n"
            f"register_model('{name}', _build)\n\n"
            "Use model_generate_template() for full boilerplate."
        )}

    if scope == "global":
        try:
            hub = _get_hub_store()
            result = hub.store_architecture(
                name=name,
                code=code,
                source_project=_get_config().name if _get_config else "",
            )
            # Also import into current process
            try:
                from crucible.models.registry import load_global_architectures
                load_global_architectures(hub._arch_plugins_dir)
            except Exception:
                pass
            from crucible.models.registry import list_families
            return {"status": "registered", "scope": "global", "family": name, "families": list_families()}
        except Exception as exc:
            return {"error": f"Failed to store global architecture: {exc}"}

    import importlib.util
    config = _get_config()
    arch_dir = config.project_root / config.store_dir / "architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)
    file_path = arch_dir / f"{name}.py"
    file_path.write_text(code, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(f"_crucible_local_arch_{name}", file_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        from crucible.models.registry import list_families
        return {"status": "registered", "scope": "local", "family": name, "families": list_families()}
    except Exception as e:
        file_path.unlink(missing_ok=True)
        return {"error": f"Failed to import: {e}"}


def model_add_activation(args: dict[str, Any]) -> dict[str, Any]:
    """Register a new activation function at runtime via restricted code expression."""
    name = args["name"]
    code = args["code"]
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812
    except ImportError:
        return {"error": "torch not installed"}
    try:
        # Restricted namespace: only torch and F, no builtins for safety
        restricted_ns: dict[str, Any] = {"torch": torch, "F": F, "__builtins__": {}}
        compiled = compile(f"__result = lambda x: {code}", "<activation>", "exec")
        exec(compiled, restricted_ns)  # noqa: S102 — intentional: sandboxed activation builder
        activation_fn = restricted_ns["__result"]
        test = torch.randn(2, 3)
        result = activation_fn(test)
        assert result.shape == test.shape
    except Exception as e:
        return {"error": f"Invalid activation code: {e}"}
    from crucible.models.components.mlp import ACTIVATIONS
    ACTIVATIONS[name] = activation_fn
    return {"status": "registered", "name": name, "activations": sorted(ACTIVATIONS.keys())}


def model_generate_template(args: dict[str, Any]) -> dict[str, Any]:
    """Generate boilerplate code for a new architecture."""
    name = args["name"]
    cn = name.title().replace("_", "")
    lines = [
        f'"""User architecture: {name}."""',
        "from __future__ import annotations",
        "from typing import Any",
        "import torch",
        "from torch import Tensor, nn",
        "from crucible.models.base import TiedEmbeddingLM",
        "from crucible.models.registry import register_model",
        "from crucible.models.components.attention import Block",
        "",
        f"class {cn}GPT(TiedEmbeddingLM):",
        "    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,",
        '                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,',
        '                 rope_base, qk_gain_init, activation="relu_sq", **kwargs):',
        "        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std, logit_softcap)",
        "        self.blocks = nn.ModuleList([",
        "            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,",
        "                  activation=activation) for _ in range(num_layers)])",
        "",
        "    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:",
        "        x = self.embed_tokens(input_ids)",
        "        x0 = x",
        "        for block in self.blocks:",
        "            x = block(x, x0)",
        "        return x",
        "",
        f"def _build_{name}(args: Any) -> {cn}GPT:",
        f"    return {cn}GPT(",
        "        vocab_size=args.vocab_size, num_layers=args.num_layers,",
        "        model_dim=args.model_dim, num_heads=args.num_heads,",
        "        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,",
        "        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,",
        "        logit_softcap=args.logit_softcap, rope_base=args.rope_base,",
        '        qk_gain_init=args.qk_gain_init, activation=getattr(args, "activation", "relu_sq"))',
        "",
        f'register_model("{name}", _build_{name})',
    ]
    return {"template": "\n".join(lines), "usage": f"Call model_add_architecture with name='{name}' and code=<this>"}


# ---------------------------------------------------------------------------
# Plugin promotion / import tools
# ---------------------------------------------------------------------------


def model_list_global_architectures(args: dict[str, Any]) -> dict[str, Any]:
    """List architecture plugins stored in the global hub."""
    try:
        hub = _get_hub_store()
        return {"architectures": hub.list_architectures()}
    except Exception as exc:
        return {"architectures": [], "note": str(exc)}


def model_promote_architecture(args: dict[str, Any]) -> dict[str, Any]:
    """Promote a project-local architecture plugin to the global hub."""
    name = args["name"]
    config = _get_config()
    plugin_path = config.project_root / config.store_dir / "architectures" / f"{name}.py"
    if not plugin_path.exists():
        return {"error": f"Local plugin {name!r} not found at {plugin_path}"}
    code = plugin_path.read_text(encoding="utf-8")
    try:
        hub = _get_hub_store()
        result = hub.store_architecture(name=name, code=code, source_project=config.name)
        return {"status": "promoted", "family": name, "metadata": result}
    except Exception as exc:
        return {"error": f"Promotion failed: {exc}"}


def model_import_architecture(args: dict[str, Any]) -> dict[str, Any]:
    """Import a global hub architecture into the project's user_architectures/ directory."""
    name = args["name"]
    try:
        hub = _get_hub_store()
        metadata = hub.get_architecture(name)
        content = hub.get_architecture_content(name)
        if metadata is None or content is None:
            return {"error": f"Architecture {name!r} not found in hub"}
        config = _get_config()
        suffix = ".py" if metadata.get("kind", "code") == "code" else ".yaml"
        target = config.project_root / config.store_dir / "architectures" / f"{name}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        # Import into current process
        try:
            import importlib.util
            if suffix == ".py":
                spec = importlib.util.spec_from_file_location(f"user_arch_{name}", target)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
            else:
                from crucible.models.composer import register_from_spec

                register_from_spec(name, target, source="local")
        except Exception:
            pass
        from crucible.models.registry import list_families
        return {"status": "imported", "family": name, "path": str(target), "families": list_families()}
    except Exception as exc:
        return {"error": f"Import failed: {exc}"}


# ---------------------------------------------------------------------------
# Composition tools — helpers
# ---------------------------------------------------------------------------


def _load_spec_dict(family: str) -> dict | None:
    """Try to load a YAML spec dict for *family* from specs/ or .crucible/architectures/.

    Returns the parsed dict or None if no spec file exists.
    """
    import yaml

    config = _get_config()
    specs_dir = config.project_root / "src" / "crucible" / "models" / "specs"
    arch_dir = config.project_root / config.store_dir / "architectures"
    for directory in (specs_dir, arch_dir):
        path = directory / f"{family}.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    return None


def _scan_spec_vars(
    obj: Any,
    var_pattern: Any,
    config: dict,
    missing: list[str],
) -> None:
    """Recursively scan a spec dict for unresolved template vars without defaults."""
    import re

    if isinstance(obj, str):
        for m in var_pattern.finditer(obj):
            var_name, default = m.group(1), m.group(2)
            if default is None and var_name not in config:
                if var_name not in missing:
                    missing.append(var_name)
    elif isinstance(obj, dict):
        for v in obj.values():
            _scan_spec_vars(v, var_pattern, config, missing)
    elif isinstance(obj, list):
        for item in obj:
            _scan_spec_vars(item, var_pattern, config, missing)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a copy of *base*."""
    import copy
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Composition tools
# ---------------------------------------------------------------------------


def model_compose(args: dict[str, Any]) -> dict[str, Any]:
    """Create architecture from declarative YAML spec. No Python code written."""
    import yaml
    from pathlib import Path as _Path

    name = args.get("name")
    spec = args.get("spec")
    scope = args.get("scope", "local")

    if not name:
        return {"error": "name is required"}
    if not spec or not isinstance(spec, dict):
        return {"error": "spec must be a non-empty dict with block, stack, etc."}

    # Validate name is a valid Python identifier
    if not name.isidentifier():
        return {"error": f"name must be a valid Python identifier, got: {name!r}"}

    # Build full spec dict
    spec_dict = {
        "name": name,
        "version": spec.get("version", 1),
        "base": spec.get("base", "tied_embedding_lm"),
        "embedding": spec.get("embedding", {}),
        "block": spec.get("block", {}),
        "stack": spec.get("stack", {}),
    }
    if "transform" in spec:
        spec_dict["transform"] = spec["transform"]
    if "init" in spec:
        spec_dict["init"] = spec["init"]
    if "augmentations" in spec:
        spec_dict["augmentations"] = spec["augmentations"]

    # Validate using ArchitectureSpec.from_dict
    try:
        from crucible.models.composer import ArchitectureSpec
        ArchitectureSpec.from_dict(spec_dict)
    except Exception as exc:
        return {"error": f"Spec validation failed: {exc}"}

    # Determine save path
    config = _get_config()
    if scope == "global":
        try:
            hub = _get_hub_store()
        except Exception as exc:
            return {"error": f"Hub not available for global scope: {exc}"}
    else:
        arch_dir = config.project_root / config.store_dir / "architectures"

    yaml_text = yaml.safe_dump(spec_dict, default_flow_style=False, sort_keys=False)
    if scope == "global":
        try:
            result = hub.store_architecture(
                name=name,
                code=yaml_text,
                source_project=config.name,
                kind="spec",
            )
            path = hub.hub_dir / result["relative_path"]
        except Exception as exc:
            return {"error": f"Failed to store global architecture spec: {exc}"}
    else:
        arch_dir.mkdir(parents=True, exist_ok=True)
        path = arch_dir / f"{name}.yaml"
        with open(path, "w") as f:
            f.write(yaml_text)

    # Register using register_from_spec
    try:
        from crucible.models.composer import register_from_spec
        register_from_spec(name, spec_dict, source="local" if scope == "local" else "global")
    except Exception as exc:
        return {"error": f"Registration failed (spec saved at {path}): {exc}"}

    return {"registered": name, "scope": scope, "path": str(path)}


def model_from_template(args: dict[str, Any]) -> dict[str, Any]:
    """Fork existing spec with overrides to create a new family."""
    import yaml

    name = args.get("name")
    base = args.get("base")
    overrides = args.get("overrides", {})

    if not name:
        return {"error": "name is required"}
    if not base:
        return {"error": "base family name is required"}
    if not name.isidentifier():
        return {"error": f"name must be a valid Python identifier, got: {name!r}"}

    # Load existing spec
    base_spec = _load_spec_dict(base)
    if base_spec is None:
        return {"error": f"No YAML spec found for base family {base!r}. Only spec-based families can be forked."}

    # Deep-merge overrides
    merged = _deep_merge(base_spec, overrides)
    merged["name"] = name

    # Validate
    try:
        from crucible.models.composer import ArchitectureSpec
        ArchitectureSpec.from_dict(merged)
    except Exception as exc:
        return {"error": f"Merged spec validation failed: {exc}"}

    # Save
    config = _get_config()
    arch_dir = config.project_root / config.store_dir / "architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)
    path = arch_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(merged, f, default_flow_style=False, sort_keys=False)

    # Register
    try:
        from crucible.models.composer import register_from_spec
        register_from_spec(name, merged, source="local")
    except Exception as exc:
        return {"error": f"Registration failed (spec saved at {path}): {exc}"}

    return {"registered": name, "base": base, "path": str(path), "spec": merged}


def model_list_stack_patterns(args: dict[str, Any]) -> dict[str, Any]:
    """List available stack wiring patterns."""
    try:
        from crucible.models.composer import STACK_PATTERNS
        patterns = []
        _descriptions = {
            "sequential": "Simple linear pass through all blocks.",
            "encoder_decoder_skip": "Encoder-decoder with learned skip connections (baseline architecture).",
            "looped": "Looped iteration over shared blocks with step scales.",
            "prefix_memory_stack": "Sequential with step scales for PrefixMemoryBlock layers.",
        }
        _params = {
            "sequential": [],
            "encoder_decoder_skip": ["num_layers"],
            "looped": ["logical_steps", "unique_blocks"],
            "prefix_memory_stack": ["num_layers"],
        }
        for name in sorted(STACK_PATTERNS):
            patterns.append({
                "name": name,
                "description": _descriptions.get(name, ""),
                "params": _params.get(name, []),
            })
        return {"patterns": patterns}
    except ImportError:
        return {"patterns": [
            {"name": "sequential", "description": "Simple linear pass through all blocks.", "params": []},
            {"name": "encoder_decoder_skip", "description": "Encoder-decoder with learned skip connections.", "params": ["num_layers"]},
            {"name": "looped", "description": "Looped iteration over shared blocks with step scales.", "params": ["logical_steps", "unique_blocks"]},
            {"name": "prefix_memory_stack", "description": "Sequential with step scales for PrefixMemoryBlock layers.", "params": ["num_layers"]},
        ]}


def model_list_block_types(args: dict[str, Any]) -> dict[str, Any]:
    """List available block types for architecture composition."""
    try:
        from crucible.models.composer import BLOCK_TYPES, _ensure_block_types
        _ensure_block_types()
        _descriptions = {
            "attention_block": "Standard transformer block with multi-head attention, MLP, and RMSNorm.",
            "prefix_memory_block": "Prefix memory block with state compression for recurrent-style architectures.",
        }
        block_types = []
        for name in sorted(BLOCK_TYPES):
            block_types.append({
                "name": name,
                "description": _descriptions.get(name, ""),
            })
        return {"block_types": block_types}
    except ImportError:
        return {"block_types": [
            {"name": "attention_block", "description": "Standard transformer block with multi-head attention, MLP, and RMSNorm."},
            {"name": "prefix_memory_block", "description": "Prefix memory block with state compression for recurrent-style architectures."},
        ]}


def model_preview_spec(args: dict[str, Any]) -> dict[str, Any]:
    """Dry-run a spec: instantiate on CPU and return param count + structure."""
    spec = args.get("spec")
    config_overrides = args.get("config", {})

    if not spec or not isinstance(spec, dict):
        return {"error": "spec must be a non-empty dict"}

    try:
        from crucible.models.composer import ArchitectureSpec, SpecResolver, ComposedArchitecture
        import types

        parsed = ArchitectureSpec.from_dict(spec)

        # Build an args namespace from config overrides with sensible defaults
        defaults = {
            "vocab_size": 50304, "model_dim": 512, "num_layers": 9,
            "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 2,
            "rope_base": 10000.0, "qk_gain_init": 1.0,
            "attention_variant": "standard", "residual_variant": "standard",
            "tie_embeddings": True, "tied_embed_init_std": 0.02,
            "logit_softcap": 30.0, "embed_bottleneck_dim": 0,
            "spectral_embed_init": False, "activation": "relu_sq",
            "model_family": spec.get("name", "preview"),
        }
        # Apply config overrides (convert string values to appropriate types)
        for k, v in config_overrides.items():
            key = k.lower()
            if isinstance(v, str):
                if v.lower() == "true":
                    defaults[key] = True
                elif v.lower() == "false":
                    defaults[key] = False
                else:
                    try:
                        defaults[key] = int(v)
                    except ValueError:
                        try:
                            defaults[key] = float(v)
                        except ValueError:
                            defaults[key] = v
            else:
                defaults[key] = v
        build_args = types.SimpleNamespace(**defaults)

        resolver = SpecResolver(parsed, build_args)
        resolved = resolver.resolve()
        model = ComposedArchitecture(resolved, build_args)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Collect layer structure
        layers = []
        for name, module in model.named_children():
            param_count = sum(p.numel() for p in module.parameters())
            layers.append({"name": name, "type": type(module).__name__, "params": param_count})

        return {
            "name": spec.get("name", "preview"),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "layers": layers,
            "config_used": {k: v for k, v in defaults.items() if k != "model_family"},
        }
    except Exception as exc:
        return {"error": f"Preview failed: {exc}"}


def model_get_spec(args: dict[str, Any]) -> dict[str, Any]:
    """Get YAML spec dict for a family, or null if code-defined."""
    family = args.get("family")
    if not family:
        return {"error": "family is required"}

    spec_dict = _load_spec_dict(family)
    if spec_dict is not None:
        return {"family": family, "kind": "spec", "spec": spec_dict}
    return {"family": family, "kind": "code", "spec": None}


# ---------------------------------------------------------------------------
# Config tools
# ---------------------------------------------------------------------------


def config_get_presets(args: dict[str, Any]) -> dict[str, Any]:
    """All presets with resolved config values."""
    config = _get_config()
    try:
        from crucible.runner.presets import get_preset, list_presets

        preset_name = args.get("preset_name", "")
        if preset_name:
            resolved = get_preset(preset_name, project_config=config)
            return {"preset": preset_name, "config": resolved}
        else:
            all_presets = {}
            for name in list_presets(project_config=config):
                all_presets[name] = get_preset(name, project_config=config)
            return {"presets": all_presets}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def config_get_project(args: dict[str, Any]) -> dict[str, Any]:
    """Get the full configuration of a project spec."""
    config = _get_config()
    try:
        from crucible.core.config import load_project_spec
        project_name = args["project_name"]
        spec = load_project_spec(project_name, config.project_root)
        result: dict[str, Any] = {
            "name": spec.name,
            "repo": getattr(spec, "repo", ""),
            "branch": getattr(spec, "branch", ""),
            "workspace": getattr(spec, "workspace", ""),
            "launcher": getattr(spec, "launcher", ""),
            "launcher_entry": getattr(spec, "launcher_entry", ""),
        }
        if spec.pod:
            result["pod"] = {
                "gpu_type": getattr(spec.pod, "gpu_type", ""),
                "container_disk": getattr(spec.pod, "container_disk", 0),
                "volume_disk": getattr(spec.pod, "volume_disk", 0),
                "interruptible": getattr(spec.pod, "interruptible", False),
                "image": getattr(spec.pod, "image", ""),
            }
        result["env_set"] = dict(getattr(spec, "env_set", {}) or {})
        result["env_forward"] = list(getattr(spec, "env_forward", []) or [])
        if spec.metrics:
            result["metrics"] = {
                "source": getattr(spec.metrics, "source", ""),
                "primary": getattr(spec.metrics, "primary", ""),
                "direction": getattr(spec.metrics, "direction", ""),
            }
        result["install"] = list(getattr(spec, "install", []) or [])
        result["timeout"] = getattr(spec, "timeout", 0)
        return result
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Queue management tools
# ---------------------------------------------------------------------------


def cancel_experiment(args: dict[str, Any]) -> dict[str, Any]:
    """Cancel queued or running experiments by name, run_id, or wave."""
    from crucible.fleet.queue import load_queue, save_queue

    config = _get_config()
    queue_path = config.project_root / "fleet_queue.jsonl"
    queue = load_queue(queue_path)

    run_id = args.get("run_id")
    experiment_name = args.get("experiment_name")
    wave = args.get("wave")

    if not run_id and not experiment_name and not wave:
        return {"error": "Provide at least one of: run_id, experiment_name, wave"}

    cancelled = []
    for entry in queue:
        if entry.get("lease_state") in ("completed", "finished", "failed"):
            continue
        match = False
        if run_id and entry.get("run_id") == run_id:
            match = True
        if experiment_name and entry.get("experiment_name") == experiment_name:
            match = True
        if wave and entry.get("wave") == wave:
            match = True
        if match:
            entry["lease_state"] = "failed"
            entry["result_status"] = "cancelled"
            entry["ended_at"] = utc_now_iso()
            cancelled.append(entry["experiment_name"])

    save_queue(queue_path, queue)
    return {"cancelled": cancelled, "count": len(cancelled)}


def clear_stale_queue(args: dict[str, Any]) -> dict[str, Any]:
    """Mark experiments as failed if assigned to nodes that no longer exist."""
    from crucible.fleet.queue import load_queue, save_queue
    from crucible.fleet.inventory import load_nodes

    config = _get_config()
    queue_path = config.project_root / "fleet_queue.jsonl"
    queue = load_queue(queue_path)
    nodes = load_nodes(config.project_root / config.nodes_file)
    live_names = {n.get("name") for n in nodes}

    cleared = []
    for entry in queue:
        if entry.get("lease_state") != "running":
            continue
        assigned = entry.get("assigned_node")
        if assigned and assigned not in live_names:
            entry["lease_state"] = "failed"
            entry["result_status"] = "stale_node"
            entry["ended_at"] = utc_now_iso()
            cleared.append({"experiment": entry["experiment_name"], "stale_node": assigned})

    save_queue(queue_path, queue)
    return {"cleared": cleared, "count": len(cleared)}


def purge_queue(args: dict[str, Any]) -> dict[str, Any]:
    """Remove all completed/failed/finished items from the fleet queue.

    REQUIRES: Nothing.
    RETURNS: {removed: int, remaining: int}
    NEXT: get_queue_status to verify, enqueue_experiment or dispatch_experiments.
    """
    from crucible.fleet.queue import load_queue, purge_finished

    config = _get_config()
    queue_path = config.project_root / "fleet_queue.jsonl"
    before = len(load_queue(queue_path))
    removed = purge_finished(queue_path)
    return {"removed": removed, "remaining": before - removed}


# ---------------------------------------------------------------------------
# Run logs tool
# ---------------------------------------------------------------------------


def get_run_logs(args: dict[str, Any]) -> dict[str, Any]:
    """Fetch stdout/stderr logs for an experiment run."""
    config = _get_config()
    run_id = args["run_id"]
    tail_lines = args.get("tail_lines", 100)

    # Step 1: Check locally collected logs
    fleet_runs_dir = config.project_root / "fleet_runs"
    local_logs: list[Path] = []
    if fleet_runs_dir.exists():
        local_logs = sorted(fleet_runs_dir.glob(f"*/logs/{run_id}*.txt"))
    # Also check project-level logs/
    project_logs_dir = config.project_root / "logs"
    if project_logs_dir.exists():
        local_logs.extend(sorted(project_logs_dir.glob(f"{run_id}*.txt")))

    if local_logs:
        combined = []
        for lf in local_logs:
            try:
                text = lf.read_text(encoding="utf-8", errors="replace")
                combined.append(f"--- {lf.name} ---\n{text}")
            except Exception:
                combined.append(f"--- {lf.name} --- (read error)")
        full_text = "\n".join(combined)
        lines = full_text.splitlines()
        total = len(lines)
        if tail_lines > 0 and total > tail_lines:
            lines = lines[-tail_lines:]
        return {
            "found": True,
            "run_id": run_id,
            "source": "local",
            "log_text": "\n".join(lines),
            "lines_returned": len(lines),
            "total_lines": total,
            "log_files": [str(lf) for lf in local_logs],
        }

    # Step 2: Try SSH to remote pod
    try:
        from crucible.fleet.inventory import load_nodes_if_exists
        from crucible.fleet.queue import load_queue
        from crucible.fleet.sync import remote_exec

        queue_path = config.project_root / "fleet_queue.jsonl"
        queue = load_queue(queue_path) if queue_path.exists() else []
        assigned_node = None
        for row in queue:
            if row.get("run_id") == run_id:
                assigned_node = row.get("assigned_node") or row.get("assigned_pod")
                break
        if not assigned_node:
            return {"found": False, "run_id": run_id, "reason": "No local logs and run_id not found in queue."}

        nodes = load_nodes_if_exists(config.project_root / config.nodes_file)
        node = next((n for n in nodes if n.get("name") == assigned_node), None)
        if not node or not node.get("ssh_host"):
            return {"found": False, "run_id": run_id, "reason": f"Node '{assigned_node}' not reachable (no SSH host). Try collect_results first."}

        workspace = node.get("workspace_path", "/workspace/project")
        tail_flag = f"-n {tail_lines}" if tail_lines > 0 else ""
        cmd = (
            f"cat {workspace}/logs/{run_id}.launcher.txt "
            f"{workspace}/logs/{run_id}.txt 2>/dev/null"
        )
        if tail_flag:
            cmd += f" | tail {tail_flag}"
        proc = remote_exec(node, cmd, check=False)
        if proc.returncode != 0 and not proc.stdout.strip():
            return {"found": False, "run_id": run_id, "reason": f"Logs not found on node '{assigned_node}'. Experiment may not have started yet."}

        text = proc.stdout or ""
        lines = text.splitlines()
        return {
            "found": True,
            "run_id": run_id,
            "source": "remote",
            "node": assigned_node,
            "log_text": "\n".join(lines),
            "lines_returned": len(lines),
        }
    except Exception as exc:
        return {"found": False, "run_id": run_id, "reason": f"SSH probe failed: {exc}. Try collect_results to download logs locally."}


# ---------------------------------------------------------------------------
# Architecture fetch tool
# ---------------------------------------------------------------------------


def model_fetch_architecture(args: dict[str, Any]) -> dict[str, Any]:
    """Fetch full source code (Python) or spec (YAML) for a registered architecture."""
    family = args["family"]
    config = _get_config()

    # Search tiers in precedence order: local > global > builtin
    search_paths: list[tuple[str, Path]] = []

    # Local
    local_arch_dir = config.project_root / config.store_dir / "architectures"
    search_paths.append(("local", local_arch_dir / f"{family}.py"))
    search_paths.append(("local", local_arch_dir / f"{family}.yaml"))

    # Global (hub)
    try:
        hub = _get_hub_store()
        metadata = hub.get_architecture(family)
        if metadata is not None:
            relative_path = metadata.get("relative_path")
            if relative_path:
                search_paths.append(("global", hub.hub_dir / relative_path))
            elif metadata.get("kind", "code") == "spec":
                search_paths.append(("global", hub._arch_specs_dir / f"{family}.yaml"))
            else:
                search_paths.append(("global", hub._arch_plugins_dir / f"{family}.py"))
    except Exception:
        pass

    # Builtin code
    import crucible.models.architectures as arch_pkg
    builtin_arch_dir = Path(arch_pkg.__file__).parent
    search_paths.append(("builtin", builtin_arch_dir / f"{family}.py"))

    # Builtin specs
    builtin_spec_dir = builtin_arch_dir.parent / "specs"
    search_paths.append(("builtin", builtin_spec_dir / f"{family}.yaml"))

    for source, path in search_paths:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                kind = "code" if path.suffix == ".py" else "spec"
                return {
                    "family": family,
                    "kind": kind,
                    "source": source,
                    "content": content,
                    "file_path": str(path),
                }
            except Exception as exc:
                return {"error": f"Found {path} but failed to read: {exc}"}

    from crucible.models.registry import list_families
    available = list_families()
    return {"error": f"Architecture '{family}' not found. Available: {available}"}


# ---------------------------------------------------------------------------
# Architecture guide tool
# ---------------------------------------------------------------------------


def get_architecture_guide(args: dict[str, Any]) -> dict[str, Any]:
    """Decision guide for creating model architectures."""
    return {
        "decision_tree": {
            "use_declarative_composition": [
                "Standard block types (attention, prefix_memory) are sufficient",
                "Standard wiring patterns (sequential, looped, encoder_decoder_skip) fit your design",
                "No custom forward pass logic needed",
                "You want rapid iteration without writing Python",
            ],
            "use_python_plugin": [
                "You need custom forward pass logic",
                "You're implementing a novel attention mechanism or memory system",
                "You need to import external libraries",
                "Your architecture doesn't fit block -> stack -> output pattern",
            ],
        },
        "workflows": {
            "declarative_composition": {
                "description": "No code required. Compose from blocks + patterns via YAML.",
                "steps": [
                    "1. model_list_stack_patterns() -- see wiring patterns",
                    "2. model_list_block_types() -- see available blocks",
                    "3. model_compose(name, spec) -- create and register",
                    "4. model_preview_spec(spec) -- validate on CPU (param count, layers)",
                    "5. enqueue_experiment(config={MODEL_FAMILY: name}) -- run it",
                ],
            },
            "python_plugin": {
                "description": "Full control. Write a Python module with custom forward logic.",
                "steps": [
                    "1. model_fetch_architecture(family) -- read an existing arch for reference",
                    "2. model_generate_template(name) -- get boilerplate code",
                    "3. Edit the code: implement __init__ + hidden() methods",
                    "4. model_add_architecture(name, code) -- register plugin",
                    "5. model_validate_config(config) -- pre-flight check",
                    "6. enqueue_experiment(config={MODEL_FAMILY: name}) -- run it",
                ],
            },
        },
        "tips": [
            "Start with declarative composition -- switch to Python only if needed",
            "Use model_from_template(name, base, overrides) to fork an existing spec",
            "model_preview_spec lets you check param count without GPU",
            "model_fetch_architecture returns source code for any registered family",
        ],
    }


# ---------------------------------------------------------------------------
# Tree search tools
# ---------------------------------------------------------------------------


def _get_tree_dir(config, name: str):
    """Resolve the directory for a named search tree."""
    store_dir = config.project_root / config.store_dir
    return store_dir / "search_trees" / name


def tree_create(args: dict[str, Any]) -> dict[str, Any]:
    """Create a new search tree over experiments."""
    config = _get_config()
    try:
        from crucible.researcher.search_tree import SearchTree

        name = args["name"]
        tree_dir = _get_tree_dir(config, name)
        roots = args.get("roots", [])
        tree = SearchTree.create(
            tree_dir=tree_dir,
            name=name,
            description=args.get("description", ""),
            roots=roots if roots else None,
            expansion_policy=args.get("expansion_policy", "agent_directed"),
            pruning_policy=args.get("pruning_policy", "agent_directed"),
            expansion_config=args.get("expansion_config", {}),
            pruning_config=args.get("pruning_config", {}),
            primary_metric=args.get("primary_metric", config.metrics.primary),
            metric_direction=args.get("metric_direction", "minimize"),
            max_depth=args.get("max_depth", 10),
            max_nodes=args.get("max_nodes", 500),
            max_expansions_per_node=args.get("max_expansions_per_node", 5),
        )
        return {
            "status": "created",
            "name": name,
            "root_node_ids": tree.meta["root_node_ids"],
            "total_nodes": tree.meta["total_nodes"],
            "tree_dir": str(tree_dir),
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def tree_get(args: dict[str, Any]) -> dict[str, Any]:
    """Get tree structure and ASCII visualization."""
    config = _get_config()
    try:
        from crucible.researcher.search_tree import SearchTree

        name = args["name"]
        tree_dir = _get_tree_dir(config, name)
        tree = SearchTree.load(tree_dir)

        summary = tree.get_tree_summary()
        ascii_tree = tree.render_ascii(max_depth=args.get("max_depth"))

        result: dict[str, Any] = {
            "summary": summary,
            "ascii_tree": ascii_tree,
        }

        best_path = tree.get_best_path()
        if best_path:
            result["best_path"] = [
                {
                    "node_id": n["node_id"],
                    "experiment_name": n["experiment_name"],
                    "depth": n["depth"],
                    "status": n["status"],
                    "result_metric": n.get("result_metric"),
                    "hypothesis": n.get("hypothesis", ""),
                }
                for n in best_path
            ]

        return result
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def tree_expand_node(args: dict[str, Any]) -> dict[str, Any]:
    """Add children to a completed node in the search tree."""
    config = _get_config()
    try:
        from crucible.researcher.search_tree import SearchTree

        name = args["name"]
        tree_dir = _get_tree_dir(config, name)
        tree = SearchTree.load(tree_dir)

        parent_id = args["parent_node_id"]
        children = args["children"]
        new_ids = tree.expand_node(parent_id, children)

        return {
            "status": "expanded",
            "parent_node_id": parent_id,
            "new_node_ids": new_ids,
            "total_nodes": tree.meta["total_nodes"],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def tree_auto_expand(args: dict[str, Any]) -> dict[str, Any]:
    """LLM-generate children for a node. Requires ANTHROPIC_API_KEY."""
    config = _get_config()
    try:
        import os

        from crucible.researcher.search_tree import SearchTree

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not set. Required for auto-expansion."}

        name = args["name"]
        tree_dir = _get_tree_dir(config, name)
        tree = SearchTree.load(tree_dir)

        node_id = args["node_id"]
        node = tree.get_node(node_id)
        if node is None:
            return {"error": f"Node '{node_id}' not found"}

        n_children = args.get("n_children", 3)
        extra_context = args.get("extra_context", "")

        ancestry = tree.get_ancestry(node_id)
        siblings = tree.get_siblings(node_id)
        summary = tree.get_tree_summary()

        ancestry_str = "\n".join(
            f"  depth={a['depth']} name={a['experiment_name']} "
            f"config={json.dumps(a['config'])} metric={a.get('result_metric')}"
            for a in ancestry
        )
        sibling_str = "\n".join(
            f"  name={s['experiment_name']} config={json.dumps(s['config'])} "
            f"metric={s.get('result_metric')} status={s['status']}"
            for s in siblings
        ) or "  (none)"

        prompt = (
            f"You are expanding a search tree of ML experiment configurations.\n\n"
            f"Tree: {summary.get('name')} - {summary.get('description', '')}\n"
            f"Primary metric: {summary.get('primary_metric')} ({summary.get('metric_direction')})\n"
            f"Best so far: {summary.get('best_metric')}\n\n"
            f"Current node path (root to current):\n{ancestry_str}\n\n"
            f"Siblings of current node:\n{sibling_str}\n\n"
            f"Current node config: {json.dumps(node['config'])}\n"
            f"Current node result: metric={node.get('result_metric')}\n"
            f"Current node hypothesis: {node.get('hypothesis', '')}\n\n"
        )
        if extra_context:
            prompt += f"Additional context: {extra_context}\n\n"

        prompt += (
            f"Generate exactly {n_children} child experiment configurations as JSON.\n"
            f"Each child should modify the parent config to test a specific hypothesis.\n"
            f"Return a JSON array of objects with: name, config (dict of overrides), "
            f"hypothesis, rationale.\n"
            f"Only return the JSON array, no other text."
        )

        try:
            import anthropic
        except ImportError:
            return {"error": "anthropic package not installed. Run: pip install anthropic"}

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        children_specs = json.loads(response_text)
        if not isinstance(children_specs, list):
            return {"error": "LLM response was not a JSON array"}

        for spec in children_specs:
            spec["generation_method"] = "llm_auto_expand"

        new_ids = tree.expand_node(node_id, children_specs)
        return {
            "status": "auto_expanded",
            "node_id": node_id,
            "new_node_ids": new_ids,
            "children": [
                {
                    "node_id": nid,
                    "name": tree.get_node(nid)["experiment_name"],
                    "hypothesis": tree.get_node(nid).get("hypothesis", ""),
                }
                for nid in new_ids
            ],
            "total_nodes": tree.meta["total_nodes"],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except json.JSONDecodeError as exc:
        return {"error": f"Failed to parse LLM response as JSON: {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def tree_prune(args: dict[str, Any]) -> dict[str, Any]:
    """Prune a node or entire branch in the search tree."""
    config = _get_config()
    try:
        from crucible.researcher.search_tree import SearchTree

        name = args["name"]
        tree_dir = _get_tree_dir(config, name)
        tree = SearchTree.load(tree_dir)

        node_id = args["node_id"]
        reason = args.get("reason", "")
        prune_branch = args.get("prune_branch", False)

        if prune_branch:
            count = tree.prune_branch(node_id, reason)
            return {
                "status": "branch_pruned",
                "node_id": node_id,
                "nodes_pruned": count,
                "total_pruned": tree.meta["pruned_nodes"],
            }
        else:
            tree.prune_node(node_id, reason)
            return {
                "status": "node_pruned",
                "node_id": node_id,
                "total_pruned": tree.meta["pruned_nodes"],
            }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def tree_enqueue_pending(args: dict[str, Any]) -> dict[str, Any]:
    """Move pending tree nodes to the fleet queue."""
    config = _get_config()
    try:
        from crucible.fleet.queue import enqueue_experiments
        from crucible.researcher.search_tree import SearchTree

        contract = _queue_contract_fields(config)
        name = args["name"]
        tree_dir = _get_tree_dir(config, name)
        tree = SearchTree.load(tree_dir)

        node_ids = args.get("node_ids")
        tier = args.get("tier", "proxy")
        backend = args.get("backend", "torch")

        if node_ids:
            nodes_to_enqueue = [
                tree.get_node(nid) for nid in node_ids
                if tree.get_node(nid) and tree.get_node(nid)["status"] == "pending"
            ]
        else:
            nodes_to_enqueue = tree.get_pending_nodes()

        if not nodes_to_enqueue:
            return {"status": "no_pending_nodes", "enqueued": 0}

        experiments = []
        for node in nodes_to_enqueue:
            experiments.append({
                "name": node["experiment_name"],
                "config": node["config"],
                "tier": tier,
                "backend": backend,
                "tags": node.get("tags", []) + [f"tree:{name}", f"node:{node['node_id']}"],
                **contract,
            })

        queue_path = config.project_root / "fleet_queue.jsonl"
        added = enqueue_experiments(queue_path, experiments, limit=len(experiments))

        enqueued_info = []
        for item in added:
            for node in nodes_to_enqueue:
                if node["experiment_name"] == item["experiment_name"]:
                    node["status"] = "queued"
                    node["run_id"] = item["run_id"]
                    tree._append_node_event("enqueue", node)
                    enqueued_info.append({
                        "node_id": node["node_id"],
                        "run_id": item["run_id"],
                        "experiment_name": node["experiment_name"],
                    })
                    break

        tree.meta["updated_at"] = utc_now_iso()
        tree._save_meta()
        tree._save_snapshot()

        return {
            "status": "enqueued",
            "enqueued": len(added),
            "items": enqueued_info,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def tree_sync_results(args: dict[str, Any]) -> dict[str, Any]:
    """Match completed queue results to tree nodes."""
    config = _get_config()
    try:
        from crucible.analysis.results import merged_results
        from crucible.researcher.search_tree import SearchTree

        name = args["name"]
        tree_dir = _get_tree_dir(config, name)
        tree = SearchTree.load(tree_dir)

        all_results = merged_results(config)
        results_by_id = {r.get("id", r.get("run_id", "")): r for r in all_results}

        synced = []
        for node in tree.nodes.values():
            if node["status"] in ("queued", "running") and node.get("run_id"):
                result = results_by_id.get(node["run_id"])
                if result and result.get("status") == "completed":
                    result_data = result.get("result", {})
                    result_data["run_id"] = node["run_id"]
                    tree.record_result(node["node_id"], result_data)
                    synced.append({
                        "node_id": node["node_id"],
                        "run_id": node["run_id"],
                        "metric": node.get("result_metric"),
                    })

        return {
            "status": "synced",
            "synced_count": len(synced),
            "synced_nodes": synced,
            "best_node_id": tree.meta.get("best_node_id"),
            "best_metric": tree.meta.get("best_metric"),
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def tree_list(args: dict[str, Any]) -> dict[str, Any]:
    """List all search trees."""
    config = _get_config()
    try:
        from crucible.researcher.search_tree import SearchTree

        store_dir = config.project_root / config.store_dir
        trees_dir = store_dir / "search_trees"

        if not trees_dir.exists():
            return {"trees": [], "total": 0}

        trees = []
        for entry in sorted(trees_dir.iterdir()):
            if entry.is_dir() and (entry / "tree.yaml").exists():
                try:
                    tree = SearchTree.load(entry)
                    trees.append(tree.get_tree_summary())
                except Exception:
                    trees.append({"name": entry.name, "status": "error", "error": "failed to load"})

        return {"trees": trees, "total": len(trees)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Modalities tool
# ---------------------------------------------------------------------------


def config_get_modalities(args: dict[str, Any]) -> dict[str, Any]:
    """List available training backends with their modality tags."""
    config = _get_config()
    try:
        from crucible.training.data_adapters import DATA_ADAPTER_REGISTRY
        from crucible.training.objectives import OBJECTIVE_REGISTRY

        backends = []
        for t in config.training:
            backends.append({
                "backend": t.backend,
                "script": t.script,
                "modality": t.modality,
            })

        return {
            "training_backends": backends,
            "data_adapters": sorted(DATA_ADAPTER_REGISTRY.keys()),
            "objectives": sorted(OBJECTIVE_REGISTRY.keys()),
        }
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# External project tools
# ---------------------------------------------------------------------------

# Persistent registry for project runs (survives MCP server restarts)
_PROJECT_RUNS_FILE = ".crucible/projects/runs.jsonl"
_PROJECT_RUN_EVENTS_FILE = ".crucible/projects/run_events.jsonl"
_PROJECT_ACTIVE_STATUSES = frozenset({"launching", "launched", "running"})
_PROJECT_TERMINAL_STATUSES = frozenset({"completed", "failed", "timeout", "killed", "interrupted"})


def _project_runs_path() -> Path:
    config = _get_config()
    return config.project_root / _PROJECT_RUNS_FILE


def _project_run_events_path() -> Path:
    config = _get_config()
    return config.project_root / _PROJECT_RUN_EVENTS_FILE


def _save_project_run(run_id: str, data: dict[str, Any]) -> None:
    """Persist a project run record to JSONL."""
    runs_path = _project_runs_path()
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    from crucible.core.io import append_jsonl, _json_ready
    payload = {"run_id": run_id, "updated_at": utc_now_iso(), **_json_ready(data)}
    record = payload
    append_jsonl(runs_path, record)


def _load_project_run(run_id: str) -> dict[str, Any] | None:
    """Load a project run record from JSONL."""
    runs_path = _project_runs_path()
    if not runs_path.exists():
        return None
    latest = None
    for record in read_jsonl(runs_path):
        if record.get("run_id") == run_id:
            latest = record
    return latest


def _load_project_runs(*, launch_id: str | None = None) -> list[dict[str, Any]]:
    """Return latest snapshots for all project runs, optionally filtered by launch."""
    latest_by_run: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(_project_runs_path()):
        run_id = str(record.get("run_id", "")).strip()
        if not run_id:
            continue
        latest_by_run[run_id] = record
    rows = list(latest_by_run.values())
    if launch_id:
        rows = [row for row in rows if row.get("launch_id") == launch_id]
    rows.sort(key=lambda row: row.get("updated_at") or row.get("launched_at") or "", reverse=True)
    return rows


def _update_project_run(run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates onto the latest project-run snapshot and persist it."""
    current = _load_project_run(run_id) or {"run_id": run_id}
    merged = {**current, **updates}
    _save_project_run(run_id, merged)
    return merged


def _append_project_run_event(run_id: str, event: str, **fields: Any) -> None:
    """Persist an append-only lifecycle event for a project run."""
    path = _project_run_events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": utc_now_iso(),
        "run_id": run_id,
        "event": event,
        **fields,
    }
    from crucible.core.io import append_jsonl, _json_ready
    append_jsonl(path, _json_ready(payload))


def _load_project_run_events(
    *,
    run_id: str | None = None,
    launch_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load project-run lifecycle events, optionally filtered."""
    records = read_jsonl(_project_run_events_path())
    if run_id:
        records = [row for row in records if row.get("run_id") == run_id]
    if launch_id:
        records = [row for row in records if row.get("launch_id") == launch_id]
    if limit is not None and limit >= 0:
        records = records[-limit:]
    return records


def _sanitize_node_token(name: str) -> str:
    import re

    token = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return token or "node"


def _make_launch_id(project_name: str) -> str:
    return f"{project_name}_{time.time_ns()}_{uuid4().hex[:6]}"


def _make_node_run_id(launch_id: str, node_name: str, *, total_nodes: int) -> str:
    if total_nodes == 1:
        return launch_id
    return f"{launch_id}_{_sanitize_node_token(node_name)}"


def _status_event_name(status: str, *, collected: bool = False) -> str:
    if collected:
        return "result_collected"
    if status in {"launching", "launched"}:
        return f"status_{status}"
    if status == "running":
        return "probe_running"
    if status == "interrupted":
        return "node_unreachable"
    if status in _PROJECT_TERMINAL_STATUSES:
        return "probe_terminal"
    return "status_observed"


def _persist_project_observation(
    run_id: str,
    updates: dict[str, Any],
    *,
    event_name: str | None = None,
    event_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a project-run snapshot update and optional event."""
    merged = _update_project_run(run_id, updates)
    if event_name:
        _append_project_run_event(
            run_id,
            event_name,
            launch_id=merged.get("launch_id"),
            project=merged.get("project"),
            status=merged.get("status"),
            node_name=merged.get("node_name") or merged.get("remote_node"),
            reason=merged.get("status_reason"),
            failure_class=merged.get("failure_class"),
            pid=merged.get("pid"),
            details=event_details or None,
        )
    return merged


def list_projects(args: dict[str, Any]) -> dict[str, Any]:
    """List all external project specs in .crucible/projects/."""
    config = _get_config()
    try:
        from crucible.core.config import list_project_specs
        specs = list_project_specs(config.project_root)
        return {"projects": specs}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def provision_project(args: dict[str, Any]) -> dict[str, Any]:
    """Provision nodes for an external project, applying pod overrides from the spec.

    REQUIRES: RUNPOD_API_KEY in .env, project spec in .crucible/projects/.
    RETURNS: {created, new_nodes: [{name, node_id}]}
    NEXT: fleet_refresh (wait ~60s), then bootstrap_project.
    """
    config = _get_config()
    try:
        from crucible.core.config import load_project_spec
        project_name = args["project_name"]
        count = args.get("count", 1)
        spec = load_project_spec(project_name, config.project_root)
        _project_contract_env(config, spec)

        from crucible.fleet.inventory import load_nodes_if_exists, next_node_index
        from crucible.fleet.manager import FleetManager
        fm = FleetManager(config)
        name_prefix = project_name[:12]
        existing_nodes = load_nodes_if_exists(config.project_root / config.nodes_file)

        # Apply pod overrides if present
        provider_overrides: dict[str, Any] = {}
        if spec.pod.image:
            provider_overrides["image_name"] = spec.pod.image
        if spec.pod.gpu_type:
            provider_overrides["gpu_type_ids"] = spec.pod.gpu_type
        if spec.pod.container_disk:
            provider_overrides["container_disk_gb"] = spec.pod.container_disk
        if spec.pod.volume_disk:
            provider_overrides["volume_gb"] = spec.pod.volume_disk
        if spec.pod.interruptible is not None:
            provider_overrides["interruptible"] = spec.pod.interruptible
        if spec.pod.gpu_count:
            provider_overrides["gpu_count"] = spec.pod.gpu_count

        previous_ids = {
            n.get("node_id") or n.get("pod_id")
            for n in existing_nodes
            if n.get("node_id") or n.get("pod_id")
        }
        nodes = fm.provision(
            count=count,
            name_prefix=name_prefix,
            start_index=next_node_index(existing_nodes, name_prefix),
            **provider_overrides,
        )
        new_nodes = [
            n for n in nodes
            if (n.get("node_id") or n.get("pod_id")) not in previous_ids
        ]
        return {
            "created": len(new_nodes),
            "new_nodes": [
                {"name": n.get("name", ""), "node_id": n.get("node_id", "")}
                for n in new_nodes
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def bootstrap_project_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Bootstrap an external project on fleet nodes: clone, venv, install, setup.

    REQUIRES: Nodes with SSH (run fleet_refresh after provision_project).
    RETURNS: {total, bootstrapped, nodes: [{name, state, project}]}
    NEXT: run_project.
    """
    config = _get_config()
    try:
        from crucible.core.config import load_project_spec
        from crucible.fleet.bootstrap import bootstrap_project as _bootstrap_project
        from crucible.fleet.inventory import (
            load_nodes_if_exists,
            upsert_node_record,
        )

        project_name = args["project_name"]
        spec = load_project_spec(project_name, config.project_root)
        node_names = args.get("node_names")

        nodes_file = config.project_root / config.nodes_file
        all_nodes = load_nodes_if_exists(nodes_file) or []
        ssh_nodes = [n for n in all_nodes if n.get("ssh_host")]

        if node_names:
            selected = set(node_names)
            ssh_nodes = [n for n in ssh_nodes if n["name"] in selected]

        if not ssh_nodes:
            return {
                "error": "No nodes with SSH found. Run fleet_refresh first.",
                "total": 0, "bootstrapped": 0, "nodes": [],
            }

        results = []
        for node in ssh_nodes:
            # Apply workspace from spec
            node["workspace_path"] = spec.workspace
            try:
                updated = _bootstrap_project(node, spec, project_root=config.project_root)
                # Clear any stale error from a previous failed bootstrap — the
                # node is now ready and should not display old error strings.
                updated.pop("error", None)
                upsert_node_record(nodes_file, updated)
                results.append(updated)
            except Exception as exc:
                node["state"] = "boot_failed"
                node["error"] = str(exc)
                upsert_node_record(nodes_file, node)
                results.append(node)

        bootstrapped = [n for n in results if n.get("state") == "ready"]
        return {
            "total": len(results),
            "bootstrapped": len(bootstrapped),
            "nodes": [
                {
                    "name": n.get("name", ""),
                    "state": n.get("state", "unknown"),
                    "project": n.get("project", ""),
                    **({"error": n.get("error", "")} if n.get("error") else {}),
                }
                for n in results
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def run_project(args: dict[str, Any]) -> dict[str, Any]:
    """Launch training for an external project on bootstrapped nodes.

    REQUIRES: Nodes bootstrapped via bootstrap_project.
    RETURNS: {run_id, nodes: [{name, pid, status}]}
    NEXT: get_fleet_status to monitor, collect_project_results when done.
    """
    config = _get_config()
    try:
        from crucible.core.config import load_project_spec
        from crucible.core.hub import HubStore
        from crucible.fleet.project_runner import launch_project
        from crucible.fleet.project_launchers import launcher_runtime_entry, resolve_launcher_bundle
        from crucible.fleet.inventory import load_nodes_if_exists

        project_name = args["project_name"]
        spec = load_project_spec(project_name, config.project_root)
        node_names = args.get("node_names")

        # Variant selection: if the caller passes ``variant=<name>``, look it
        # up in ``spec.variants`` and merge its env dict into ``overrides``
        # FIRST, so the caller's explicit ``overrides`` still win if the same
        # key is in both. This is the fix for the "variants dict is inert"
        # trap documented in docs/crucible-config-hierarchy.md §4. Passing an
        # unknown variant name is a loud error (never silently ignored).
        variant_arg = args.get("variant") or ""
        caller_overrides = dict(args.get("overrides", {}))
        if variant_arg:
            if variant_arg not in spec.variants:
                available = sorted(spec.variants.keys()) or ["(none)"]
                return {
                    "error": (
                        f"Variant {variant_arg!r} not found in project spec "
                        f"{project_name!r}. Available variants: {', '.join(available)}."
                    ),
                    "run_id": "",
                    "nodes": [],
                }
            variant_env = dict(spec.variants[variant_arg])
        else:
            variant_env = {}

        overrides = {**variant_env, **caller_overrides}
        contract_env = _project_contract_env(config, spec)
        overrides.update(contract_env)
        spec.env_set.update(contract_env)
        launch_id = _make_launch_id(project_name)
        variant_name = str(
            variant_arg
            or overrides.get("CRUCIBLE_VARIANT_NAME")
            or overrides.get("WANDB_RUN_NAME")
            or launch_id
        )
        launcher_info = None
        launcher_path = ""
        if spec.launcher:
            hub_dir = HubStore.resolve_hub_dir(config_hub_dir=getattr(config, "hub_dir", ""))
            launcher_info = resolve_launcher_bundle(
                project_root=config.project_root,
                launcher_name=spec.launcher,
                hub_dir=hub_dir,
            )
            if launcher_info is None:
                return {
                    "error": (
                        f"Launcher bundle {spec.launcher!r} not found in local plugins, "
                        "installed hub packages, or configured taps."
                    ),
                    "run_id": launch_id,
                    "nodes": [],
                }
            entry = spec.launcher_entry or launcher_info["entry"]
            launcher_path = launcher_runtime_entry(spec.workspace, spec.launcher, entry)

        nodes_file = config.project_root / config.nodes_file
        all_nodes = load_nodes_if_exists(nodes_file) or []
        ready_nodes = [
            n for n in all_nodes
            if n.get("env_ready") and n.get("project") == project_name
        ]

        if node_names:
            selected = set(node_names)
            ready_nodes = [n for n in ready_nodes if n["name"] in selected]

        if not ready_nodes:
            return {
                "error": f"No nodes bootstrapped for project {project_name!r}. "
                         f"Run bootstrap_project first.",
                "run_id": launch_id, "nodes": [],
            }

        launched = []
        total_nodes = len(ready_nodes)
        for node in ready_nodes:
            node["workspace_path"] = spec.workspace
            node_run_id = _make_node_run_id(launch_id, node["name"], total_nodes=total_nodes)
            wandb_run_name = str(overrides.get("WANDB_RUN_NAME") or variant_name or node_run_id)
            launch_overrides = {
                **overrides,
                "WANDB_RUN_NAME": wandb_run_name,
                "CRUCIBLE_REMOTE_NODE": node["name"],
                "CRUCIBLE_EXECUTION_PROVIDER": config.provider.type.lower(),
                "CRUCIBLE_ENFORCE_CONTRACT": "1",
                "CRUCIBLE_VARIANT_NAME": variant_name,
            }
            if launcher_path and "TRAIN_SCRIPT" not in launch_overrides:
                launch_overrides["TRAIN_SCRIPT"] = launcher_path
            contract = contract_metadata(config, env={**os.environ, **launch_overrides}, remote_node=node["name"])
            base_record = {
                "launch_id": launch_id,
                "project": spec.name,
                "variant_name": variant_name,
                "result_name": variant_name,
                "node_name": node["name"],
                "remote_node": node["name"],
                "ssh_host": node.get("ssh_host", ""),
                "ssh_port": node.get("ssh_port", 22),
                "workspace_path": spec.workspace,
                "overrides": overrides,
                "resolved_overrides": launch_overrides,
                "train_command": spec.train,
                "spec_name": spec.name,
                "spec_snapshot": {
                    "repo": spec.repo,
                    "branch": spec.branch,
                    "workspace": spec.workspace,
                    "launcher": spec.launcher,
                    "launcher_entry": spec.launcher_entry,
                    "train": spec.train,
                },
                "execution_provider": contract["execution_provider"],
                "contract_status": contract["contract_status"],
                "launcher": spec.launcher or None,
                "launcher_entry": spec.launcher_entry or (launcher_info["entry"] if launcher_info else ""),
                "launcher_source": launcher_info["source"] if launcher_info else "",
                "launcher_manifest": launcher_info["manifest"] if launcher_info else None,
                "launcher_runtime_path": launcher_path or None,
                "wandb": {
                    **contract["wandb"],
                    "run_name": wandb_run_name,
                },
            }
            _persist_project_observation(
                node_run_id,
                {
                    **base_record,
                    "status": "launching",
                    "status_reason": "launch_requested",
                },
                event_name="launch_requested",
            )
            try:
                _append_project_run_event(
                    node_run_id,
                    "launch_started",
                    launch_id=launch_id,
                    project=spec.name,
                    status="launching",
                    node_name=node["name"],
                    details={"variant_name": variant_name},
                )
                result = launch_project(node, spec, node_run_id, overrides=launch_overrides)
                _persist_project_observation(
                    node_run_id,
                    {
                        **base_record,
                        "pid": result["pid"],
                        "status": "launched",
                        "status_reason": None,
                        "launched_at": utc_now_iso(),
                        "last_observed_at": utc_now_iso(),
                    },
                    event_name="launch_succeeded",
                    event_details={"pid": result["pid"]},
                )
                launched.append({
                    "name": node["name"],
                    "run_id": node_run_id,
                    "pid": result["pid"],
                    "status": "launched",
                    "variant_name": variant_name,
                })
            except Exception as exc:
                _persist_project_observation(
                    node_run_id,
                    {
                        **base_record,
                        "status": "failed",
                        "status_reason": str(exc),
                        "failure_class": "launch_failed",
                        "completed_at": utc_now_iso(),
                    },
                    event_name="launch_failed",
                    event_details={"error": str(exc)},
                )
                launched.append({
                    "name": node["name"],
                    "run_id": node_run_id,
                    "pid": None,
                    "status": f"failed: {exc}",
                })
        response: dict[str, Any] = {"launch_id": launch_id, "nodes": launched, "variant_name": variant_name}
        if len(launched) == 1 and launched[0].get("run_id"):
            response["run_id"] = launched[0]["run_id"]
        if spec.launcher:
            response["launcher"] = spec.launcher
        return response
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def run_project_chain(args: dict[str, Any]) -> dict[str, Any]:
    """Run a sequence of project variants on the same node, auto-chaining.

    REQUIRES: Node bootstrapped via bootstrap_project.
    RETURNS: {chain_id, variants_total, results: [{variant, run_id, status, duration_s}]}
    NEXT: collect_project_results for individual runs.
    """
    config = _get_config()
    try:
        from crucible.core.config import load_project_spec
        from crucible.fleet.inventory import load_nodes_if_exists
        from crucible.fleet.project_runner import chain_project_variants

        project_name = args["project_name"]
        variants = args["variants"]
        node_name = args["node_name"]
        overrides = dict(args.get("overrides", {}))
        poll_interval = int(args.get("poll_interval", 30))

        spec = load_project_spec(project_name, config.project_root)

        # Validate all variants exist before starting
        missing = [v for v in variants if v not in spec.variants]
        if missing:
            available = sorted(spec.variants.keys()) or ["(none)"]
            return {
                "error": f"Unknown variants: {missing}. Available: {', '.join(available)}",
            }

        # Find the target node
        nodes_file = config.project_root / config.nodes_file
        all_nodes = load_nodes_if_exists(nodes_file) or []
        node = next((n for n in all_nodes if n["name"] == node_name), None)
        if node is None:
            return {"error": f"Node {node_name!r} not found in inventory."}
        if not node.get("env_ready"):
            return {"error": f"Node {node_name!r} not bootstrapped (env_ready=false)."}

        node["workspace_path"] = spec.workspace

        # Add contract env
        contract_env = _project_contract_env(config, spec)
        overrides.update(contract_env)

        chain_id = _make_launch_id(f"{project_name}_chain")

        results = chain_project_variants(
            node=node,
            spec=spec,
            variants=variants,
            overrides=overrides,
            poll_interval=poll_interval,
        )

        return {
            "chain_id": chain_id,
            "project": project_name,
            "node": node_name,
            "variants_total": len(variants),
            "variants_completed": sum(1 for r in results if r["status"] == "completed"),
            "results": results,
        }

    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def _observe_project_run(
    run_info: dict[str, Any],
    *,
    persist_result_row: bool,
    event_name: str,
) -> dict[str, Any]:
    """Probe a persisted external project run and update its latest snapshot."""
    config = _get_config()
    run_id = run_info["run_id"]
    if run_info.get("pid") is None and run_info.get("status") in _PROJECT_TERMINAL_STATUSES:
        latest = _persist_project_observation(
            run_id,
            {
                **run_info,
                "last_observed_at": utc_now_iso(),
            },
            event_name=event_name,
            event_details={"status": run_info.get("status")},
        )
        return {
            "run_id": run_id,
            "launch_id": latest.get("launch_id"),
            "variant_name": latest.get("variant_name"),
            "status": latest.get("status"),
            "metrics": latest.get("result"),
            "log_tail": "",
            "log_path": latest.get("log_path", ""),
            "wandb": latest.get("wandb"),
            "contract_status": latest.get("contract_status", "legacy_missing_contract"),
            "failure_class": latest.get("failure_class"),
            "remote_node_state": latest.get("remote_node_state"),
        }

    from crucible.core.config import load_project_spec
    from crucible.fleet.inventory import load_nodes_if_exists
    from crucible.fleet.project_runner import collect_project_result

    nodes = load_nodes_if_exists(config.project_root / config.nodes_file) or []
    node = next((n for n in nodes if n["name"] == run_info["node_name"]), None)
    if node is None:
        node = {
            "name": run_info["node_name"],
            "ssh_host": run_info.get("ssh_host", ""),
            "ssh_port": run_info.get("ssh_port", 22),
            "state": "unreachable",
        }
    node["workspace_path"] = run_info.get("workspace_path", "/workspace/project")

    spec = load_project_spec(run_info["project"], config.project_root)
    if run_info.get("wandb"):
        if run_info["wandb"].get("project"):
            spec.env_set["WANDB_PROJECT"] = run_info["wandb"]["project"]
        if run_info["wandb"].get("entity"):
            spec.env_set["WANDB_ENTITY"] = run_info["wandb"]["entity"]
        if run_info["wandb"].get("mode"):
            spec.env_set["WANDB_MODE"] = run_info["wandb"]["mode"]

    result = collect_project_result(
        node=node,
        spec=spec,
        run_id=run_id,
        pid=run_info["pid"],
        wandb_run_name=(run_info.get("wandb") or {}).get("run_name"),
        experiment_meta={
            "name": run_info.get("result_name") or run_info.get("variant_name") or f"{spec.name}-{run_id}",
            "config": run_info.get("resolved_overrides") or run_info.get("overrides") or {},
            "project": run_info.get("project", spec.name),
            "launcher": run_info.get("launcher"),
            "launcher_source": run_info.get("launcher_source"),
            "variant_name": run_info.get("variant_name"),
            "launch_id": run_info.get("launch_id"),
        },
        local_logs_dir=config.project_root / config.logs_dir,
        results_file=(config.project_root / config.fleet_results_file) if persist_result_row else None,
    )
    if run_info.get("wandb"):
        merged_wandb = dict(run_info["wandb"])
        for key, value in (result.get("wandb") or {}).items():
            if value is not None:
                merged_wandb[key] = value
        result["wandb"] = merged_wandb
    if run_info.get("execution_provider"):
        result["execution_provider"] = run_info["execution_provider"]
    if run_info.get("remote_node"):
        result["remote_node"] = run_info["remote_node"]
    if run_info.get("contract_status") and result.get("contract_status") == "compliant":
        result["contract_status"] = run_info["contract_status"]

    updates = {
        **run_info,
        "status": result["status"],
        "status_reason": result.get("failure_class"),
        "failure_class": result.get("failure_class"),
        "last_observed_at": result.get("last_observed_at") or utc_now_iso(),
        "completed_at": utc_now_iso() if result["status"] in _PROJECT_TERMINAL_STATUSES else None,
        "result": result.get("result"),
        "log_path": result.get("log_path"),
        "wandb": result.get("wandb"),
        "contract_status": result.get("contract_status"),
        "remote_node_state": result.get("remote_node_state"),
    }
    latest = _persist_project_observation(
        run_id,
        updates,
        event_name=event_name,
        event_details={
            "status": result["status"],
            "failure_class": result.get("failure_class"),
        },
    )

    response = {
        "run_id": run_id,
        "launch_id": latest.get("launch_id"),
        "variant_name": latest.get("variant_name"),
        "status": latest.get("status"),
        "metrics": result.get("result"),
        "log_tail": result.get("log_tail", "")[-1000:],
        "log_path": result.get("log_path", ""),
        "wandb": result.get("wandb"),
        "contract_status": result.get("contract_status", "legacy_missing_contract"),
        "failure_class": result.get("failure_class"),
        "remote_node_state": result.get("remote_node_state"),
    }
    if latest.get("launcher"):
        response["launcher"] = latest["launcher"]
    return response


def collect_project_results(args: dict[str, Any]) -> dict[str, Any]:
    """Collect results from an external project run.

    REQUIRES: run_project has been called.
    RETURNS: {status, metrics, log_tail, run_id}
    """
    config = _get_config()
    try:
        launch_id = args.get("launch_id")
        run_id = args.get("run_id")

        if launch_id:
            runs = _load_project_runs(launch_id=launch_id)
            if not runs:
                return {"error": f"No runs found for launch_id {launch_id!r}."}
            collected = [
                _observe_project_run(run_info, persist_result_row=True, event_name="result_collected")
                for run_info in runs
            ]
            summary: dict[str, int] = {}
            for row in collected:
                status = str(row.get("status") or "unknown")
                summary[status] = summary.get(status, 0) + 1
            return {"launch_id": launch_id, "runs": collected, "summary": summary}

        if not run_id:
            return {"error": "Provide either run_id or launch_id."}

        run_info = _load_project_run(run_id)
        if run_info is None:
            return {"error": f"No run found for {run_id!r}. Was run_project called?"}

        response = _observe_project_run(run_info, persist_result_row=True, event_name="result_collected")
        if response["status"] not in {"running", "interrupted"} and response["contract_status"] != "compliant":
            response["error"] = (
                f"Experiment completed without a compliant W&B run "
                f"(contract_status={response['contract_status']})."
            )
        return response
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def get_project_run_status(args: dict[str, Any]) -> dict[str, Any]:
    """Probe and return the latest lifecycle view for an external project run."""
    try:
        run_id = args["run_id"]
        run_info = _load_project_run(run_id)
        if run_info is None:
            return {"error": f"No run found for {run_id!r}."}

        response = _observe_project_run(run_info, persist_result_row=False, event_name="status_observed")
        response["events"] = _load_project_run_events(run_id=run_id, limit=int(args.get("event_limit", 10)))
        return response
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Recipe tools — save / list / get session playbooks
# ---------------------------------------------------------------------------

_RECIPE_NAME_RE = None  # lazy-compiled


def _validate_recipe_name(name: str) -> str | None:
    """Return None if valid, or an error message."""
    import re

    global _RECIPE_NAME_RE
    if _RECIPE_NAME_RE is None:
        _RECIPE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
    if not _RECIPE_NAME_RE.match(name):
        return (
            f"Invalid recipe name {name!r}. "
            "Use lowercase alphanumeric, hyphens, and underscores (max 128 chars). "
            "Must start with a letter or digit."
        )
    return None


def _recipes_dir(config: Any) -> Path:
    return config.project_root / config.store_dir / "recipes"


def recipe_save(args: dict[str, Any]) -> dict[str, Any]:
    """Save a session recipe — a step-by-step reproduction guide.

    REQUIRES: Nothing.
    RETURNS: {saved, path, name, overwritten}
    NEXT: recipe_list to see all recipes, recipe_get to retrieve one.
    """
    import yaml
    from crucible.core.errors import RecipeError

    config = _get_config()
    try:
        name = args.get("name", "")
        if not name:
            raise RecipeError("Recipe name is required.")

        err = _validate_recipe_name(name)
        if err:
            raise RecipeError(err)

        steps = args.get("steps", [])
        if not steps:
            raise RecipeError("At least one step is required.")

        for i, step in enumerate(steps):
            if not isinstance(step, dict) or "tool" not in step:
                raise RecipeError(
                    f"Step {i} must be a dict with at least a 'tool' key."
                )

        recipes_dir = _recipes_dir(config)
        recipes_dir.mkdir(parents=True, exist_ok=True)

        path = recipes_dir / f"{name}.yaml"
        overwritten = path.exists()

        recipe = {
            "name": name,
            "title": args.get("title", ""),
            "created_at": utc_now_iso(),
            "created_by": args.get("created_by", "mcp-agent"),
            "goal": args.get("goal", ""),
            "project_spec": args.get("project_spec", ""),
            "environment": args.get("environment", {}),
            "steps": steps,
            "results": args.get("results", {}),
            "gotchas": args.get("gotchas", []),
            "tags": args.get("tags", []),
        }

        path.write_text(
            yaml.safe_dump(recipe, default_flow_style=False, sort_keys=False)
        )

        return {"saved": True, "path": str(path), "name": name, "overwritten": overwritten}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def recipe_list(args: dict[str, Any]) -> dict[str, Any]:
    """List all saved session recipes.

    REQUIRES: Nothing.
    RETURNS: {recipes: [{name, title, created_at, tags, project_spec, goal}], total}
    NEXT: recipe_get for full details.
    """
    import yaml

    config = _get_config()
    try:
        recipes_dir = _recipes_dir(config)
        if not recipes_dir.exists():
            return {"recipes": [], "total": 0}

        tag_filter = args.get("tag")
        recipes = []
        for path in sorted(recipes_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text())
                if not isinstance(data, dict):
                    continue
            except Exception:
                continue

            tags = data.get("tags", [])
            if tag_filter and tag_filter not in tags:
                continue

            recipes.append({
                "name": data.get("name", path.stem),
                "title": data.get("title", ""),
                "created_at": data.get("created_at", ""),
                "tags": tags,
                "project_spec": data.get("project_spec", ""),
                "goal": data.get("goal", ""),
            })

        return {"recipes": recipes, "total": len(recipes)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def recipe_get(args: dict[str, Any]) -> dict[str, Any]:
    """Get a saved session recipe with full step-by-step details.

    REQUIRES: Recipe exists.
    RETURNS: Full recipe with steps, environment, gotchas, results.
    NEXT: Follow the steps to reproduce.
    """
    import yaml
    from crucible.core.errors import RecipeError

    config = _get_config()
    try:
        name = args.get("name", "")
        if not name:
            raise RecipeError("Recipe name is required.")

        path = _recipes_dir(config) / f"{name}.yaml"
        if not path.exists():
            raise RecipeError(
                f"Recipe {name!r} not found. Use recipe_list to see available recipes."
            )

        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise RecipeError(f"Recipe {name!r} has invalid format.")

        return data
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Plugin registry tools (optimizers, schedulers, providers, loggers, callbacks,
# composer block types, stack patterns, augmentations)
# ---------------------------------------------------------------------------


def _plugin_add_common(args: dict[str, Any], plugin_type: str) -> dict[str, Any]:
    """Common logic for all plugin-add tools: save code to disk and load via importlib."""
    import importlib.util
    import sys as _sys

    name = args.get("name", "")
    code = args.get("code", "")
    if not name or not code:
        return {"error": "Both 'name' and 'code' are required."}
    try:
        config = _get_config()
        plugin_dir = config.project_root / config.store_dir / config.plugins.local_dir / plugin_type
        plugin_dir.mkdir(parents=True, exist_ok=True)
        path = plugin_dir / f"{name}.py"
        path.write_text(code, encoding="utf-8")

        # Load via importlib (same mechanism as PluginRegistry.load_plugins
        # and model_add_architecture) so the module lands in sys.modules and
        # won't double-execute on restart/rediscovery.
        module_name = f"_crucible_plugin_{plugin_type}_local_{name}"
        _sys.modules.pop(module_name, None)  # allow re-registration
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return {"error": f"Could not create module spec for {path}"}
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        return {"status": "registered", "name": name, "plugin_type": plugin_type,
                "path": str(path)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Generic plugin tools (consolidated from 15 type-specific tools)
# ---------------------------------------------------------------------------

_PLUGIN_LIST_DISPATCH: dict[str, tuple[str, str, str]] = {
    # type -> (module_path, function_or_registry_name, response_key)
    # For modules with list_*_detailed functions:
    "optimizers": ("crucible.training.optimizers", "list_optimizers_detailed", "optimizers"),
    "schedulers": ("crucible.training.schedulers", "list_schedulers_detailed", "schedulers"),
    "providers": ("crucible.fleet.provider_registry", "list_providers_detailed", "providers"),
    "loggers": ("crucible.runner.loggers", "list_loggers_detailed", "loggers"),
    "callbacks": ("crucible.training.callbacks", "list_callbacks_detailed", "callbacks"),
    # For composer registries (use PluginRegistry.list_plugins_detailed directly):
    "block_types": ("crucible.models.composer", "BLOCK_TYPE_REGISTRY", "block_types"),
    "stack_patterns": ("crucible.models.composer", "STACK_PATTERN_REGISTRY", "stack_patterns"),
    "augmentations": ("crucible.models.composer", "AUGMENTATION_REGISTRY", "augmentations"),
}

_PLUGIN_SCHEMA_DISPATCH: dict[str, tuple[str, str]] = {
    # type -> (module_path, registry_name)
    "optimizers": ("crucible.training.optimizers", "OPTIMIZER_REGISTRY"),
    "schedulers": ("crucible.training.schedulers", "SCHEDULER_REGISTRY"),
}

_VALID_PLUGIN_TYPES = sorted(_PLUGIN_LIST_DISPATCH.keys())


def plugin_list(args: dict[str, Any]) -> dict[str, Any]:
    """List registered plugins of a given type."""
    import importlib
    plugin_type = args.get("type", "")
    if plugin_type not in _PLUGIN_LIST_DISPATCH:
        return {"error": f"Unknown plugin type {plugin_type!r}. Valid: {_VALID_PLUGIN_TYPES}"}
    module_path, attr_name, key = _PLUGIN_LIST_DISPATCH[plugin_type]
    try:
        mod = importlib.import_module(module_path)
        attr = getattr(mod, attr_name)
        # attr is either a function (list_*_detailed) or a PluginRegistry instance
        items = attr() if callable(attr) else attr.list_plugins_detailed()
        return {key: items}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def plugin_add(args: dict[str, Any]) -> dict[str, Any]:
    """Register a new plugin from Python code."""
    plugin_type = args.get("type", "")
    if plugin_type not in _PLUGIN_LIST_DISPATCH:
        return {"error": f"Unknown plugin type {plugin_type!r}. Valid: {_VALID_PLUGIN_TYPES}"}
    return _plugin_add_common(args, plugin_type)


def plugin_get_schema(args: dict[str, Any]) -> dict[str, Any]:
    """Get the config schema for a named plugin (optimizers, schedulers only)."""
    import importlib
    plugin_type = args.get("type", "")
    if plugin_type not in _PLUGIN_SCHEMA_DISPATCH:
        supported = sorted(_PLUGIN_SCHEMA_DISPATCH.keys())
        return {"error": f"Schema not available for {plugin_type!r}. Supported: {supported}"}
    module_path, registry_name = _PLUGIN_SCHEMA_DISPATCH[plugin_type]
    name = args.get("name", "")
    try:
        mod = importlib.import_module(module_path)
        registry = getattr(mod, registry_name)
        return {"type": plugin_type, "name": name, "schema": registry.get_schema(name)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Community tap tools
# ---------------------------------------------------------------------------


def _get_tap_manager() -> Any:
    from crucible.core.tap import TapManager
    hub = _get_hub()
    if hub is None:
        return None
    return TapManager(hub.hub_dir)


def hub_tap_add(args: dict[str, Any]) -> dict[str, Any]:
    """Add a community plugin tap (git repo).\n\nREQUIRES: url.\nRETURNS: {name, url, added_at}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized. Run crucible hub init first."}
    try:
        return tm.add_tap(args.get("url", ""), name=args.get("name", ""))
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def hub_tap_remove(args: dict[str, Any]) -> dict[str, Any]:
    """Remove a tap.\n\nREQUIRES: name.\nRETURNS: {removed: true}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    try:
        tm.remove_tap(args.get("name", ""))
        return {"removed": True}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def hub_tap_list(args: dict[str, Any]) -> dict[str, Any]:
    """List configured taps.\n\nREQUIRES: nothing.\nRETURNS: {taps: [...]}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    return {"taps": tm.list_taps()}


def hub_tap_sync(args: dict[str, Any]) -> dict[str, Any]:
    """Pull latest from taps.\n\nREQUIRES: nothing (optional name).\nRETURNS: {synced, errors}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    try:
        return tm.sync_tap(args.get("name", ""))
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}


def hub_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search for plugins across all taps.\n\nREQUIRES: nothing (optional query, type).\nRETURNS: {results: [...]}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    results = tm.search(args.get("query", ""), plugin_type=args.get("type", ""))
    return {"results": results, "total": len(results)}


def hub_install(args: dict[str, Any]) -> dict[str, Any]:
    """Install a plugin from a tap.\n\nREQUIRES: name.\nRETURNS: {status, name, type, version, tap, path}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    try:
        return tm.install(args.get("name", ""), tap=args.get("tap", ""))
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}


def hub_uninstall(args: dict[str, Any]) -> dict[str, Any]:
    """Uninstall a tap plugin.\n\nREQUIRES: name.\nRETURNS: {removed: true}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    try:
        tm.uninstall(args.get("name", ""))
        return {"removed": True}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def hub_installed(args: dict[str, Any]) -> dict[str, Any]:
    """List installed tap plugins.\n\nREQUIRES: nothing (optional type).\nRETURNS: {packages: [...]}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    packages = tm.list_installed(plugin_type=args.get("type", ""))
    return {"packages": packages, "total": len(packages)}


def hub_publish(args: dict[str, Any]) -> dict[str, Any]:
    """Publish a local plugin to a tap repo.\n\nREQUIRES: name, type, tap.\nRETURNS: {status, path, next_steps}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    try:
        config = _get_config()
        return tm.publish(
            args.get("name", ""),
            args.get("type", ""),
            args.get("tap", ""),
            project_root=config.project_root,
            store_dir=config.store_dir,
            plugins_subdir=config.plugins.local_dir,
        )
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def hub_tap_push(args: dict[str, Any]) -> dict[str, Any]:
    """Push a tap repo to its remote.\n\nREQUIRES: tap name.\nRETURNS: {status, tap}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    try:
        return tm.push(args.get("tap", ""))
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}


def hub_submit_pr(args: dict[str, Any]) -> dict[str, Any]:
    """Open a PR from a tap fork to its upstream (requires gh CLI, falls back to manual instructions).\n\nREQUIRES: tap name.\nRETURNS: {status, pr_url | instructions}"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    try:
        return tm.submit_pr(
            args.get("tap", ""),
            title=args.get("title", ""),
            body=args.get("body", ""),
        )
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}


def hub_package_info(args: dict[str, Any]) -> dict[str, Any]:
    """Get detailed info about a package.\n\nREQUIRES: name.\nRETURNS: manifest + installed status"""
    tm = _get_tap_manager()
    if tm is None:
        return {"error": "Hub not initialized."}
    info = tm.get_package_info(args.get("name", ""))
    if info is None:
        return {"error": f"Package {args.get('name', '')!r} not found in any tap"}
    return info


# ---------------------------------------------------------------------------
# Trace tools
# ---------------------------------------------------------------------------


def trace_list(args: dict[str, Any]) -> dict[str, Any]:
    """List all session traces with metadata.

    REQUIRES: Nothing.
    RETURNS: {traces: [{session_id, started_at, ended_at, tool_calls, tool_counts, trace_file}]}
    """
    config = _get_config()
    trace_dir = config.project_root / config.store_dir / "traces"
    if not trace_dir.exists():
        return {"traces": []}

    import yaml

    traces = []
    for meta_path in sorted(trace_dir.glob("*.meta.yaml"), reverse=True):
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            if meta:
                traces.append(meta)
        except Exception:
            continue
    return {"traces": traces}


def trace_get(args: dict[str, Any]) -> dict[str, Any]:
    """Get the full trace entries for a session.

    REQUIRES: session_id.
    RETURNS: {session_id, entries: [...], meta: {...}} or {error}
    """
    session_id = args.get("session_id", "")
    if not session_id:
        return {"error": "session_id is required"}

    config = _get_config()
    trace_dir = config.project_root / config.store_dir / "traces"
    trace_path = trace_dir / f"{session_id}.jsonl"
    meta_path = trace_dir / f"{session_id}.meta.yaml"

    if not trace_path.exists():
        return {"error": f"Trace {session_id!r} not found"}

    from crucible.mcp.tracer import load_trace, load_trace_meta

    entries = load_trace(trace_path)
    meta = load_trace_meta(meta_path) if meta_path.exists() else None
    return {"session_id": session_id, "entries": entries, "meta": meta}


# ---------------------------------------------------------------------------
# Data tools
# ---------------------------------------------------------------------------


def data_list(args: dict[str, Any]) -> dict[str, Any]:
    """List registered data sources."""
    try:
        from crucible.core.data_sources import list_data_sources, describe_data_source

        sources = []
        for name in list_data_sources():
            info = describe_data_source(name)
            if info:
                sources.append(info)

        return {"sources": sources}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def data_status(args: dict[str, Any]) -> dict[str, Any]:
    """Check data status for a source."""
    try:
        from crucible.core.data_sources import build_data_source

        name = args.get("name")
        if not name:
            return {"error": "name is required"}

        source_type = args.get("type", "huggingface")
        config = args.get("config", {})

        source = build_data_source(source_type, name=name, config=config)
        status_result = source.status()

        return {
            "name": name,
            "status": status_result.status.value,
            "manifest": status_result.manifest,
            "shard_count": status_result.shard_count,
            "last_prepared": status_result.last_prepared.isoformat() if status_result.last_prepared else None,
            "issues": status_result.issues,
        }
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def data_prepare(args: dict[str, Any]) -> dict[str, Any]:
    """Prepare (download/cache) data for a source."""
    try:
        from crucible.core.data_sources import build_data_source

        name = args.get("name")
        if not name:
            return {"error": "name is required"}

        source_type = args.get("type", "huggingface")
        config = args.get("config", {})
        force = args.get("force", False)
        background = args.get("background", False)

        source = build_data_source(source_type, name=name, config=config)
        result = source.prepare(force=force, background=background)

        return {
            "success": result.success,
            "job_id": result.job_id,
            "message": result.message,
            "shards_downloaded": result.shards_downloaded,
        }
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def data_validate(args: dict[str, Any]) -> dict[str, Any]:
    """Validate data integrity."""
    try:
        from crucible.core.data_sources import build_data_source

        name = args.get("name")
        if not name:
            return {"error": "name is required"}

        source_type = args.get("type", "huggingface")
        config = args.get("config", {})

        source = build_data_source(source_type, name=name, config=config)
        result = source.validate()

        return {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
        }
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def data_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search for available data."""
    try:
        from crucible.core.data_sources import build_data_source

        query = args.get("query", "")
        source_type = args.get("type", "huggingface")

        source = build_data_source(source_type, name="_search", config={})
        results = source.search(query)

        return {
            "results": [
                {
                    "name": r.name,
                    "source": r.source,
                    "description": r.description,
                    "shard_count": r.shard_count,
                }
                for r in results
            ]
        }
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def data_link(args: dict[str, Any]) -> dict[str, Any]:
    """Link data source to an experiment run."""
    try:
        run_id = args.get("run_id")
        data_name = args.get("data_name")

        if not run_id or not data_name:
            return {"error": "run_id and data_name are required"}

        from pathlib import Path
        import json

        config = _get_config()
        store_path = config.project_root / ".crucible" / "store.jsonl"
        store_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "type": "data_link",
            "run_id": run_id,
            "data_name": data_name,
        }
        with open(store_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return {"linked": True, "run_id": run_id, "data_name": data_name}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def data_get_linked(args: dict[str, Any]) -> dict[str, Any]:
    """Get data linked to an experiment run."""
    try:
        run_id = args.get("run_id")
        if not run_id:
            return {"error": "run_id is required"}

        from pathlib import Path
        import json

        config = _get_config()
        store_path = config.project_root / ".crucible" / "store.jsonl"
        if not store_path.exists():
            return {"data_sources": []}

        links = []
        with open(store_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("type") == "data_link" and entry.get("run_id") == run_id:
                    links.append({"name": entry.get("data_name")})

        return {"data_sources": links}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ------------------------------------------------------------------
# Research DAG bridge tools
# ------------------------------------------------------------------


def _get_dag_bridge() -> Any:
    """Return a ResearchDAGBridge for the current project.

    Reads Spider Chat URL from persisted DAG state (set during init),
    with SPIDERCHAT_URL env var as override.
    """
    from crucible.research_dag.bridge import ResearchDAGBridge
    from crucible.research_dag.dag_state import DAGState
    config = _get_config()

    # Load persisted config to recover URL from init
    state = DAGState(config.project_root / ".crucible" / "research_dag")
    state.load()
    persisted_url = state.spiderchat_url

    # Env var overrides persisted URL
    spiderchat_url = os.environ.get("SPIDERCHAT_URL", "") or persisted_url
    spiderchat_token = os.environ.get("SPIDERCHAT_TOKEN", "")
    bridge = ResearchDAGBridge(
        project_dir=config.project_root,
        spiderchat_url=spiderchat_url,
        spiderchat_token=spiderchat_token,
    )
    bridge.load()
    return bridge


def research_dag_init(args: dict[str, Any]) -> dict[str, Any]:
    """Initialize research DAG bridge. Spider Chat is optional — works in local-only mode without it.

    RETURNS: Bridge status summary with mode ('connected' or 'local-only').
    NEXT: research_dag_push_node or research_dag_sync.
    """
    try:
        from crucible.research_dag.bridge import ResearchDAGBridge
        config = _get_config()
        # URL from args or env; token ONLY from env (never from args — prevents exfiltration)
        spiderchat_url = args.get("spiderchat_url", os.environ.get("SPIDERCHAT_URL", ""))
        spiderchat_token = os.environ.get("SPIDERCHAT_TOKEN", "")

        bridge = ResearchDAGBridge(
            project_dir=config.project_root,
            spiderchat_url=spiderchat_url,
            spiderchat_token=spiderchat_token,
        )
        result = bridge.init(
            flow_id=args.get("flow_id", ""),
            project_name=args.get("project_name", config.name),
        )
        return {"status": "initialized", **result}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def research_dag_sync(args: dict[str, Any]) -> dict[str, Any]:
    """Bidirectional sync between Crucible search tree and Spider Chat canvas.

    REQUIRES: research_dag_init completed, a search tree exists.
    RETURNS: Sync summary (pushed, updated, pulled counts, manual_hypotheses).
    NEXT: dispatch_experiments (for pulled manual hypotheses) or continue research.
    """
    try:
        config = _get_config()
        bridge = _get_dag_bridge()

        # Load tree
        tree_name = args.get("tree_name")
        tree_nodes: list[dict[str, Any]] = []
        best_metric: float | None = None
        primary_metric = config.metrics.primary if hasattr(config, "metrics") else "val_bpb"

        if tree_name:
            from crucible.researcher.search_tree import SearchTree
            tree_dir = _get_tree_dir(config, tree_name)
            tree = SearchTree.load(tree_dir)
            tree_nodes = list(tree.nodes.values())
            best_metric = tree.meta.get("best_metric")
            primary_metric = tree.meta.get("primary_metric", primary_metric)

        result = bridge.sync(
            tree_nodes=tree_nodes,
            flow_id=args.get("flow_id", ""),
            primary_metric=primary_metric,
            best_metric=best_metric,
        )
        return {"status": "synced", **result}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def research_dag_push_node(args: dict[str, Any]) -> dict[str, Any]:
    """Push a single Crucible experiment or hypothesis to Spider Chat canvas.

    REQUIRES: research_dag_init completed.
    RETURNS: canvas_node_id of the created/updated node.
    NEXT: Create edges or continue pushing nodes.
    """
    try:
        bridge = _get_dag_bridge()
        experiment = {
            "node_id": args.get("node_id", args.get("name", "")),
            "experiment_name": args.get("name", ""),
            "hypothesis": args.get("hypothesis", ""),
            "rationale": args.get("rationale", ""),
            "config": args.get("config", {}),
            "status": args.get("status", "pending"),
            "result": args.get("result"),
            "result_metric": args.get("result_metric"),
            "generation_method": args.get("generation_method", "manual"),
        }
        canvas_id = bridge.push_experiment_node(
            experiment=experiment,
            flow_id=args.get("flow_id", ""),
            parent_canvas_ids=args.get("parent_canvas_ids"),
            primary_metric=args.get("primary_metric", "val_bpb"),
            best_metric=args.get("best_metric"),
        )
        return {"status": "pushed", "canvas_node_id": canvas_id}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def research_dag_pull_manual(args: dict[str, Any]) -> dict[str, Any]:
    """Import manually-created Spider Chat canvas nodes as Crucible hypotheses.

    REQUIRES: research_dag_init completed, manual info nodes exist in flow.
    RETURNS: List of hypothesis dicts ready for Crucible experiment queue.
    NEXT: Enrich with configs, then enqueue_experiment or design_enqueue_batch.
    """
    try:
        bridge = _get_dag_bridge()
        hypotheses = bridge.pull_manual_nodes(flow_id=args.get("flow_id", ""))
        return {
            "status": "pulled",
            "count": len(hypotheses),
            "hypotheses": hypotheses,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def research_dag_status(args: dict[str, Any]) -> dict[str, Any]:
    """Show current research DAG bridge status and mapping summary.

    REQUIRES: research_dag_init completed.
    RETURNS: Mapping counts, status breakdown, flow_id.
    """
    try:
        bridge = _get_dag_bridge()
        return {"status": "ok", **bridge.status()}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


TOOL_DISPATCH: dict[str, Any] = {
    # Existing tools
    "get_fleet_status": get_fleet_status,
    "get_leaderboard": get_leaderboard,
    "get_queue_status": get_queue_status,
    "enqueue_experiment": enqueue_experiment,
    "get_experiment_result": get_experiment_result,
    "provision_nodes": provision_nodes,
    "destroy_nodes": destroy_nodes,
    "cleanup_orphans": cleanup_orphans,
    "stop_nodes": stop_nodes,
    "start_nodes": start_nodes,
    # RunPod enhanced operations (GraphQL)
    "runpod_list_volumes": runpod_list_volumes,
    "runpod_create_volume": runpod_create_volume,
    "runpod_delete_volume": runpod_delete_volume,
    "runpod_gpu_availability": runpod_gpu_availability,
    "runpod_list_templates": runpod_list_templates_tool,
    "runpod_create_template": runpod_create_template_tool,
    "sync_code": sync_code,
    "fleet_refresh": fleet_refresh,
    "bootstrap_nodes": bootstrap_nodes,
    "dispatch_experiments": dispatch_experiments,
    "collect_results": collect_results,
    "get_research_state": get_research_state,
    "get_sensitivity": get_sensitivity,
    # Design tools
    "design_browse_experiments": design_browse_experiments,
    "design_compare_experiments": design_compare_experiments,
    "design_generate_hypotheses": design_generate_hypotheses,
    "design_batch_from_hypotheses": design_batch_from_hypotheses,
    "design_enqueue_batch": design_enqueue_batch,
    # Context tools
    "context_get_analysis": context_get_analysis,
    "context_push_finding": context_push_finding,
    "context_get_findings": context_get_findings,
    # Version tools
    "version_save_design": version_save_design,
    "version_list_designs": version_list_designs,
    "version_diff": version_diff,
    "version_get_design": version_get_design,
    "version_run_design": version_run_design,
    "version_link_result": version_link_result,
    # Note tools
    "note_add": note_add,
    "note_get": note_get,
    "note_search": note_search,
    # W&B tools
    "wandb_log_image": wandb_log_image,
    "wandb_get_url": wandb_get_url,
    "wandb_annotate": wandb_annotate,
    # Hub tools
    "hub_status": hub_status,
    "hub_sync": hub_sync,
    "track_create": track_create,
    "track_list": track_list,
    "track_switch": track_switch,
    "hub_findings_query": hub_findings_query,
    "finding_promote": finding_promote,
    # Briefing tools
    "get_research_briefing": get_research_briefing,
    "annotate_run": annotate_run,
    # Literature search tools
    "research_literature_search": research_literature_search,
    # Model extensibility tools
    "model_list_families": model_list_families,
    "model_list_activations": model_list_activations,
    "model_list_components": model_list_components,
    "model_get_config_schema": model_get_config_schema,
    "model_validate_config": model_validate_config,
    "model_add_architecture": model_add_architecture,
    "model_add_activation": model_add_activation,
    "model_generate_template": model_generate_template,
    # Plugin promotion / import tools
    "model_list_global_architectures": model_list_global_architectures,
    "model_promote_architecture": model_promote_architecture,
    "model_import_architecture": model_import_architecture,
    # Composition tools
    "model_compose": model_compose,
    "model_from_template": model_from_template,
    "model_list_stack_patterns": model_list_stack_patterns,
    "model_list_block_types": model_list_block_types,
    "model_preview_spec": model_preview_spec,
    "model_get_spec": model_get_spec,
    # Queue management tools
    "cancel_experiment": cancel_experiment,
    "clear_stale_queue": clear_stale_queue,
    "purge_queue": purge_queue,
    # Config tools
    "config_get_presets": config_get_presets,
    "config_get_project": config_get_project,
    # New tools (Phase 1)
    "get_run_logs": get_run_logs,
    "model_fetch_architecture": model_fetch_architecture,
    "get_architecture_guide": get_architecture_guide,
    # Tree search tools (Phase 2)
    "tree_create": tree_create,
    "tree_get": tree_get,
    "tree_expand_node": tree_expand_node,
    "tree_auto_expand": tree_auto_expand,
    "tree_prune": tree_prune,
    "tree_enqueue_pending": tree_enqueue_pending,
    "tree_sync_results": tree_sync_results,
    "tree_list": tree_list,
    # Modalities tool (Phase 3)
    "config_get_modalities": config_get_modalities,
    # External project tools
    "list_projects": list_projects,
    "provision_project": provision_project,
    "bootstrap_project": bootstrap_project_tool,
    "run_project": run_project,
    "run_project_chain": run_project_chain,
    "collect_project_results": collect_project_results,
    "get_project_run_status": get_project_run_status,
    # Recipe tools
    "recipe_save": recipe_save,
    "recipe_list": recipe_list,
    "recipe_get": recipe_get,
    # Plugin registry tools (consolidated)
    "plugin_list": plugin_list,
    "plugin_add": plugin_add,
    "plugin_get_schema": plugin_get_schema,
    # Community tap tools
    "hub_tap_add": hub_tap_add,
    "hub_tap_remove": hub_tap_remove,
    "hub_tap_list": hub_tap_list,
    "hub_tap_sync": hub_tap_sync,
    "hub_search": hub_search,
    "hub_install": hub_install,
    "hub_uninstall": hub_uninstall,
    "hub_installed": hub_installed,
    "hub_publish": hub_publish,
    "hub_tap_push": hub_tap_push,
    "hub_submit_pr": hub_submit_pr,
    "hub_package_info": hub_package_info,
    # Trace tools
    "trace_list": trace_list,
    "trace_get": trace_get,
    # Data tools
    "data_list": data_list,
    "data_status": data_status,
    "data_prepare": data_prepare,
    "data_validate": data_validate,
    "data_search": data_search,
    "data_link": data_link,
    "data_get_linked": data_get_linked,
    # Research DAG bridge tools
    "research_dag_init": research_dag_init,
    "research_dag_sync": research_dag_sync,
    "research_dag_push_node": research_dag_push_node,
    "research_dag_pull_manual": research_dag_pull_manual,
    "research_dag_status": research_dag_status,
}
