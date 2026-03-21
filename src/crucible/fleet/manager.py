"""FleetManager: top-level orchestrator that ties all fleet modules together."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.errors import ConfigError
from crucible.core.io import atomic_write_json, read_jsonl
from crucible.core.log import log_error, log_info, log_step, log_success, log_warn, utc_now_iso
from crucible.fleet.bootstrap import (
    BOOTSTRAP_ATTEMPTS,
    bootstrap_node,
    bootstrap_node_worker,
    start_bootstrap_supervisor,
)
from crucible.fleet.day_run import (
    append_event,
    create_day_run_dir,
    day_tag,
    record_day_progress,
    render_day_status,
    update_day_summary,
    write_day_summary,
)
from crucible.fleet.inventory import (
    count_bootstrapped_ready,
    load_nodes,
    load_nodes_if_exists,
    load_nodes_snapshot,
    next_node_index,
    save_nodes,
    save_nodes_threadsafe,
    upsert_node_record,
)
from crucible.fleet.monitor import render_monitor, render_nodes
from crucible.fleet.provider import FleetProvider
from crucible.fleet.queue import (
    enqueue_experiments,
    load_queue,
    load_wave_spec,
    prepare_wave_experiments,
    reconcile_queue_with_results,
    reset_queue,
    results_by_id,
    save_queue,
)
from crucible.fleet.scheduler import (
    collect_from_node,
    dispatch,
    load_baseline_curve,
    merge_results,
    refresh_and_save_nodes,
    run_wave,
)


class FleetManager:
    """Top-level API that coordinates providers, inventory, queue, and scheduling.

    Usage::

        cfg = load_config()
        fm = FleetManager(cfg)
        nodes = fm.provision(count=4, name_prefix="crucible-day")
        fm.bootstrap(nodes)
        fm.enqueue(spec_path=Path("wave1.json"))
        fm.dispatch()
        fm.collect()
        fm.destroy()

    The manager can also run fully-automated day or night runs via
    :meth:`run_day` and :meth:`run_night`.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.project_root = config.project_root
        self.nodes_file = self.project_root / config.nodes_file
        self.queue_path = self.project_root / "fleet_queue.jsonl"
        self.fleet_results_file = self.project_root / config.fleet_results_file
        self.fleet_runs_dir = self.project_root / "fleet_runs"
        self.day_runs_dir = self.project_root / "day_runs"
        self.sync_excludes = list(config.sync_excludes)
        self.results_file_rel = config.remote_results_file
        self._provider: FleetProvider | None = None

    # ------------------------------------------------------------------
    # Provider construction (lazy)
    # ------------------------------------------------------------------

    @property
    def provider(self) -> FleetProvider:
        if self._provider is None:
            self._provider = self._build_provider(self.config)
        return self._provider

    @staticmethod
    def _build_provider(config: ProjectConfig) -> FleetProvider:
        ptype = config.provider.type.lower()
        if ptype == "runpod":
            from crucible.fleet.providers.runpod import RunPodProvider
            return RunPodProvider(
                image_name=config.provider.image or "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                gpu_type_ids=config.provider.gpu_types or None,
                ssh_key=config.provider.ssh_key,
                defaults=config.provider.defaults,
            )
        from crucible.fleet.providers.ssh import SSHProvider
        return SSHProvider(
            ssh_key=config.provider.ssh_key,
            defaults=config.provider.defaults,
        )

    # ------------------------------------------------------------------
    # Provision
    # ------------------------------------------------------------------

    def provision(
        self,
        *,
        count: int,
        name_prefix: str = "crucible",
        start_index: int = 1,
        replacement: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Provision *count* nodes via the configured provider and persist."""
        nodes = self.provider.provision(
            count=count,
            name_prefix=name_prefix,
            start_index=start_index,
            replacement=replacement,
            **kwargs,
        )
        save_nodes_threadsafe(self.nodes_file, nodes)
        return nodes

    def provision_and_wait(
        self,
        *,
        count: int,
        name_prefix: str = "crucible",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Provision and block until all nodes are SSH-reachable."""
        nodes = self.provision(count=count, name_prefix=name_prefix, **kwargs)
        nodes = self.provider.wait_ready(nodes)
        save_nodes_threadsafe(self.nodes_file, nodes)
        return nodes

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap(
        self,
        nodes: list[dict[str, Any]] | None = None,
        *,
        train_shards: int = 1,
        skip_install: bool = False,
        skip_data: bool = False,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Bootstrap one or more nodes (sync, install, data download)."""
        if nodes is None:
            nodes = load_nodes(self.nodes_file)
        updated: list[dict[str, Any]] = []
        for node in nodes:
            if selected_names and node["name"] not in selected_names:
                updated.append(node)
                continue
            updated.append(
                bootstrap_node(
                    node,
                    project_root=self.project_root,
                    sync_excludes=self.sync_excludes,
                    train_shards=train_shards,
                    skip_install=skip_install,
                    skip_data=skip_data,
                )
            )
        save_nodes(self.nodes_file, updated)
        return updated

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        *,
        spec_path: Path | None = None,
        experiments: list[dict[str, Any]] | None = None,
        backend: str = "torch",
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        """Enqueue experiments from a spec file or an explicit list."""
        if spec_path is not None:
            experiments = load_wave_spec(spec_path, default_backend=backend)
        if experiments is None:
            raise ValueError("Either spec_path or experiments must be provided.")
        return enqueue_experiments(self.queue_path, experiments, limit=limit)

    def reset_queue(self) -> None:
        """Clear the fleet queue."""
        reset_queue(self.queue_path)

    def queue_status(self) -> list[dict[str, Any]]:
        """Return the current queue contents."""
        return load_queue(self.queue_path)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        nodes: list[dict[str, Any]] | None = None,
        *,
        max_assignments: int = 8,
    ) -> list[dict[str, Any]]:
        """Dispatch queued experiments to idle ready nodes."""
        if nodes is None:
            nodes = load_nodes_snapshot(self.nodes_file)
        queue = load_queue(self.queue_path)
        return dispatch(
            nodes, queue,
            queue_path=self.queue_path,
            max_assignments=max_assignments,
            run_script=self.config.runner_script,
            timeout_map=self.config.timeout_map,
        )

    # ------------------------------------------------------------------
    # Collect
    # ------------------------------------------------------------------

    def collect(
        self,
        nodes: list[dict[str, Any]] | None = None,
    ) -> None:
        """Rsync results from all nodes and merge into fleet results."""
        if nodes is None:
            nodes = load_nodes(self.nodes_file)
        for node in nodes:
            collect_from_node(
                node,
                fleet_runs_dir=self.fleet_runs_dir,
                results_file_rel=self.results_file_rel,
            )
        merge_results(self.fleet_runs_dir, self.fleet_results_file)

    # ------------------------------------------------------------------
    # Destroy
    # ------------------------------------------------------------------

    def destroy(
        self,
        nodes: list[dict[str, Any]] | None = None,
        *,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Destroy nodes via the provider and update the nodes file."""
        if nodes is None:
            nodes = load_nodes_if_exists(self.nodes_file)
        remaining = self.provider.destroy(nodes, selected_names=selected_names)
        save_nodes(self.nodes_file, remaining)
        return remaining

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Return a human-readable status string."""
        nodes = load_nodes_if_exists(self.nodes_file)
        if not nodes:
            return "No nodes registered."
        return render_nodes(nodes)

    def monitor(
        self,
        nodes: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return a fleet-wide monitor table (probes nodes via SSH)."""
        if nodes is None:
            nodes = load_nodes_if_exists(self.nodes_file)
        return render_monitor(nodes)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(
        self,
        nodes: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Refresh node records from the provider API and save."""
        if nodes is None:
            nodes = load_nodes(self.nodes_file)
        refreshed = self.provider.refresh(nodes)
        save_nodes(self.nodes_file, refreshed)
        return refreshed

    # ------------------------------------------------------------------
    # Replacement helpers
    # ------------------------------------------------------------------

    def _replace_stalled_nodes(
        self,
        *,
        nodes: list[dict[str, Any]],
        selected_names: set[str],
        name_prefix: str,
        stage_label: str,
    ) -> list[dict[str, Any]]:
        """Destroy stalled nodes and provision replacements."""
        if not selected_names:
            return nodes
        log_warn(
            f"{stage_label}: replacing stalled nodes: "
            f"{', '.join(sorted(selected_names))}"
        )
        survivors = self.provider.destroy(nodes, selected_names=selected_names)
        start_index = next_node_index(nodes, name_prefix)
        replacements = self.provider.provision(
            count=len(selected_names),
            name_prefix=name_prefix,
            start_index=start_index,
            replacement=True,
        )
        updated = survivors + replacements
        save_nodes_threadsafe(self.nodes_file, updated)
        return updated

    def _replace_bootstrap_node(
        self,
        failed_name: str,
        *,
        name_prefix: str,
    ) -> dict[str, Any]:
        """Replace a single failed node and wait until ready."""
        nodes = load_nodes_snapshot(self.nodes_file)
        old_names = {n["name"] for n in nodes}
        updated = self._replace_stalled_nodes(
            nodes=nodes,
            selected_names={failed_name},
            name_prefix=name_prefix,
            stage_label="bootstrap_recovery",
        )
        replacements = [n for n in updated if n["name"] not in old_names]
        if len(replacements) != 1:
            raise RuntimeError(
                f"Expected one replacement for {failed_name}, found {len(replacements)}",
            )
        replacement = replacements[0]
        replacement = self.provider.wait_ready(
            [replacement], timeout_seconds=900, poll_seconds=15,
        )[0]
        upsert_node_record(self.nodes_file, replacement)
        return replacement

    def _provision_replacement_nodes(
        self,
        *,
        existing_nodes: list[dict[str, Any]],
        needed: int,
        name_prefix: str,
        train_shards: int,
    ) -> list[dict[str, Any]]:
        """Provision, wait, and bootstrap replacement nodes."""
        if needed <= 0:
            return []
        start_index = next_node_index(existing_nodes, name_prefix)
        replacements = self.provider.provision(
            count=needed,
            name_prefix=name_prefix,
            start_index=start_index,
            replacement=True,
        )
        all_nodes = existing_nodes + replacements
        save_nodes(self.nodes_file, all_nodes)
        replacements = self.provider.wait_ready(
            replacements, timeout_seconds=900, poll_seconds=15,
        )
        bootstrapped: list[dict[str, Any]] = []
        for node in replacements:
            log_step(f"bootstrap {node['name']}: bootstrap replacement")
            bootstrapped.append(
                bootstrap_node(
                    node,
                    project_root=self.project_root,
                    sync_excludes=self.sync_excludes,
                    train_shards=train_shards,
                    skip_install=False,
                    skip_data=False,
                )
            )
            log_success(f"{node['name']}: replacement ready")
        return bootstrapped

    # ------------------------------------------------------------------
    # run_day -- full multi-wave automated day run
    # ------------------------------------------------------------------

    def run_day(
        self,
        *,
        count: int,
        name_prefix: str = "crucible-day",
        train_shards: int = 1,
        monitor_interval: int = 30,
        destroy_on_exit: bool = True,
        keep_on_failure: bool = False,
        dry_run: bool = False,
        stop_after: str | None = None,
        min_ready_to_start: int = 2,
        bootstrap_workers: int = 8,
        bootstrap_replacement_budget: int = 8,
        wave_specs: list[tuple[str, Path]] | None = None,
        wave_completion_thresholds: dict[str, int] | None = None,
    ) -> Path:
        """Run the full multi-wave day automatically.

        Parameters
        ----------
        wave_specs : list of (wave_name, spec_path) tuples
            The waves to run in order.
        wave_completion_thresholds : dict mapping wave names to min completed
        """
        day_dir = create_day_run_dir(self.day_runs_dir)
        thresholds = wave_completion_thresholds or {}
        summary: dict[str, Any] = {
            "started_at": utc_now_iso(),
            "count": count,
            "name_prefix": name_prefix,
            "train_shards": train_shards,
            "monitor_interval": monitor_interval,
            "destroy_on_exit": destroy_on_exit,
            "keep_on_failure": keep_on_failure,
            "dry_run": dry_run,
            "stop_after": stop_after,
            "day_tag": day_tag(day_dir),
            "summary_path": str((day_dir / "summary.json").resolve()),
            "events_path": str((day_dir / "events.jsonl").resolve()),
            "leaderboard_path": str((day_dir / "leaderboard.md").resolve()),
            "phase": "starting",
            "current_wave": None,
            "wave_order": [w[0] for w in (wave_specs or [])],
            "nodes_total": 0,
            "nodes_ready": 0,
            "nodes_bootstrapped": 0,
            "nodes_failed": 0,
            "nodes_healthy": 0,
            "nodes_unhealthy": 0,
            "nodes_replaced": 0,
            "queue_total": 0,
            "queue_running": 0,
            "queue_finished": 0,
            "queue_completed": 0,
            "retryable_runs": 0,
            "failed_retry_budget": 0,
            "replacements_used": 0,
            "min_ready_to_start": min_ready_to_start,
            "bootstrap_workers": bootstrap_workers,
            "bootstrap_replacement_budget": bootstrap_replacement_budget,
            "bootstrap_ready_count": 0,
            "wave1_started": False,
            "last_event": "day_started",
            "waves": [],
            "status": "running",
        }
        write_day_summary(day_dir, summary)
        append_event(day_dir, "day_started")
        log_info(f"Day run started: {day_dir}")
        log_info(f"Summary: {summary['summary_path']}")

        if dry_run:
            for wave_name, path in (wave_specs or []):
                append_event(day_dir, "wave_planned", wave=wave_name, spec=str(path))
            summary["status"] = "dry_run"
            update_day_summary(
                day_dir, summary, phase="dry_run", last_event="dry_run_ready",
            )
            log_info(f"Dry run planned: {[w[0] for w in (wave_specs or [])]}")
            return day_dir

        nodes: list[dict[str, Any]] = []
        recovery: dict[str, Any] = {
            "max_attempts_per_run": 3,
            "max_replacements": count,
            "target_total_nodes": count,
            "replacements_used": 0,
            "train_shards": train_shards,
        }

        try:
            self.reset_queue()
            append_event(day_dir, "queue_reset")
            record_day_progress(
                day_dir, summary, phase="queue_reset", last_event="queue_reset",
            )
            log_step("Fleet queue reset")

            record_day_progress(
                day_dir, summary, phase="provisioning", last_event="provisioning",
            )
            log_step(f"Creating {count} nodes")
            nodes = self.provision(count=count, name_prefix=name_prefix)
            append_event(day_dir, "nodes_provisioned", count=len(nodes))
            record_day_progress(
                day_dir, summary,
                phase="nodes_created", last_event="nodes_provisioned", nodes=nodes,
            )
            atomic_write_json(day_dir / "node_snapshot.json", {"nodes": nodes})
            log_success(f"Nodes created: {len(nodes)}/{count}")

            # Prepare waves from specs
            extra_tags = [day_tag(day_dir)]
            prepared_waves: list[tuple[str, list[dict[str, Any]]]] = []
            for wave_name, path in (wave_specs or []):
                experiments = prepare_wave_experiments(
                    load_wave_spec(path), extra_tags=extra_tags,
                )
                for exp in experiments:
                    exp["wave"] = wave_name
                prepared_waves.append((wave_name, experiments))

            # Bootstrap
            min_ready = max(1, min(count, min_ready_to_start))
            record_day_progress(
                day_dir, summary,
                phase="bootstrapping", last_event="bootstrap_started", nodes=nodes,
            )
            log_step(
                f"Starting parallel bootstrap; first wave starts "
                f"after {min_ready} ready node(s)"
            )
            bootstrap_state = start_bootstrap_supervisor(
                day_dir=day_dir,
                nodes=nodes,
                nodes_file=self.nodes_file,
                project_root=self.project_root,
                sync_excludes=self.sync_excludes,
                train_shards=train_shards,
                target_ready_count=count,
                min_ready_to_start=min_ready,
                bootstrap_workers=max(1, min(bootstrap_workers, count)),
                replacement_budget=bootstrap_replacement_budget,
                replace_fn=lambda name: self._replace_bootstrap_node(
                    name, name_prefix=name_prefix,
                ),
                refresh_fn=self.provider.refresh,
            )
            accounted_bootstrap_replacements = 0
            launch_ready_recorded = False

            while True:
                current_nodes = load_nodes_snapshot(self.nodes_file) or nodes
                current_ready = count_bootstrapped_ready(current_nodes)
                bootstrap_state["ready_count"] = current_ready
                summary["replacements_used"] = (
                    recovery["replacements_used"]
                    + bootstrap_state["replacements_used"]
                )
                if current_ready >= min_ready:
                    bootstrap_state["min_ready_event"].set()
                    if not launch_ready_recorded:
                        append_event(
                            day_dir, "wave1_launch_ready",
                            ready_count=current_ready, required=min_ready,
                        )
                        launch_ready_recorded = True
                    break
                record_day_progress(
                    day_dir, summary,
                    phase="bootstrapping", last_event="bootstrap_waiting",
                    nodes=current_nodes,
                )
                if (
                    bootstrap_state["blocking_error"] is not None
                    and current_ready < min_ready
                ):
                    raise bootstrap_state["blocking_error"]
                log_info(f"Bootstrapped ready {bootstrap_state['ready_count']}/{count}")
                bootstrap_state["min_ready_event"].wait(timeout=5)

            recovery["replacements_used"] += bootstrap_state["replacements_used"]
            accounted_bootstrap_replacements = bootstrap_state["replacements_used"]
            summary["replacements_used"] = recovery["replacements_used"]
            nodes = load_nodes_snapshot(self.nodes_file) or nodes
            record_day_progress(
                day_dir, summary,
                phase="wave1_ready", last_event="bootstrap_threshold_met",
                nodes=nodes,
            )
            log_success(
                f"Bootstrap threshold reached: "
                f"{bootstrap_state['ready_count']}/{count} ready"
            )

            # Helper for provision_replacement_fn used by run_wave
            def _provision_replacements(
                *, existing_nodes: list[dict[str, Any]], needed: int,
            ) -> list[dict[str, Any]]:
                return self._provision_replacement_nodes(
                    existing_nodes=existing_nodes,
                    needed=needed,
                    name_prefix=name_prefix,
                    train_shards=train_shards,
                )

            # Run first wave immediately (bootstrap continues in background)
            if prepared_waves:
                first_wave_name, first_wave_experiments = prepared_waves[0]
                bootstrap_state["wave1_started"] = True
                append_event(
                    day_dir, "wave1_first_dispatch",
                    ready_count=bootstrap_state["ready_count"],
                )
                update_day_summary(
                    day_dir, summary,
                    wave1_started=True,
                    bootstrap_ready_count=bootstrap_state["ready_count"],
                )
                results, nodes = run_wave(
                    day_dir=day_dir,
                    summary=summary,
                    wave_name=first_wave_name,
                    experiments=first_wave_experiments,
                    nodes=nodes,
                    nodes_file=self.nodes_file,
                    queue_path=self.queue_path,
                    fleet_runs_dir=self.fleet_runs_dir,
                    fleet_results_file=self.fleet_results_file,
                    project_root=self.project_root,
                    sync_excludes=self.sync_excludes,
                    dispatch_limit=count,
                    monitor_interval=monitor_interval,
                    min_completed=thresholds.get(first_wave_name, 1),
                    recovery=recovery,
                    refresh_fn=self.provider.refresh,
                    provision_replacement_fn=_provision_replacements,
                    results_file_rel=self.results_file_rel,
                    run_script=self.config.runner_script,
                    timeout_map=self.config.timeout_map,
                )
                summary["waves"].append({
                    "name": first_wave_name,
                    "completed": len([
                        r for r in results if r.get("status") == "completed"
                    ]),
                })
                write_day_summary(day_dir, summary)
                if stop_after == first_wave_name:
                    summary["status"] = "stopped"
                    write_day_summary(day_dir, summary)
                    return day_dir

            # Wait for bootstrap to finish
            bootstrap_state["done_event"].wait()
            bootstrap_state["thread"].join(timeout=1)
            if bootstrap_state["blocking_error"] is not None:
                raise bootstrap_state["blocking_error"]
            if bootstrap_state["degraded_error"] is not None:
                log_warn(
                    f"Bootstrap degraded after wave1 start: "
                    f"{bootstrap_state['degraded_error']}"
                )
                append_event(
                    day_dir, "bootstrap_degraded_after_start",
                    error=str(bootstrap_state["degraded_error"]),
                )
            recovery["replacements_used"] += max(
                0,
                bootstrap_state["replacements_used"]
                - accounted_bootstrap_replacements,
            )
            summary["replacements_used"] = recovery["replacements_used"]
            nodes = load_nodes_snapshot(self.nodes_file) or nodes
            record_day_progress(
                day_dir, summary,
                phase="bootstrapped", last_event="bootstrap_complete", nodes=nodes,
            )

            # Run remaining waves
            for wave_name, experiments in prepared_waves[1:]:
                results, nodes = run_wave(
                    day_dir=day_dir,
                    summary=summary,
                    wave_name=wave_name,
                    experiments=experiments,
                    nodes=nodes,
                    nodes_file=self.nodes_file,
                    queue_path=self.queue_path,
                    fleet_runs_dir=self.fleet_runs_dir,
                    fleet_results_file=self.fleet_results_file,
                    project_root=self.project_root,
                    sync_excludes=self.sync_excludes,
                    dispatch_limit=count,
                    monitor_interval=monitor_interval,
                    min_completed=thresholds.get(wave_name, 1),
                    recovery=recovery,
                    refresh_fn=self.provider.refresh,
                    provision_replacement_fn=_provision_replacements,
                    results_file_rel=self.results_file_rel,
                    run_script=self.config.runner_script,
                    timeout_map=self.config.timeout_map,
                )
                summary["waves"].append({
                    "name": wave_name,
                    "completed": len([
                        r for r in results if r.get("status") == "completed"
                    ]),
                })
                write_day_summary(day_dir, summary)
                if stop_after == wave_name:
                    summary["status"] = "stopped"
                    write_day_summary(day_dir, summary)
                    return day_dir

            summary["status"] = "completed"
            summary["finished_at"] = utc_now_iso()
            update_day_summary(
                day_dir, summary,
                phase="completed", current_wave=None, last_event="day_completed",
            )
            append_event(day_dir, "day_completed")
            log_success("Day run complete")
            return day_dir

        except BaseException as exc:
            summary["status"] = "failed"
            summary["finished_at"] = utc_now_iso()
            summary["error"] = str(exc)
            update_day_summary(
                day_dir, summary, phase="failed", last_event="day_failed",
            )
            append_event(day_dir, "day_failed", error=str(exc))
            log_error(str(exc))
            if keep_on_failure:
                return day_dir
            raise
        finally:
            if (
                nodes
                and destroy_on_exit
                and not (summary.get("status") == "failed" and keep_on_failure)
            ):
                record_day_progress(
                    day_dir, summary,
                    phase="teardown", last_event="nodes_destroying", nodes=nodes,
                )
                log_step(f"Destroying {len(nodes)} nodes")
                remaining = self.provider.destroy(nodes)
                save_nodes(self.nodes_file, remaining)
                append_event(day_dir, "nodes_destroyed", count=len(nodes))
                record_day_progress(
                    day_dir, summary,
                    phase="teardown_complete", last_event="nodes_destroyed",
                    nodes=remaining,
                )
                log_success("Nodes destroyed")

    # ------------------------------------------------------------------
    # run_night -- overnight batch run
    # ------------------------------------------------------------------

    def run_night(
        self,
        *,
        count: int,
        name_prefix: str = "crucible-night",
        train_shards: int = 1,
        monitor_interval: int = 30,
        destroy_on_exit: bool = True,
        keep_on_failure: bool = False,
        dry_run: bool = False,
        fresh: bool = True,
        spec_path: Path,
    ) -> Path:
        """Run a fresh overnight batch from a single spec file."""
        day_dir = create_day_run_dir(self.day_runs_dir)
        summary: dict[str, Any] = {
            "started_at": utc_now_iso(),
            "count": count,
            "name_prefix": name_prefix,
            "train_shards": train_shards,
            "monitor_interval": monitor_interval,
            "destroy_on_exit": destroy_on_exit,
            "keep_on_failure": keep_on_failure,
            "dry_run": dry_run,
            "fresh": fresh,
            "spec_path": str(spec_path),
            "day_tag": day_tag(day_dir),
            "summary_path": str((day_dir / "summary.json").resolve()),
            "events_path": str((day_dir / "events.jsonl").resolve()),
            "leaderboard_path": str((day_dir / "leaderboard.md").resolve()),
            "phase": "starting",
            "current_wave": None,
            "wave_order": ["night"],
            "nodes_total": 0,
            "nodes_ready": 0,
            "nodes_bootstrapped": 0,
            "nodes_failed": 0,
            "nodes_healthy": 0,
            "nodes_unhealthy": 0,
            "nodes_replaced": 0,
            "queue_total": 0,
            "queue_running": 0,
            "queue_finished": 0,
            "queue_completed": 0,
            "retryable_runs": 0,
            "failed_retry_budget": 0,
            "replacements_used": 0,
            "min_ready_to_start": count,
            "bootstrap_workers": count,
            "bootstrap_replacement_budget": count,
            "bootstrap_ready_count": 0,
            "wave1_started": False,
            "last_event": "night_started",
            "waves": [],
            "status": "running",
        }
        write_day_summary(day_dir, summary)
        append_event(day_dir, "night_started")
        log_info(f"Night run started: {day_dir}")
        log_info(f"Summary: {summary['summary_path']}")

        experiments = prepare_wave_experiments(
            load_wave_spec(spec_path), extra_tags=[day_tag(day_dir)],
        )
        for exp in experiments:
            exp["wave"] = "night"

        if dry_run:
            append_event(
                day_dir, "wave_planned",
                wave="night", spec=str(spec_path), experiments=len(experiments),
            )
            summary["status"] = "dry_run"
            update_day_summary(
                day_dir, summary, phase="dry_run", last_event="dry_run_ready",
            )
            log_info(f"Dry run planned: night ({len(experiments)} runs)")
            return day_dir

        nodes: list[dict[str, Any]] = []
        recovery: dict[str, Any] = {
            "max_attempts_per_run": 3,
            "max_replacements": count,
            "target_total_nodes": count,
            "replacements_used": 0,
            "train_shards": train_shards,
        }

        try:
            if fresh:
                record_day_progress(
                    day_dir, summary, phase="cleanup", last_event="cleanup_started",
                )
                tracked = load_nodes_if_exists(self.nodes_file)
                if tracked:
                    log_step(f"Destroying {len(tracked)} tracked nodes")
                    self.provider.destroy(tracked)
                # RunPod-specific stale cleanup
                destroyed_stale = 0
                from crucible.fleet.providers.runpod import RunPodProvider as _RP
                if isinstance(self.provider, _RP):
                    from crucible.fleet.providers.runpod import destroy_stale_named_pods
                    destroyed_stale = destroy_stale_named_pods(
                        prefixes=[name_prefix, "crucible-day"],
                    )
                save_nodes(self.nodes_file, [])
                append_event(
                    day_dir, "cleanup_complete",
                    tracked=len(tracked), stale=destroyed_stale,
                )
                log_success(
                    f"Cleanup complete: tracked={len(tracked)} stale={destroyed_stale}"
                )

            self.reset_queue()
            append_event(day_dir, "queue_reset")
            record_day_progress(
                day_dir, summary, phase="queue_reset", last_event="queue_reset",
            )
            log_step("Fleet queue reset")

            record_day_progress(
                day_dir, summary, phase="provisioning", last_event="provisioning",
            )
            log_step(f"Creating {count} nodes")
            nodes = self.provision_and_wait(count=count, name_prefix=name_prefix)
            append_event(day_dir, "nodes_provisioned", count=len(nodes))
            record_day_progress(
                day_dir, summary,
                phase="nodes_ready_for_bootstrap",
                last_event="nodes_provisioned",
                nodes=nodes,
            )
            atomic_write_json(day_dir / "node_snapshot.json", {"nodes": nodes})
            log_success(f"Nodes ready for bootstrap: {len(nodes)}/{count}")

            # Bootstrap all nodes before starting night wave
            record_day_progress(
                day_dir, summary,
                phase="bootstrapping", last_event="bootstrap_started",
                nodes=nodes,
            )
            log_step(f"Starting overnight bootstrap for {count} node(s)")
            bootstrap_state = start_bootstrap_supervisor(
                day_dir=day_dir,
                nodes=nodes,
                nodes_file=self.nodes_file,
                project_root=self.project_root,
                sync_excludes=self.sync_excludes,
                train_shards=train_shards,
                target_ready_count=count,
                min_ready_to_start=count,
                bootstrap_workers=max(1, count),
                replacement_budget=count,
                replace_fn=lambda name: self._replace_bootstrap_node(
                    name, name_prefix=name_prefix,
                ),
                refresh_fn=self.provider.refresh,
            )

            while True:
                current_nodes = load_nodes_snapshot(self.nodes_file) or nodes
                current_ready = count_bootstrapped_ready(current_nodes)
                bootstrap_state["ready_count"] = current_ready
                summary["replacements_used"] = (
                    recovery["replacements_used"]
                    + bootstrap_state["replacements_used"]
                )
                if current_ready >= count:
                    break
                record_day_progress(
                    day_dir, summary,
                    phase="bootstrapping", last_event="bootstrap_waiting",
                    nodes=current_nodes,
                )
                if bootstrap_state["blocking_error"] is not None:
                    raise bootstrap_state["blocking_error"]
                log_info(f"Bootstrapped ready {current_ready}/{count}")
                bootstrap_state["min_ready_event"].wait(timeout=5)

            bootstrap_state["done_event"].wait()
            bootstrap_state["thread"].join(timeout=1)
            if bootstrap_state["blocking_error"] is not None:
                raise bootstrap_state["blocking_error"]
            if bootstrap_state["degraded_error"] is not None:
                raise bootstrap_state["degraded_error"]
            recovery["replacements_used"] += bootstrap_state["replacements_used"]
            summary["replacements_used"] = recovery["replacements_used"]
            nodes = load_nodes_snapshot(self.nodes_file) or nodes
            record_day_progress(
                day_dir, summary,
                phase="night_ready", last_event="bootstrap_complete", nodes=nodes,
            )
            log_success(
                f"Overnight fleet ready: {count_bootstrapped_ready(nodes)}/{count}"
            )

            update_day_summary(
                day_dir, summary,
                wave1_started=True,
                bootstrap_ready_count=count_bootstrapped_ready(nodes),
            )
            baseline_curve = load_baseline_curve(self.fleet_runs_dir)
            if baseline_curve:
                log_info(
                    f"Early stopping enabled: {len(baseline_curve)} baseline data "
                    f"points, threshold=6000 steps, margin=5%"
                )
            else:
                log_warn(
                    "Early stopping disabled: no baseline curve found in fleet_runs/"
                )

            def _provision_replacements(
                *, existing_nodes: list[dict[str, Any]], needed: int,
            ) -> list[dict[str, Any]]:
                return self._provision_replacement_nodes(
                    existing_nodes=existing_nodes,
                    needed=needed,
                    name_prefix=name_prefix,
                    train_shards=train_shards,
                )

            results, nodes = run_wave(
                day_dir=day_dir,
                summary=summary,
                wave_name="night",
                experiments=experiments,
                nodes=nodes,
                nodes_file=self.nodes_file,
                queue_path=self.queue_path,
                fleet_runs_dir=self.fleet_runs_dir,
                fleet_results_file=self.fleet_results_file,
                project_root=self.project_root,
                sync_excludes=self.sync_excludes,
                dispatch_limit=count,
                monitor_interval=monitor_interval,
                min_completed=1,
                recovery=recovery,
                refresh_fn=self.provider.refresh,
                provision_replacement_fn=_provision_replacements,
                baseline_curve=baseline_curve or None,
                results_file_rel=self.results_file_rel,
            )
            summary["waves"].append({
                "name": "night",
                "completed": len([
                    r for r in results if r.get("status") == "completed"
                ]),
            })
            summary["status"] = "completed"
            summary["finished_at"] = utc_now_iso()
            update_day_summary(
                day_dir, summary,
                phase="completed", current_wave=None, last_event="night_completed",
            )
            append_event(day_dir, "night_completed")
            log_success("Night run complete")
            return day_dir

        except BaseException as exc:
            summary["status"] = "failed"
            summary["finished_at"] = utc_now_iso()
            summary["error"] = str(exc)
            update_day_summary(
                day_dir, summary, phase="failed", last_event="night_failed",
            )
            append_event(day_dir, "night_failed", error=str(exc))
            log_error(str(exc))
            if keep_on_failure:
                return day_dir
            raise
        finally:
            if (
                nodes
                and destroy_on_exit
                and not (summary.get("status") == "failed" and keep_on_failure)
            ):
                record_day_progress(
                    day_dir, summary,
                    phase="teardown", last_event="nodes_destroying", nodes=nodes,
                )
                log_step(f"Destroying {len(nodes)} nodes")
                remaining = self.provider.destroy(nodes)
                save_nodes(self.nodes_file, remaining)
                append_event(day_dir, "nodes_destroyed", count=len(nodes))
                record_day_progress(
                    day_dir, summary,
                    phase="teardown_complete", last_event="nodes_destroyed",
                    nodes=remaining,
                )
                log_success("Nodes destroyed")

    # ------------------------------------------------------------------
    # watch_day
    # ------------------------------------------------------------------

    def watch_day(self, day_dir: Path, *, watch_seconds: int = 10) -> None:
        """Watch a day run summary, refreshing periodically."""
        summary_path = day_dir / "summary.json"
        if not summary_path.exists():
            raise ConfigError(f"summary.json not found in {day_dir}")
        while True:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(render_day_status(summary), flush=True)
            nodes = load_nodes_if_exists(self.nodes_file)
            if nodes and int(summary.get("nodes_total", 0)) > 0:
                print(render_monitor(nodes), flush=True)
            status = summary.get("status")
            if watch_seconds <= 0 or status in {
                "completed", "failed", "dry_run", "stopped",
            }:
                return
            print("", flush=True)
            time.sleep(watch_seconds)
