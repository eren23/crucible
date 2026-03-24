"""Wave-based scheduling, dispatch assignments, early stopping, wave completion."""
from __future__ import annotations

import json
import re
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crucible.core.errors import RunnerError
from crucible.core.io import read_jsonl
from crucible.core.log import log_error, log_info, log_step, log_success, log_warn, utc_now_iso
from crucible.fleet.bootstrap import bootstrap_node, bootstrap_node_worker
from crucible.fleet.day_run import append_event, record_day_progress, write_day_leaderboard
from crucible.fleet.inventory import (
    BAD_API_STATES,
    NODES_LOCK,
    classify_health,
    count_bootstrapped_ready,
    load_nodes_snapshot,
    ready_state,
    save_nodes,
    upsert_node_record,
)
from crucible.fleet.monitor import probe_node_status, render_monitor
from crucible.fleet.queue import (
    load_queue,
    reconcile_queue_with_results,
    results_by_id,
    save_queue,
    summarize_idle_capacity,
    summarize_queue,
    wave_result_rows,
    wave_rows,
)
from crucible.fleet.sync import remote_exec

RECOVERABLE_LEASE_STATES = {"queued", "retryable", "running"}
REMOTE_LOG_DIR = "logs"


# ---------------------------------------------------------------------------
# Experiment launcher
# ---------------------------------------------------------------------------

def launch_experiment(
    node: dict[str, Any],
    item: dict[str, Any],
    *,
    run_script: str = "crucible/runner/run_experiment.py",
    timeout_map: dict[str, dict[str, int]] | None = None,
) -> str:
    """Launch an experiment on a remote node.  Returns the remote PID."""
    workspace = node.get("workspace_path", "/workspace/project")
    py = node.get("python_bin", "python3")

    # Determine timeout
    timeout = 1800  # default 30 min
    if timeout_map:
        backend_map = timeout_map.get(item["backend"], {})
        timeout = backend_map.get(item["tier"], timeout)

    cmd = [
        py, run_script,
        "--backend", item["backend"],
        "--preset", item["tier"],
        "--timeout", str(timeout),
        "--name", item["experiment_name"],
        "--experiment-id", item["run_id"],
    ]
    for tag in item.get("tags", []):
        cmd += ["--tag", tag]
    for key, value in item["config"].items():
        cmd += ["--set", f"{key}={value}"]

    launcher_log = f"{REMOTE_LOG_DIR}/{item['run_id']}.launcher.txt"
    launcher = (
        f"cd {shlex.quote(workspace)} && {shlex.quote(py)} - <<'PY'\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        f"log_path = Path({launcher_log!r})\n"
        "log_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "with log_path.open('a', encoding='utf-8') as log:\n"
        f"    proc = subprocess.Popen({cmd!r}, stdout=log, stderr=subprocess.STDOUT, "
        "stdin=subprocess.DEVNULL, start_new_session=True)\n"
        "print(proc.pid)\n"
        "PY"
    )
    return remote_exec(node, launcher).stdout.strip()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(
    nodes: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    *,
    queue_path: Path,
    max_assignments: int,
    run_script: str = "crucible/runner/run_experiment.py",
    timeout_map: dict[str, dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Assign queued (or retryable) runs to idle ready nodes.

    Returns the updated queue.
    """
    idle_nodes = [n for n in nodes if ready_state(n) == "ready"]
    assignments = 0
    dispatchable = [i for i in queue if i.get("lease_state") in {"queued", "retryable"}]
    dispatchable.sort(key=lambda x: (-x.get("priority", 0), x.get("created_at", "")))
    for item in dispatchable:
        if assignments >= max_assignments:
            break
        active_names = {
            row.get("assigned_node") or row.get("assigned_pod")
            for row in queue
            if row.get("lease_state") == "running"
        }
        node = next((n for n in idle_nodes if n["name"] not in active_names), None)
        if node is None:
            break
        pid = launch_experiment(
            node, item, run_script=run_script, timeout_map=timeout_map,
        )
        item["assigned_node"] = node["name"]
        item["lease_state"] = "running"
        item["attempt"] = int(item.get("attempt", 0)) + 1
        item["started_at"] = utc_now_iso()
        item["remote_pid"] = pid
        assignments += 1
    if assignments:
        save_queue(queue_path, queue)
    return queue


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def collect_from_node(
    node: dict[str, Any],
    *,
    fleet_runs_dir: Path,
    results_file_rel: str = "experiments.jsonl",
) -> None:
    """Rsync logs and results from a remote node."""
    from crucible.fleet.sync import rsync_base, _run

    local_dir = fleet_runs_dir / node["name"]
    local_dir.mkdir(parents=True, exist_ok=True)
    workspace = node.get("workspace_path", "/workspace/project")
    remote_logs = f"{node.get('user', 'root')}@{node['ssh_host']}:{workspace}/logs/"
    remote_results = (
        f"{node.get('user', 'root')}@{node['ssh_host']}:{workspace}/{results_file_rel}"
    )
    _run(rsync_base(node) + [remote_logs, str(local_dir / "logs") + "/"], check=False)
    _run(rsync_base(node) + [remote_results, str(local_dir / "experiments.jsonl")], check=False)


def merge_results(
    fleet_runs_dir: Path,
    fleet_results_file: Path,
) -> None:
    """Merge per-node result files into a single fleet results JSONL."""
    from crucible.core.io import write_jsonl

    merged: dict[str, dict[str, Any]] = {
        row["id"]: row for row in read_jsonl(fleet_results_file) if "id" in row
    }
    for path in fleet_runs_dir.glob("*/experiments.jsonl"):
        for row in read_jsonl(path):
            if "id" in row:
                merged[row["id"]] = row
    write_jsonl(fleet_results_file, list(merged.values()))


# ---------------------------------------------------------------------------
# Lease recovery
# ---------------------------------------------------------------------------

def recover_running_leases(
    *,
    day_dir: Path,
    summary: dict[str, Any],
    queue: list[dict[str, Any]],
    queue_path: Path,
    nodes: list[dict[str, Any]],
    wave_name: str,
    result_index: dict[str, dict[str, Any]],
    max_attempts_per_run: int,
) -> tuple[list[dict[str, Any]], int]:
    """Detect leases on dead nodes and mark them retryable (or exhausted)."""
    node_state = {n["name"]: str(n.get("state") or "").lower() for n in nodes}
    changed = 0
    for row in queue:
        if row.get("wave") != wave_name or row.get("lease_state") != "running":
            continue
        if row.get("run_id") in result_index:
            continue
        assigned = row.get("assigned_node") or row.get("assigned_pod")
        health = node_state.get(assigned or "", "lost")
        if health == "ready":
            status = probe_node_status(next((n for n in nodes if n["name"] == assigned), {}))
            last_hb = status.get("last_heartbeat_at")
            if last_hb:
                try:
                    hb_time = datetime.fromisoformat(last_hb)
                    stale_seconds = (datetime.now(timezone.utc) - hb_time).total_seconds()
                    if stale_seconds <= 600:
                        continue
                    health = "heartbeat_stale"
                except (ValueError, TypeError):
                    continue
            else:
                continue
        row["ended_at"] = utc_now_iso()
        row["failed_node"] = assigned
        row["failure_reason"] = f"{health}_node"
        row["assigned_node"] = None
        row["remote_pid"] = None
        if int(row.get("attempt", 0)) >= max_attempts_per_run:
            row["lease_state"] = "finished"
            row["result_status"] = "failed_retry_budget"
            log_warn(f"{wave_name}: {row['experiment_name']} exhausted retry budget after {health}")
        else:
            row["lease_state"] = "retryable"
            row["result_status"] = "node_lost"
            append_event(
                day_dir, "lease_marked_retryable",
                wave=wave_name, run_id=row["run_id"], node=assigned, reason=health,
            )
            log_warn(
                f"{wave_name}: {row['experiment_name']} marked retryable "
                f"from {assigned} ({health})"
            )
        changed += 1
    if changed:
        save_queue(queue_path, queue)
        record_day_progress(
            day_dir, summary,
            phase="recovering", current_wave=wave_name,
            last_event=f"{wave_name}:recovering",
            nodes=nodes, queue=queue, wave_name=wave_name,
        )
    return queue, changed


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

_TRAIN_LOSS_RE = re.compile(r"step:(\d+)/\d+\s+train_loss:(\d+\.\d+)")


def load_baseline_curve(fleet_runs_dir: Path) -> list[tuple[int, float]]:
    """Parse baseline_control log files for step->train_loss pairs."""
    candidates: list[Path] = []
    if not fleet_runs_dir.exists():
        return []
    for node_dir in fleet_runs_dir.iterdir():
        if not node_dir.is_dir():
            continue
        logs_dir = node_dir / "logs"
        if not logs_dir.is_dir():
            continue
        for p in logs_dir.glob("*baseline_control*.txt"):
            if p.name.endswith((".launcher.txt", ".manifest.json", ".status.json")):
                continue
            candidates.append(p)
    if not candidates:
        return []
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        pairs = [(int(m.group(1)), float(m.group(2))) for m in _TRAIN_LOSS_RE.finditer(text)]
        if len(pairs) >= 50:
            return sorted(pairs, key=lambda t: t[0])
    return []


def _interpolate_baseline(curve: list[tuple[int, float]], step: int) -> float | None:
    """Linear interpolation of baseline train_loss at a given step."""
    if not curve:
        return None
    if step <= curve[0][0]:
        return curve[0][1]
    if step >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        s0, l0 = curve[i]
        s1, l1 = curve[i + 1]
        if s0 <= step <= s1:
            t = (step - s0) / (s1 - s0) if s1 != s0 else 0.0
            return l0 + t * (l1 - l0)
    return curve[-1][1]


def early_stop_underperformers(
    *,
    day_dir: Path,
    wave_name: str,
    queue: list[dict[str, Any]],
    queue_path: Path,
    nodes: list[dict[str, Any]],
    baseline_curve: list[tuple[int, float]],
    step_threshold: int = 6000,
    margin: float = 0.05,
) -> int:
    """Kill running experiments that underperform the baseline curve."""
    if not baseline_curve:
        return 0
    node_index = {n["name"]: n for n in nodes}
    stopped = 0
    for row in queue:
        if row.get("wave") != wave_name or row.get("lease_state") != "running":
            continue
        if "baseline" in row.get("tags", []):
            continue
        assigned = row.get("assigned_node") or row.get("assigned_pod")
        node = node_index.get(assigned or "")
        if node is None:
            continue
        status = probe_node_status(node)
        step = status.get("step")
        train_loss = status.get("latest_train_loss")
        if step is None or train_loss is None or step < step_threshold:
            continue
        ref_loss = _interpolate_baseline(baseline_curve, step)
        if ref_loss is None:
            continue
        threshold_loss = ref_loss * (1 + margin)
        if train_loss <= threshold_loss:
            continue
        pid = row.get("remote_pid")
        if pid:
            remote_exec(node, f"kill {pid}", check=False)
        row["lease_state"] = "finished"
        row["result_status"] = "early_stopped"
        row["ended_at"] = utc_now_iso()
        row["assigned_node"] = None
        row["remote_pid"] = None
        stopped += 1
        append_event(
            day_dir, "early_stopped",
            wave=wave_name, experiment=row["experiment_name"],
            step=step, train_loss=train_loss,
            ref_loss=ref_loss, threshold=threshold_loss,
        )
        log_warn(
            f"{wave_name}: early-stopped {row['experiment_name']} at step {step} "
            f"(loss={train_loss:.4f} > ref={ref_loss:.4f}*{1+margin:.2f}={threshold_loss:.4f})"
        )
    if stopped:
        save_queue(queue_path, queue)
    return stopped


# ---------------------------------------------------------------------------
# Capacity helpers
# ---------------------------------------------------------------------------

def pending_capacity_need(
    queue: list[dict[str, Any]], *, wave_name: str, dispatch_limit: int,
) -> int:
    """How many nodes are needed to serve the remaining wave queue."""
    active = [
        row for row in queue
        if row.get("wave") == wave_name and row.get("lease_state") in RECOVERABLE_LEASE_STATES
    ]
    return min(dispatch_limit, len(active))


def live_capacity_gap(*, nodes: list[dict[str, Any]], target_total: int) -> int:
    return max(0, target_total - len(nodes))


def bootstrap_recovery_candidates(
    nodes: list[dict[str, Any]],
    *,
    wave_queue: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Find nodes that could be re-bootstrapped to fill capacity."""
    from crucible.fleet.sync import ssh_ok

    active_assignments = {
        row.get("assigned_node") or row.get("assigned_pod")
        for row in wave_queue
        if row.get("lease_state") == "running"
    }
    candidates: list[dict[str, Any]] = []
    for node in nodes:
        if len(candidates) >= limit:
            break
        if node.get("name") in active_assignments:
            continue
        if ready_state(node) == "ready":
            continue
        if str(node.get("api_state") or "").lower() in BAD_API_STATES:
            continue
        if not node.get("ssh_host") or not ssh_ok(node):
            continue
        candidates.append(node)
    return candidates


def recover_bootstrap_incomplete_nodes(
    *,
    day_dir: Path,
    wave_name: str,
    nodes: list[dict[str, Any]],
    nodes_file: Path,
    queue: list[dict[str, Any]],
    project_root: Path,
    sync_excludes: list[str],
    train_shards: int,
    target_total_nodes: int,
    data_download_cmd: str | None = None,
) -> list[dict[str, Any]]:
    """Try to bootstrap partially-ready nodes when more capacity is needed."""
    wq = wave_rows(queue, wave_name)
    if not any(row.get("lease_state") in {"queued", "retryable"} for row in wq):
        return nodes
    if len(nodes) > target_total_nodes:
        return nodes
    needed = max(
        0,
        pending_capacity_need(queue, wave_name=wave_name, dispatch_limit=target_total_nodes)
        - count_bootstrapped_ready(nodes),
    )
    if needed <= 0:
        return nodes
    candidates = bootstrap_recovery_candidates(nodes, wave_queue=wq, limit=needed)
    for node in candidates:
        append_event(day_dir, "bootstrap_recovery_requested", wave=wave_name, node=node["name"])
        log_warn(f"{wave_name}: retrying bootstrap for {node['name']}")
        updated = bootstrap_node_worker(
            node,
            nodes_file=nodes_file,
            project_root=project_root,
            sync_excludes=sync_excludes,
            train_shards=train_shards,
            data_download_cmd=data_download_cmd,
        )
        upsert_node_record(nodes_file, updated)
        append_event(day_dir, "bootstrap_recovery_completed", wave=wave_name, node=node["name"])
        log_success(f"{wave_name}: bootstrap recovered for {node['name']}")
    return load_nodes_snapshot(nodes_file) or nodes


# ---------------------------------------------------------------------------
# Refresh + classify all nodes
# ---------------------------------------------------------------------------

def refresh_and_save_nodes(
    nodes: list[dict[str, Any]],
    *,
    nodes_file: Path,
    refresh_fn: Any = None,
) -> list[dict[str, Any]]:
    """Refresh node records from provider API, classify health, and save."""
    from crucible.fleet.inventory import merge_node_snapshots, load_nodes_if_exists

    seed = load_nodes_snapshot(nodes_file) or nodes
    if refresh_fn is not None:
        refreshed = refresh_fn(seed)
    else:
        refreshed = seed
    with NODES_LOCK:
        latest = load_nodes_if_exists(nodes_file)
        merged = merge_node_snapshots(latest or seed, refreshed)
        for n in merged:
            n["state"] = classify_health(n)
        save_nodes(nodes_file, merged)
    return merged


# ---------------------------------------------------------------------------
# Wave runner
# ---------------------------------------------------------------------------

def run_wave(
    *,
    day_dir: Path,
    summary: dict[str, Any],
    wave_name: str,
    experiments: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    nodes_file: Path,
    queue_path: Path,
    fleet_runs_dir: Path,
    fleet_results_file: Path,
    project_root: Path,
    sync_excludes: list[str],
    dispatch_limit: int,
    monitor_interval: int,
    min_completed: int,
    recovery: dict[str, Any],
    refresh_fn: Any = None,
    provision_replacement_fn: Any = None,
    baseline_curve: list[tuple[int, float]] | None = None,
    early_stop_step_threshold: int = 6000,
    early_stop_margin: float = 0.05,
    run_script: str = "crucible/runner/run_experiment.py",
    timeout_map: dict[str, dict[str, int]] | None = None,
    results_file_rel: str = "experiments.jsonl",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a single wave: enqueue, dispatch, monitor, collect, until done.

    Returns ``(results_for_wave, updated_nodes)``.
    """
    from crucible.fleet.day_run import (
        day_tag,
        leaderboard_snippet,
        remaining_waves,
        render_day_status,
    )
    from crucible.fleet.queue import enqueue_experiments

    log_step(
        f"wave: Starting {wave_name}: {len(experiments)} runs planned. "
        f"Left today: {', '.join(remaining_waves(summary)) or 'none'}"
    )
    append_event(day_dir, "wave_started", wave=wave_name, experiments=len(experiments))
    record_day_progress(
        day_dir, summary,
        phase="wave_enqueuing", current_wave=wave_name,
        last_event=f"{wave_name}:started", nodes=nodes,
    )

    added = enqueue_experiments(queue_path, experiments, limit=0)
    queue = load_queue(queue_path)
    record_day_progress(
        day_dir, summary,
        phase="wave_enqueued", current_wave=wave_name,
        last_event=f"{wave_name}:enqueued", nodes=nodes, queue=queue, wave_name=wave_name,
    )
    append_event(day_dir, "wave_enqueued", wave=wave_name, queued=len(added))

    queue = dispatch(
        nodes, queue, queue_path=queue_path,
        max_assignments=dispatch_limit,
        run_script=run_script, timeout_map=timeout_map,
    )
    record_day_progress(
        day_dir, summary,
        phase="wave_running", current_wave=wave_name,
        last_event=f"{wave_name}:dispatched", nodes=nodes, queue=queue, wave_name=wave_name,
    )
    running_now = len([
        r for r in queue
        if r.get("wave") == wave_name and r.get("lease_state") == "running"
    ])
    append_event(day_dir, "wave_dispatched", wave=wave_name, running=running_now)
    log_info(f"{wave_name}: dispatched {running_now} runs")

    while True:
        previous_states = {n["name"]: str(n.get("state") or "").lower() for n in nodes}
        nodes = refresh_and_save_nodes(nodes, nodes_file=nodes_file, refresh_fn=refresh_fn)
        for n in nodes:
            st = str(n.get("state") or "").lower()
            if (
                st != previous_states.get(n["name"])
                and st in BAD_API_STATES | {"unreachable", "ssh_timeout"}
            ):
                append_event(
                    day_dir, "node_marked_unhealthy",
                    wave=wave_name, node=n["name"], state=st,
                )
                log_warn(f"{n['name']} unhealthy: {st}")

        record_day_progress(
            day_dir, summary,
            phase="collecting", current_wave=wave_name,
            last_event=f"{wave_name}:collecting", nodes=nodes,
        )
        for n in nodes:
            collect_from_node(n, fleet_runs_dir=fleet_runs_dir, results_file_rel=results_file_rel)
        merge_results(fleet_runs_dir, fleet_results_file)
        all_results = read_jsonl(fleet_results_file)
        result_index = results_by_id(all_results)
        queue = reconcile_queue_with_results(load_queue(queue_path), result_index)
        save_queue(queue_path, queue)

        queue, _ = recover_running_leases(
            day_dir=day_dir, summary=summary,
            queue=queue, queue_path=queue_path,
            nodes=nodes, wave_name=wave_name,
            result_index=result_index,
            max_attempts_per_run=recovery["max_attempts_per_run"],
        )
        if baseline_curve:
            early_stop_underperformers(
                day_dir=day_dir, wave_name=wave_name,
                queue=queue, queue_path=queue_path,
                nodes=nodes, baseline_curve=baseline_curve,
                step_threshold=early_stop_step_threshold,
                margin=early_stop_margin,
            )

        nodes = recover_bootstrap_incomplete_nodes(
            day_dir=day_dir, wave_name=wave_name,
            nodes=nodes, nodes_file=nodes_file,
            queue=queue, project_root=project_root,
            sync_excludes=sync_excludes,
            train_shards=recovery["train_shards"],
            target_total_nodes=recovery["target_total_nodes"],
        )
        queue = load_queue(queue_path)

        # Dynamic replacement provisioning
        healthy_count = sum(1 for n in nodes if str(n.get("state") or "").lower() == "ready")
        active_needed = pending_capacity_need(
            queue, wave_name=wave_name, dispatch_limit=dispatch_limit,
        )
        replacement_budget_left = max(
            0, recovery["max_replacements"] - recovery["replacements_used"],
        )
        replacements_needed = max(0, active_needed - healthy_count)
        available_slots = live_capacity_gap(
            nodes=nodes, target_total=recovery["target_total_nodes"],
        )
        replacements_needed = min(replacements_needed, available_slots)
        if replacements_needed > 0 and replacement_budget_left > 0 and provision_replacement_fn:
            to_add = min(replacements_needed, replacement_budget_left)
            append_event(day_dir, "replacement_requested", wave=wave_name, needed=to_add)
            log_warn(f"{wave_name}: provisioning {to_add} replacement node(s)")
            new_nodes = provision_replacement_fn(
                existing_nodes=nodes, needed=to_add,
            )
            if new_nodes:
                nodes.extend(new_nodes)
                recovery["replacements_used"] += len(new_nodes)
                summary["replacements_used"] = recovery["replacements_used"]
                append_event(
                    day_dir, "replacement_bootstrapped",
                    wave=wave_name, count=len(new_nodes),
                )
                nodes = refresh_and_save_nodes(
                    nodes, nodes_file=nodes_file, refresh_fn=refresh_fn,
                )
                log_success(f"{wave_name}: added {len(new_nodes)} replacement node(s)")

        queue = dispatch(
            nodes, queue, queue_path=queue_path,
            max_assignments=dispatch_limit,
            run_script=run_script, timeout_map=timeout_map,
        )
        idle_capacity = summarize_idle_capacity(
            nodes, queue, wave_name=wave_name,
        )["nodes_ready_idle"]
        queued_runs = summarize_queue(queue, wave_name=wave_name)["queue_queued"]
        if idle_capacity > 0 and queued_runs > 0:
            append_event(
                day_dir, "idle_ready_capacity_detected",
                wave=wave_name, idle_ready_nodes=idle_capacity, queued_runs=queued_runs,
            )
            log_warn(
                f"{wave_name}: {idle_capacity} ready idle node(s) "
                f"while {queued_runs} run(s) remain queued"
            )
        record_day_progress(
            day_dir, summary,
            phase="wave_running", current_wave=wave_name,
            last_event=f"{wave_name}:monitor", nodes=nodes, queue=queue, wave_name=wave_name,
        )

        results_for_wave = wave_result_rows(queue, result_index, wave_name)
        wq = wave_rows(queue, wave_name)
        terminal = [row for row in wq if row.get("lease_state") in {"completed", "finished"}]
        completed = [
            row for row in wq
            if row.get("result_status") == "completed"
            or result_index.get(row["run_id"], {}).get("status") == "completed"
        ]

        print(render_day_status(summary), flush=True)
        print(render_monitor(nodes), flush=True)
        print("", flush=True)

        if len(terminal) >= len(experiments):
            if len(completed) < min_completed:
                append_event(
                    day_dir, "wave_failed",
                    wave=wave_name, completed=len(completed),
                    total=len(experiments), min_completed=min_completed,
                )
                log_error(
                    f"{wave_name} completion threshold missed: completed={len(completed)} "
                    f"required={min_completed} total={len(experiments)}"
                )
                raise RunnerError(
                    f"Wave {wave_name} did not meet completion threshold: "
                    f"completed={len(completed)} required={min_completed} "
                    f"total={len(experiments)}"
                )
            append_event(
                day_dir, "wave_completed",
                wave=wave_name, completed=len(completed), total=len(experiments),
            )
            dtag = day_tag(day_dir)
            wave_results_tagged = [
                row for row in all_results if dtag in row.get("tags", [])
            ]
            write_day_leaderboard(day_dir, wave_results_tagged)
            record_day_progress(
                day_dir, summary,
                phase="wave_completed", current_wave=wave_name,
                last_event=f"{wave_name}:completed",
                nodes=nodes, queue=queue, wave_name=wave_name,
            )
            log_success(f"{wave_name} complete: completed {len(completed)}/{len(experiments)}")
            for line in leaderboard_snippet(wave_results_tagged):
                log_info(f"leaderboard {line}")
            return results_for_wave, nodes

        time.sleep(monitor_interval)
