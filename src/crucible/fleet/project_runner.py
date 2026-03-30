"""Run external projects on fleet pods via detached SSH processes.

All shell-interpolated values are sanitized via ``shlex.quote`` or embedded
via Python string literals to prevent injection. The env_forward denylist in
sync.py provides an additional layer of protection for credential forwarding.
"""
from __future__ import annotations

import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crucible.core.errors import RunnerError
from crucible.core.io import append_jsonl, _json_ready
from crucible.core.log import log_info, log_step, log_success, log_warn
from crucible.fleet.inventory import BAD_API_STATES
from crucible.fleet.sync import remote_exec, rsync_base, _run


def launch_project(
    node: dict[str, Any],
    spec: Any,
    run_id: str,
    *,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Launch training as a detached process on the pod. Returns immediately."""
    workspace = spec.workspace
    ws = shlex.quote(workspace)
    log_dir = f"{workspace}/logs"
    log_dir_quoted = shlex.quote(log_dir)
    log_file = f"{log_dir}/{run_id}.log"
    name = node["name"]

    # Build env var exports for overrides (all values quoted)
    effective_overrides = dict(overrides or {})
    effective_overrides.setdefault("WANDB_RUN_NAME", run_id)
    effective_overrides.setdefault("CRUCIBLE_RUN_ID", run_id)
    effective_overrides.setdefault("CRUCIBLE_REMOTE_NODE", name)
    effective_overrides.setdefault("CRUCIBLE_EXECUTION_PROVIDER", node.get("provider", "runpod"))
    effective_overrides.setdefault("CRUCIBLE_ENFORCE_CONTRACT", "1")

    override_exports = ""
    if effective_overrides:
        parts = [f"export {k}={shlex.quote(str(v))}" for k, v in effective_overrides.items()]
        override_exports = " && ".join(parts) + " && "

    # Activation + env sourcing
    activate = f"cd {ws}"
    if spec.python:
        activate += " && source .venv/bin/activate"
    source_env = f"if [ -f {ws}/.env ]; then source {ws}/.env; fi"

    train_cmd = spec.train
    launch_snippet = (
        "import pathlib, subprocess; "
        f"pathlib.Path({log_dir!r}).mkdir(parents=True, exist_ok=True); "
        f"log = open({log_file!r}, 'ab', buffering=0); "
        f"proc = subprocess.Popen(['bash', '-lc', {train_cmd!r}], "
        "stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, "
        "start_new_session=True); "
        "print(proc.pid)"
    )
    cmd = (
        f"{activate} && {source_env} && "
        f"{override_exports}"
        f"mkdir -p {log_dir_quoted} && "
        f"python -c {shlex.quote(launch_snippet)}"
    )

    log_step(f"{name}: launching {spec.name!r} training (run_id={run_id})")
    proc = remote_exec(node, cmd, check=False)

    if proc.returncode != 0:
        raise RunnerError(
            f"Failed to launch training on {name}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )

    pid_str = (proc.stdout or "").strip().split("\n")[-1]
    try:
        pid = int(pid_str)
    except ValueError:
        raise RunnerError(
            f"Could not parse PID from launch output on {name}: {pid_str!r}"
        )

    log_success(f"{name}: training launched with PID {pid}")
    return {
        "run_id": run_id,
        "pid": pid,
        "node": name,
        "project": spec.name,
        "wandb_run_name": effective_overrides.get("WANDB_RUN_NAME", run_id),
        "start_time": datetime.now(timezone.utc).isoformat(),
    }


def check_project_running(node: dict[str, Any], pid: int) -> bool:
    """Check if a training process is still running on the pod."""
    probe = probe_project_process(node, pid)
    return bool(probe.get("running"))


def probe_project_process(node: dict[str, Any], pid: int) -> dict[str, Any]:
    """Probe a detached remote process, preserving SSH reachability details."""
    try:
        proc = remote_exec(
            node,
            f"kill -0 {int(pid)} 2>/dev/null && echo running || echo stopped",
            check=False,
        )
    except Exception as exc:
        return {"reachable": False, "running": None, "error": str(exc)}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0 and not stdout:
        return {
            "reachable": False,
            "running": None,
            "error": stderr or stdout or f"ssh_returncode_{proc.returncode}",
        }
    return {
        "reachable": True,
        "running": "running" in stdout,
        "error": stderr or None,
    }


def _node_state_label(node: dict[str, Any]) -> str:
    api_state = str(node.get("api_state") or "").lower()
    state = str(node.get("state") or "").lower()
    return api_state or state or "unknown"


def _classify_project_status(
    *,
    process_probe: dict[str, Any],
    parsed: dict[str, Any] | None,
    log_text: str,
    node: dict[str, Any],
    wandb_info: dict[str, Any],
) -> tuple[str, str | None]:
    """Resolve the best status/failure_class for an external project run."""
    if process_probe.get("running") is True:
        return "running", None

    parsed_status = parsed.get("status") if parsed else None
    if parsed_status == "completed":
        return "completed", None

    if wandb_info:
        return "completed", None

    node_state = _node_state_label(node)
    if (not process_probe.get("reachable")) or node_state in BAD_API_STATES or node_state in {"unreachable", "ssh_timeout"}:
        failure_class = node_state if node_state != "unknown" else "node_unreachable"
        return "interrupted", failure_class

    if parsed_status == "partial_recoverable":
        return "failed", "no_terminal_marker"

    from crucible.runner.output_parser import classify_failure

    status, failure_class = classify_failure(1, log_text, timed_out=False)
    if status == "completed":
        status = "failed"
    return status, failure_class or "unknown_exit"


def collect_project_result(
    node: dict[str, Any],
    spec: Any,
    run_id: str,
    pid: int,
    *,
    wandb_run_name: str | None = None,
    experiment_meta: dict[str, Any] | None = None,
    local_logs_dir: Path | None = None,
    results_file: Path | None = None,
) -> dict[str, Any]:
    """Collect results from an external project run.

    Rsyncs the log file, parses stdout for metrics, optionally fetches
    WandB metrics, and writes to results JSONL.
    """
    from crucible.runner.output_parser import OutputParser

    ws = spec.workspace
    name = node["name"]
    remote_log = f"{ws}/logs/{run_id}.log"

    still_running = check_project_running(node, pid)
    process_probe = {"reachable": True, "running": still_running, "error": None}
    node_state = _node_state_label(node)
    if (not still_running) and node_state in BAD_API_STATES | {"unreachable", "ssh_timeout"}:
        process_probe = {"reachable": False, "running": False, "error": node_state}
    status = "running" if still_running else "failed"
    failure_class: str | None = None

    # Rsync log file from pod
    logs_dir = local_logs_dir or Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    local_log = logs_dir / f"{run_id}.log"

    log_sync_failed = False
    try:
        user = node.get("user", "root")
        host = node["ssh_host"]
        remote_path = f"{user}@{host}:{remote_log}"
        rsync_cmd = rsync_base(node) + [remote_path, str(local_log)]
        _run(rsync_cmd, check=False)
    except Exception as exc:
        log_sync_failed = True
        log_warn(f"{name}: failed to rsync log: {exc}")

    # Parse stdout metrics from log
    parser = OutputParser()
    log_text = ""
    stdout_metrics: dict[str, Any] = {}
    if local_log.exists():
        log_text = local_log.read_text(encoding="utf-8", errors="replace")
        parsed = parser.parse(log_text)
        if parsed is not None:
            stdout_metrics = parsed.get("result", {})
        else:
            parsed = None
    else:
        parsed = None

    # Fetch WandB metrics if configured
    wandb_metrics: dict[str, float] = {}
    wandb_info: dict[str, Any] = {}
    wandb_required = bool(spec.env_set.get("WANDB_PROJECT"))
    if wandb_required and not still_running:
        try:
            from crucible.runner.wandb_logger import fetch_wandb_run_info
            wandb_project = spec.env_set.get("WANDB_PROJECT", "")
            wandb_entity = spec.env_set.get("WANDB_ENTITY", "")
            if wandb_project:
                wandb_info = fetch_wandb_run_info(
                    project=wandb_project,
                    entity=wandb_entity or None,
                    run_name=wandb_run_name or run_id,
                )
                wandb_metrics = wandb_info.get("metrics", {})
        except Exception as exc:
            log_warn(f"{name}: WandB metric fetch failed: {exc}")

    if (not still_running) and process_probe.get("reachable") and log_sync_failed and not local_log.exists():
        deeper_probe = probe_project_process(node, pid)
        if deeper_probe.get("reachable") is False:
            process_probe = deeper_probe

    # Merge: WandB takes precedence over stdout
    merged = {**stdout_metrics, **wandb_metrics}
    status, failure_class = _classify_project_status(
        process_probe=process_probe,
        parsed=parsed,
        log_text=log_text,
        node=node,
        wandb_info=wandb_info,
    )

    contract_status = "compliant"
    if wandb_required and status not in {"running", "interrupted"} and not wandb_info:
        contract_status = "wandb_missing"

    result: dict[str, Any] = {
        "id": run_id,
        "project": spec.name,
        "name": f"{spec.name}-{run_id}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": name,
        "status": status,
        "failure_class": failure_class,
        "result": _json_ready(merged) if merged else None,
        "returncode": None if still_running else (0 if status == "completed" else 1),
        "log_path": str(local_log),
        "config": {},
        "execution_provider": node.get("provider", "runpod"),
        "remote_node": name,
        "remote_node_state": _node_state_label(node),
        "last_observed_at": datetime.now(timezone.utc).isoformat(),
        "contract_status": contract_status,
        "wandb": {
            "required": wandb_required,
            "project": spec.env_set.get("WANDB_PROJECT", "") or None,
            "entity": spec.env_set.get("WANDB_ENTITY", "") or None,
            "mode": spec.env_set.get("WANDB_MODE", "online"),
            "run_name": wandb_run_name or run_id,
            "url": wandb_info.get("url"),
            "enabled": bool(wandb_info),
        },
    }

    if experiment_meta:
        result.update({k: v for k, v in experiment_meta.items() if v is not None})
        if experiment_meta.get("config"):
            result["config"] = experiment_meta["config"]
        if experiment_meta.get("name"):
            result["name"] = experiment_meta["name"]

    # Persist if training is done
    if results_file is not None and not still_running:
        append_jsonl(results_file, result)

    # Include log tail for quick inspection
    result["log_tail"] = log_text[-2000:] if log_text else ""
    return result
