"""Run external projects on fleet pods via detached SSH processes.

All values interpolated into shell commands are sanitized via shlex.quote
to prevent injection. The env_forward denylist in sync.py provides an
additional layer of protection for credential forwarding.
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
from crucible.fleet.sync import remote_exec, rsync_base, _run


def launch_project(
    node: dict[str, Any],
    spec: Any,
    run_id: str,
    *,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Launch training as a detached process on the pod. Returns immediately."""
    ws = shlex.quote(spec.workspace)
    log_dir = f"{ws}/logs"
    log_file = f"{log_dir}/{shlex.quote(run_id)}.log"
    name = node["name"]

    # Build env var exports for overrides (all values quoted)
    override_exports = ""
    if overrides:
        parts = [f"export {k}={shlex.quote(v)}" for k, v in overrides.items()]
        override_exports = " && ".join(parts) + " && "

    # Activation + env sourcing
    activate = f"cd {ws}"
    if spec.python:
        activate += " && source .venv/bin/activate"
    source_env = f"if [ -f {ws}/.env ]; then source {ws}/.env; fi"

    # Detached command via nohup + bash -c (so inline env vars work)
    train_cmd = spec.train
    cmd = (
        f"mkdir -p {log_dir} && "
        f"{activate} && {source_env} && "
        f"{override_exports}"
        f"nohup bash -c {shlex.quote(train_cmd)} > {log_file} 2>&1 & echo $!"
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
        "start_time": datetime.now(timezone.utc).isoformat(),
    }


def check_project_running(node: dict[str, Any], pid: int) -> bool:
    """Check if a training process is still running on the pod."""
    proc = remote_exec(
        node,
        f"kill -0 {int(pid)} 2>/dev/null && echo running || echo stopped",
        check=False,
    )
    return "running" in (proc.stdout or "")


def collect_project_result(
    node: dict[str, Any],
    spec: Any,
    run_id: str,
    pid: int,
    *,
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
    status = "running" if still_running else "completed"

    # Rsync log file from pod
    logs_dir = local_logs_dir or Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    local_log = logs_dir / f"{run_id}.log"

    try:
        user = node.get("user", "root")
        host = node["ssh_host"]
        remote_path = f"{user}@{host}:{remote_log}"
        rsync_cmd = rsync_base(node) + [remote_path, str(local_log)]
        _run(rsync_cmd, check=False)
    except Exception as exc:
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
            if parsed["status"] == "completed":
                status = "completed"

    # Fetch WandB metrics if configured
    wandb_metrics: dict[str, float] = {}
    if spec.metrics.source == "wandb" and not still_running:
        try:
            from crucible.runner.wandb import fetch_wandb_metrics
            wandb_project = spec.env_set.get("WANDB_PROJECT", "")
            wandb_entity = spec.env_set.get("WANDB_ENTITY", "")
            if wandb_project:
                wandb_metrics = fetch_wandb_metrics(
                    project=wandb_project,
                    entity=wandb_entity or None,
                )
        except Exception as exc:
            log_warn(f"{name}: WandB metric fetch failed: {exc}")

    # Merge: WandB takes precedence over stdout
    merged = {**stdout_metrics, **wandb_metrics}

    result: dict[str, Any] = {
        "id": run_id,
        "project": spec.name,
        "name": f"{spec.name}-{run_id}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": name,
        "status": status,
        "result": _json_ready(merged) if merged else None,
        "returncode": None if still_running else 0,
        "log_path": str(local_log),
        "config": {},
    }

    # Persist if training is done
    if results_file is not None and not still_running:
        append_jsonl(results_file, result)

    # Include log tail for quick inspection
    result["log_tail"] = log_text[-2000:] if log_text else ""
    return result
