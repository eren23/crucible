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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from crucible.core.errors import RunnerError
from crucible.core.io import append_jsonl, _json_ready
from crucible.core.log import log_info, log_step, log_success, log_warn
from crucible.fleet.inventory import BAD_API_STATES
from crucible.fleet.sync import remote_exec, rsync_base, _run

if TYPE_CHECKING:
    from crucible.core.config import ProjectSpec
    from crucible.core.types import NodeRecord


def launch_project(
    node: NodeRecord,
    spec: ProjectSpec,
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

    # Build a single export prefix that lives INSIDE the inner ``bash -lc``
    # train command.  Putting exports at the outer SSH-shell level (before
    # ``python -c``) is fragile: the inner ``bash -lc`` is a login shell that
    # sources ``/etc/profile`` + ``~/.bash_profile``, and many train commands
    # then ``source .env`` themselves.  Either of those can clobber values we
    # set at the outer level — observed in practice with TEMP / SEED silently
    # falling back to defaults despite ``run_project(overrides={...})``.
    # Exporting *inside* the inner shell, AFTER profile and after the user's
    # own ``source .env``, makes overrides authoritative.
    override_exports_inline = ""
    if effective_overrides:
        parts = [f"export {k}={shlex.quote(str(v))}" for k, v in effective_overrides.items()]
        override_exports_inline = " && ".join(parts) + " && "

    # Activation + env sourcing.  ``set -a`` forces every variable
    # assigned by ``source`` (including bare ``KEY=val`` lines, not just
    # ``export KEY=val``) to be exported into the subprocess environment.
    # This guards against bootstrap writing a .env without ``export``
    # prefixes — a bug we hit in practice which caused WANDB_API_KEY to
    # silently drop and training to log offline.
    activate = f"cd {ws}"
    if spec.python:
        activate += " && source .venv/bin/activate"
    source_env = (
        f"if [ -f {ws}/.env ]; then set -a; source {ws}/.env; set +a; fi"
    )

    # Preflight: if the spec forwards WANDB_API_KEY, enforce that it is
    # actually exported after sourcing .env.  Fail loudly at launch time
    # rather than silently training offline.
    forwarded = list(getattr(spec, "env_forward", []) or [])
    preflight = ""
    if "WANDB_API_KEY" in forwarded:
        preflight = (
            ' && if [ -z "${WANDB_API_KEY:-}" ]; then '
            'echo "[Crucible] ERROR: WANDB_API_KEY missing after sourcing .env. '
            'Refusing to launch — W&B would silently go offline." >&2; '
            'exit 101; fi'
        )

    wrapped_train_cmd = f"{override_exports_inline}{spec.train}"
    exit_code_file = f"{log_dir}/{run_id}.exit_code"
    launch_snippet = (
        "import pathlib, subprocess, threading; "
        f"pathlib.Path({log_dir!r}).mkdir(parents=True, exist_ok=True); "
        f"log = open({log_file!r}, 'ab', buffering=0); "
        f"proc = subprocess.Popen(['bash', '-lc', {wrapped_train_cmd!r}], "
        "stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, "
        "start_new_session=True); "
        "print(proc.pid); "
        f"threading.Thread(target=lambda: open({exit_code_file!r},'w').write(str(proc.wait())), daemon=True).start()"
    )
    cmd = (
        f"{activate} && {source_env}{preflight} && "
        f"mkdir -p {log_dir_quoted} && "
        f"python -c {shlex.quote(launch_snippet)}"
    )

    log_step(f"{name}: launching {spec.name!r} training (run_id={run_id})")
    proc = remote_exec(
        node,
        cmd,
        check=False,
        timeout=max(int(getattr(spec, "launch_timeout", 300) or 300), 1),
    )

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


def chain_project_variants(
    node: NodeRecord,
    spec: ProjectSpec,
    variants: list[str],
    *,
    overrides: dict[str, str] | None = None,
    poll_interval: int = 30,
    on_variant_start: Callable[[str, dict[str, Any]], None] | None = None,
    on_variant_complete: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run a sequence of project variants on the same node, auto-chaining.

    Launches the first variant, polls until completion, then launches the
    next. Returns a list of results, one per variant.

    Args:
        node: Node dict with ssh_host, ssh_port, workspace_path, etc.
        spec: ProjectSpec with variants dict and train command.
        variants: Ordered list of variant names from spec.variants.
        overrides: Extra env overrides applied to ALL variants (wins over variant env).
        poll_interval: Seconds between completion checks (default 30).
        on_variant_start: Optional callback(variant_name, run_id, node_name).
        on_variant_complete: Optional callback(variant_name, result_dict).

    Returns:
        List of dicts, each with: variant, run_id, pid, status, duration_s.
    """
    results = []
    caller_overrides = dict(overrides or {})

    for i, variant_name in enumerate(variants):
        if variant_name not in spec.variants:
            available = sorted(spec.variants.keys()) or ["(none)"]
            raise RunnerError(
                f"Variant {variant_name!r} not in project {spec.name!r}. "
                f"Available: {', '.join(available)}"
            )

        # Merge: variant env + caller overrides (caller wins)
        variant_env = dict(spec.variants[variant_name])
        merged = {**variant_env, **caller_overrides}
        merged.setdefault("CRUCIBLE_VARIANT_NAME", variant_name)
        merged.setdefault("WANDB_RUN_NAME", variant_name)

        run_id = (
            f"{spec.name}_{int(time.time_ns())}_{variant_name[:20]}"
        )

        log_step(
            f"chain [{i + 1}/{len(variants)}] launching variant "
            f"{variant_name!r} on {node['name']}"
        )

        if on_variant_start:
            on_variant_start(variant_name, run_id, node["name"])

        start_time = time.time()
        try:
            launch_result = launch_project(node, spec, run_id, overrides=merged)
            pid = launch_result["pid"]
        except Exception as exc:
            log_warn(f"chain: variant {variant_name!r} failed to launch: {exc}")
            results.append({
                "variant": variant_name,
                "run_id": run_id,
                "pid": None,
                "status": "launch_failed",
                "error": str(exc),
                "duration_s": 0,
            })
            continue

        # Poll until process finishes
        log_info(f"chain: polling {node['name']} PID {pid} every {poll_interval}s")
        final_probe = None
        while True:
            time.sleep(poll_interval)
            final_probe = probe_project_process(
                node, pid, run_id=run_id, workspace=spec.workspace,
            )
            if not final_probe.get("running"):
                break

        duration_s = round(time.time() - start_time)
        exit_code = final_probe.get("exit_code") if final_probe else None
        status = "completed" if exit_code == 0 else (
            "failed" if exit_code is not None else "completed"
        )

        log_fn = log_success if status == "completed" else log_warn
        log_fn(
            f"chain [{i + 1}/{len(variants)}] variant {variant_name!r} "
            f"{status} in {duration_s}s (exit_code={exit_code})"
        )

        result = {
            "variant": variant_name,
            "run_id": run_id,
            "pid": pid,
            "status": status,
            "exit_code": exit_code,
            "duration_s": duration_s,
        }
        results.append(result)

        if on_variant_complete:
            on_variant_complete(variant_name, result)

    return results


def check_project_running(node: NodeRecord, pid: int) -> bool:
    """Check if a training process is still running on the pod."""
    probe = probe_project_process(node, pid)
    return bool(probe.get("running"))


def probe_project_process(
    node: NodeRecord,
    pid: int,
    run_id: str | None = None,
    workspace: str = "/workspace/project",
) -> dict[str, Any]:
    """Probe a detached remote process, including exit code if finished."""
    try:
        exit_code_path = f"{workspace}/logs/{run_id}.exit_code" if run_id else ""
        read_exit = f" && cat {shlex.quote(exit_code_path)} 2>/dev/null" if exit_code_path else ""
        cmd = f"kill -0 {int(pid)} 2>/dev/null && echo running || (echo stopped{read_exit})"
        proc = remote_exec(node, cmd, check=False)
    except Exception as exc:
        return {"reachable": False, "running": None, "exit_code": None, "error": str(exc)}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0 and not stdout:
        return {
            "reachable": False,
            "running": None,
            "exit_code": None,
            "error": stderr or stdout or f"ssh_returncode_{proc.returncode}",
        }

    is_running = "running" in stdout
    exit_code = None
    if not is_running and stdout:
        lines = stdout.strip().split("\n")
        if len(lines) >= 2:
            try:
                exit_code = int(lines[-1].strip())
            except ValueError:
                pass
    return {
        "reachable": True,
        "running": is_running,
        "exit_code": exit_code,
        "error": stderr or None,
    }


def _node_state_label(node: NodeRecord) -> str:
    api_state = str(node.get("api_state") or "").lower()
    state = str(node.get("state") or "").lower()
    return api_state or state or "unknown"


def _wandb_is_live(wandb_info: dict[str, Any], stale_after_s: int = 180) -> bool:
    """Return True if W&B reports the run is alive right now.

    Treats ``state == "running"`` as authoritative when the heartbeat is
    fresh. The stale-window guards against zombie W&B runs whose process
    crashed without finishing — those keep ``state="running"`` but stop
    sending heartbeats. Default 180s matches W&B's own zombie-detection
    grace period.

    Safety bias: an unparseable heartbeat returns False, not True. False
    here means "fall through to other classifiers" (which can still mark
    the run completed if there are post-mortem metrics). True here would
    pin the run as ``running`` indefinitely, blocking ``collect_project_result``
    from ever finalizing — a worse failure mode than a one-off mis-classify.
    """
    state = str(wandb_info.get("state") or "").lower()
    if state != "running":
        return False
    heartbeat_at = wandb_info.get("heartbeat_at")
    if not heartbeat_at:
        # No heartbeat info but state=running — trust W&B (better than
        # mis-classifying as failed). The stale check is best-effort.
        return True
    from datetime import datetime, timezone
    try:
        if isinstance(heartbeat_at, (int, float)):
            ts = datetime.fromtimestamp(float(heartbeat_at), tz=timezone.utc)
        else:
            text = str(heartbeat_at).rstrip("Z")
            ts = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, OverflowError):
        log_warn(f"W&B heartbeat unparseable ({heartbeat_at!r}) — treating as not-live.")
        return False
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return age <= stale_after_s


def _classify_project_status(
    *,
    process_probe: dict[str, Any],
    parsed: dict[str, Any] | None,
    log_text: str,
    node: NodeRecord,
    wandb_info: dict[str, Any],
) -> tuple[str, str | None]:
    """Resolve the best status/failure_class for an external project run.

    Status precedence:
      1. local SSH probe says the pid is alive       → ``running``
      2. W&B reports state=running with fresh heartbeat → ``running``
         (covers the "stdout fell quiet between problems" failure mode
         where the pid disappeared from this side because the SSH probe
         timed out, but W&B is still receiving step-level updates)
      3. stdout parser saw a terminal marker         → ``completed``
      4. W&B has any post-mortem record              → ``completed``
      5. node unreachable / interrupted              → ``interrupted``
      6. otherwise classify via ``classify_failure`` → ``failed``
    """
    if process_probe.get("running") is True:
        return "running", None

    if _wandb_is_live(wandb_info):
        return "running", None

    parsed_status = parsed.get("status") if parsed else None
    if parsed_status == "completed":
        return "completed", None

    # W&B post-mortem: only treat as completed when W&B itself says so.
    # ``state == "finished"`` is the only success terminal; "failed",
    # "crashed", "killed" must surface as failures with the W&B state as
    # the failure_class so callers see the real reason.
    wandb_state = str(wandb_info.get("state") or "").lower() if wandb_info else ""
    if wandb_state == "finished":
        return "completed", None
    if wandb_state in {"failed", "crashed", "killed"}:
        return "failed", f"wandb_{wandb_state}"

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
    node: NodeRecord,
    spec: ProjectSpec,
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

    # Collect checkpoints (best-effort)
    try:
        ws = node.get("workspace_path", "/workspace/project")
        remote_ckpt = f"{user}@{host}:{ws}/checkpoints/"
        local_ckpt_dir = logs_dir.parent / "checkpoints" / run_id
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        _run(rsync_base(node) + [remote_ckpt, str(local_ckpt_dir) + "/"], check=False)
    except Exception as exc:
        log_warn(f"{name}: failed to rsync checkpoints: {exc}")

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

    # Fetch WandB metrics + lifecycle state. We fetch even when the local
    # SSH probe says ``still_running`` so the classifier can cross-check:
    # if the SSH probe spuriously reports the pid gone (BAD_API_STATES,
    # transient ssh_timeout, etc.) but W&B still has a fresh heartbeat,
    # the run is alive and must not be marked failed.
    wandb_metrics: dict[str, float] = {}
    wandb_info: dict[str, Any] = {}
    wandb_required = bool(spec.env_set.get("WANDB_PROJECT"))
    if wandb_required:
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
