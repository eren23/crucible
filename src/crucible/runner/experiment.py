"""Core run_experiment() function with streaming output and OOM retry.

This is the main entry point for executing a training experiment as a
monitored subprocess.  It handles:

  - Preset resolution and config merging
  - Subprocess launch with streaming stdout capture
  - Live status tracking via RunTracker heartbeats
  - Output parsing for structured metrics extraction
  - Failure classification (OOM, timeout, signal, NaN, etc.)
  - Automatic OOM retry with halved batch size
  - Result persistence to JSONL

The training contract:
  - INPUT:  Environment variables define the experiment config
  - OUTPUT: Stdout lines matching known patterns carry metrics
"""
from __future__ import annotations

import os
import selectors
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crucible.core.errors import RunnerError
from crucible.core.io import append_jsonl, _json_ready
from crucible.core.log import log_info, log_warn
from crucible.core.config import ProjectConfig, load_config
from crucible.runner.output_parser import (
    OutputParser,
    classify_failure,
    steps_seen as _steps_seen_default,
    tail,
)
from crucible.runner.presets import get_preset
from crucible.runner.tracker import RunTracker
from crucible.runner.wandb import WandbLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_python(project_root: Path) -> str:
    """Find the best Python interpreter: venv first, then sys.executable."""
    venv_python = project_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _resolve_script(
    backend: str,
    project_config: ProjectConfig | None = None,
    project_root: Path | None = None,
) -> Path:
    """Resolve the training script path for the given backend.

    Checks crucible.yaml training config first, then falls back to
    convention-based discovery (train.py in project root).
    """
    # Check project config for explicit script mappings
    if project_config is not None:
        for tc in project_config.training:
            if tc.backend == backend:
                script = project_config.project_root / tc.script
                if script.exists():
                    return script

    # Fallback: look for common training script names
    root = project_root or Path.cwd()
    candidates = [
        root / f"train_{backend}.py",
        root / "train.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No training script found for backend {backend!r}. "
        f"Searched: {', '.join(str(c) for c in candidates)}. "
        f"Configure it in crucible.yaml under training:"
    )


def _read_log_file(log_path: Path) -> str:
    """Read a log file if it exists."""
    if log_path.exists():
        return log_path.read_text(encoding="utf-8")
    return ""


def _update_tracker_from_line(
    tracker: RunTracker,
    parser: OutputParser,
    line: str,
) -> None:
    """Parse a single output line and push relevant info to the tracker."""
    parsed = parser.parse_line(line)
    if parsed is None:
        stripped = line.strip()
        if stripped:
            tracker.update(last_output_line=stripped)
        return

    line_type = parsed["type"]

    if line_type == "warmup":
        tracker.heartbeat(
            "warming_up",
            warmup_step=parsed["step"],
            warmup_total=parsed["total"],
        )
    elif line_type == "train_loss":
        tracker.heartbeat(
            "training",
            step=parsed["step"],
            total_steps=parsed["total_steps"],
            latest_train_loss=parsed["train_loss"],
            last_output_line=line.strip(),
        )
    elif line_type == "val":
        tracker.heartbeat(
            "validating",
            step=parsed["step"],
            total_steps=parsed["total_steps"],
            latest_val_loss=parsed["val_loss"],
            latest_val_bpb=parsed.get("val_bpb"),  # kept for output_parser compat
            last_output_line=line.strip(),
        )
    elif line_type == "serializing":
        tracker.heartbeat("serializing", last_output_line=line.strip())
    elif line_type in ("final", "final_ttt"):
        tracker.heartbeat("finalizing", last_output_line=line.strip())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(
    config: dict[str, str],
    name: str,
    experiment_id: str | None = None,
    tags: list[str] | None = None,
    timeout_seconds: int = 1800,
    *,
    backend: str = "torch",
    preset: str = "proxy",
    stream_output: bool = True,
    project_root: str | Path | None = None,
    project_config: ProjectConfig | None = None,
    results_file: str | Path | None = None,
    parser: OutputParser | None = None,
) -> dict[str, Any]:
    """Run a single training experiment as a monitored subprocess.

    Args:
        config: Environment variable overrides for this experiment.
        name: Human-readable experiment name.
        experiment_id: Unique run ID (auto-generated if None).
        tags: Tags for categorisation.
        timeout_seconds: Hard wall-clock timeout (0 = no limit).
        backend: Training backend name (resolves training script).
        preset: Preset name for baseline config.
        stream_output: Whether to print stdout lines in real time.
        project_root: Project root directory (auto-detected if None).
        project_config: Pre-loaded ProjectConfig (loaded if None).
        results_file: Path to JSONL results file (from config if None).
        parser: Custom OutputParser (default patterns if None).

    Returns:
        Result dict with keys: id, name, timestamp, backend, preset,
        config, result, model_bytes, status, tags, error, failure_class,
        returncode, log_path, status_path, manifest_path.
    """
    # -- Resolve project config and paths --
    if project_config is None:
        try:
            project_config = load_config()
        except (FileNotFoundError, ValueError, OSError):
            project_config = ProjectConfig()

    root = Path(project_root).resolve() if project_root else project_config.project_root
    if parser is None:
        parser = OutputParser()

    # -- Resolve output paths --
    logs_dir = root / project_config.logs_dir
    if results_file is not None:
        results_path = Path(results_file)
    else:
        results_path = root / project_config.results_file

    # -- Resolve experiment identity --
    exp_id = experiment_id or f"exp_{int(time.time())}"
    timestamp = datetime.now(timezone.utc).isoformat()

    # -- Resolve script and python --
    script = _resolve_script(backend, project_config=project_config, project_root=root)
    python = _resolve_python(root)

    # -- Build tracker --
    tracker = RunTracker(exp_id, out_dir=logs_dir, project_root=root)

    # -- Merge preset + config --
    resolved_config = get_preset(preset, project_config=project_config)
    resolved_config.update(config)
    resolved_config["RUN_ID"] = exp_id
    resolved_config["RUN_BACKEND"] = backend
    resolved_config["RUN_PRESET"] = preset

    env = os.environ.copy()
    env.update(resolved_config)
    env["PYTHONUNBUFFERED"] = "1"

    tracker.update(
        state="queued",
        phase="queued",
        experiment_name=name,
        backend=backend,
        preset=preset,
        tags=tags or [],
        timeout_seconds=timeout_seconds,
        pid=None,
    )

    # -- Init W&B logger (inert if WANDB_PROJECT unset) --
    wandb_logger = WandbLogger.create(
        run_id=exp_id,
        config=resolved_config,
        backend=backend,
        tracker=tracker,
        tags=tags,
    )
    if wandb_logger.enabled:
        wandb_logger.update_config({
            "crucible_run_id": exp_id,
            "crucible_preset": preset,
        })

    # -- Build result skeleton --
    result: dict[str, Any] = {
        "id": exp_id,
        "name": name,
        "timestamp": timestamp,
        "backend": backend,
        "preset": preset,
        "config": config,
        "result": None,
        "model_bytes": None,
        "status": "queued",
        "tags": tags or [],
        "error": None,
        "failure_class": None,
        "returncode": None,
        "log_path": str(tracker.log_path.resolve()),
        "status_path": str(tracker.status_path.resolve()),
        "manifest_path": str(tracker.manifest_path.resolve()),
    }

    oom_retry_used = False
    proc: subprocess.Popen[str] | None = None

    try:
        while True:
            lines: list[str] = []
            timed_out = False

            tracker.write_manifest(
                backend=backend,
                script_path=script,
                config=resolved_config,
                tags=tags,
                extra={
                    "experiment_name": name,
                    "timeout_seconds": timeout_seconds,
                    "preset": preset,
                    "runner_python": python,
                    "oom_retry_used": oom_retry_used,
                    "parent_run_id": config.get("PARENT_RUN_ID") or None,
                },
            )

            proc = subprocess.Popen(
                [python, str(script)],
                env=env,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            result["status"] = "running"
            tracker.update(
                state="starting",
                phase="launching",
                pid=proc.pid,
                oom_retry_used=oom_retry_used,
            )

            assert proc.stdout is not None
            start = time.monotonic()
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)

            # -- Streaming read loop --
            while True:
                if timeout_seconds > 0 and time.monotonic() - start > timeout_seconds:
                    timed_out = True
                    proc.kill()
                    break

                events = selector.select(timeout=0.25)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    lines.append(line)
                    _update_tracker_from_line(tracker, parser, line)
                    if stream_output:
                        print(f"[{exp_id}] {line.rstrip()}", flush=True)

                if proc.poll() is not None:
                    break

            # -- Drain remaining output --
            remaining = proc.stdout.read()
            selector.close()
            if remaining:
                lines.append(remaining)
                for rem_line in remaining.splitlines():
                    _update_tracker_from_line(tracker, parser, rem_line)
                    if stream_output:
                        print(f"[{exp_id}] {rem_line.rstrip()}", flush=True)

            proc.wait(timeout=5)
            result["returncode"] = proc.returncode
            combined_output = "".join(lines)

            # -- Parse output for metrics --
            parsed = parser.parse(combined_output)

            # Try log file as fallback for missing metrics
            if parsed is None or parsed["status"] != "completed":
                log_text = _read_log_file(tracker.log_path)
                if log_text:
                    log_parsed = parser.parse(log_text)
                    if log_parsed is not None and (
                        parsed is None or log_parsed["status"] == "completed"
                    ):
                        parsed = log_parsed
                    combined_output += "\n" + log_text

            # -- Classify outcome --
            final_status, failure_class = classify_failure(
                proc.returncode, combined_output, timed_out
            )
            steps_completed = (
                parsed["result"].get("steps_completed", 0)
                if parsed is not None
                else _steps_seen_default(combined_output)
            )

            # -- OOM retry logic --
            train_batch_tokens = int(
                resolved_config.get("TRAIN_BATCH_TOKENS", "0") or "0"
            )
            if (
                not oom_retry_used
                and failure_class == "oom_suspected"
                and steps_completed < 20
                and train_batch_tokens >= 2
            ):
                reduced_tokens = max(1, train_batch_tokens // 2)
                resolved_config["TRAIN_BATCH_TOKENS"] = str(reduced_tokens)
                env["TRAIN_BATCH_TOKENS"] = str(reduced_tokens)
                result["oom_retry_from_train_batch_tokens"] = train_batch_tokens
                result["oom_retry_to_train_batch_tokens"] = reduced_tokens
                oom_retry_used = True
                tracker.update(
                    state="retrying",
                    phase="retrying",
                    failure_class=failure_class,
                    last_output_line=(
                        f"oom_retry train_batch_tokens:"
                        f"{train_batch_tokens}->{reduced_tokens}"
                    ),
                )
                if stream_output:
                    print(
                        f"[{exp_id}] oom_retry train_batch_tokens:"
                        f"{train_batch_tokens}->{reduced_tokens}",
                        flush=True,
                    )
                continue  # Retry the while True loop

            # -- Record final result --
            result["failure_class"] = failure_class

            if parsed is not None and final_status == "completed":
                result["status"] = "completed"
                result["result"] = parsed["result"]
                result["model_bytes"] = parsed["model_bytes"]
                tracker.finalize(
                    "completed",
                    phase="completed",
                    returncode=proc.returncode,
                    failure_class=None,
                    result=result["result"],
                    model_bytes=result["model_bytes"],
                )
            elif parsed is not None:
                result["status"] = "partial_recoverable"
                result["result"] = parsed["result"]
                result["model_bytes"] = parsed["model_bytes"]
                result["error"] = (
                    f"Recovered partial metrics from failed run. "
                    f"failure_class={failure_class or 'unknown'} "
                    f"returncode={proc.returncode}"
                )
                tracker.finalize(
                    "partial_recoverable",
                    phase="completed",
                    returncode=proc.returncode,
                    failure_class=failure_class,
                    result=result["result"],
                    error=result["error"],
                )
            else:
                result["status"] = final_status
                result["error"] = (
                    f"{failure_class or 'unknown_failure'} "
                    f"returncode={proc.returncode}\n"
                    f"output_tail: {tail(combined_output)}"
                )
                tracker.finalize(
                    final_status,
                    phase="completed",
                    returncode=proc.returncode,
                    failure_class=failure_class,
                    error=result["error"],
                )
            break  # Done -- exit the while True loop

    except KeyboardInterrupt:
        result["status"] = "failed"
        result["error"] = "Interrupted by user"
        result["failure_class"] = "interrupted"
        if proc is not None and proc.returncode is None:
            proc.kill()
        tracker.finalize(
            "failed",
            phase="completed",
            returncode=result["returncode"],
            failure_class="interrupted",
            error=result["error"],
        )
        raise
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["failure_class"] = "runner_exception"
        if proc is not None and proc.returncode is None:
            proc.kill()
        tracker.finalize(
            "failed",
            phase="completed",
            returncode=result["returncode"],
            failure_class="runner_exception",
            error=result["error"],
        )

    # -- Finish W&B run --
    if wandb_logger.enabled:
        if result.get("result"):
            wandb_logger.update_summary(result["result"])
        exit_code = 0 if result["status"] == "completed" else 1
        wandb_logger.finish(exit_code=exit_code)

    # -- Persist result --
    append_jsonl(results_path, result)
    return result
