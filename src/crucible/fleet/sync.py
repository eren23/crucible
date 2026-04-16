"""SSH/rsync helpers: ssh_base, rsync_base, remote_exec, sync_repo, sync_env_file."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from crucible.core.types import NodeRecord


# ---------------------------------------------------------------------------
# Subprocess wrapper
# ---------------------------------------------------------------------------

def _run(
    cmd: list[str],
    *,
    capture_output: bool = True,
    check: bool = True,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# SSH / rsync building blocks
# ---------------------------------------------------------------------------

def ssh_base(node: NodeRecord) -> list[str]:
    """Build the base ``ssh`` command list for a node."""
    ssh_key = str(Path(node.get("ssh_key", "~/.ssh/id_ed25519")).expanduser())
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={node.get('connect_timeout', 12)}",
        "-i", ssh_key,
        f"{node.get('user', 'root')}@{node['ssh_host']}",
        "-p", str(node.get("ssh_port", 22)),
    ]


def scp_to_node(node: NodeRecord, local_path: str, remote_path: str) -> None:
    """Copy a local file to a remote node via scp."""
    ssh_key = str(Path(node.get("ssh_key", "~/.ssh/id_ed25519")).expanduser())
    port = str(node.get("ssh_port", 22))
    user_host = f"{node.get('user', 'root')}@{node['ssh_host']}"
    cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={node.get('connect_timeout', 12)}",
        "-i", ssh_key,
        "-P", port,
        local_path,
        f"{user_host}:{remote_path}",
    ]
    _run(cmd, check=True)


def rsync_base(node: NodeRecord) -> list[str]:
    """Build the base ``rsync`` command list for a node."""
    ssh_key = str(Path(node.get("ssh_key", "~/.ssh/id_ed25519")).expanduser())
    ssh_cmd = " ".join([
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={node.get('connect_timeout', 12)}",
        "-i", ssh_key,
        "-p", str(node.get("ssh_port", 22)),
    ])
    return [
        "rsync",
        "-az",
        "--no-o",
        "--no-g",
        "-e", ssh_cmd,
    ]


# ---------------------------------------------------------------------------
# Remote execution
# ---------------------------------------------------------------------------

def remote_exec(
    node: NodeRecord,
    command: str,
    *,
    check: bool = True,
    timeout: int | None = 120,
) -> subprocess.CompletedProcess[str]:
    """Execute a shell command on a remote node via SSH."""
    return _run(ssh_base(node) + [command], check=check, timeout=timeout)


def remote_python(node: NodeRecord, code: str) -> subprocess.CompletedProcess[str]:
    """Execute a Python snippet on a remote node."""
    py = shlex.quote(node.get("python_bin", "python3"))
    workspace = shlex.quote(node.get("workspace_path", "/workspace/project"))
    command = f"cd {workspace} && {py} - <<'PY'\n{code}\nPY"
    return remote_exec(node, command)


def checked_remote_exec(
    node: NodeRecord,
    label: str,
    command: str,
    *,
    timeout: int | None = 600,
) -> subprocess.CompletedProcess[str]:
    """Execute a command on a remote node; raise RuntimeError on failure."""
    proc = remote_exec(node, command, check=False, timeout=timeout)
    if proc.returncode == 0:
        return proc
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    detail = stderr or stdout
    if len(detail) > 400:
        detail = detail[-400:]
    raise RuntimeError(
        f"{label} failed on {node['name']} ({node['ssh_host']}:{node.get('ssh_port', 22)}) "
        f"rc={proc.returncode}: {detail or 'no output'}"
    )


# ---------------------------------------------------------------------------
# SSH connectivity probe
# ---------------------------------------------------------------------------

def ssh_ok(node: NodeRecord) -> bool:
    """Return True if we can reach the node over SSH."""
    if not node.get("ssh_host"):
        return False
    proc = remote_exec(node, "echo ready", check=False)
    return proc.returncode == 0


def _classify_ssh_failure(
    proc: subprocess.CompletedProcess[str] | None,
    thrown: BaseException | None = None,
) -> str:
    """Classify an SSH failure into one of: not_ready, auth, timeout, other.

    - ``not_ready``: transient connection refused / host unknown during boot.
      Safe to retry with backoff.
    - ``auth``: permission denied; fatal. Fix keys and retry manually.
    - ``timeout``: command exceeded the wall-clock budget. Might be
      retryable for slow steps; caller decides.
    - ``other``: unknown failure; conservative default.
    """
    if isinstance(thrown, subprocess.TimeoutExpired):
        return "timeout"
    if proc is None:
        return "other"
    stderr = (proc.stderr or "").lower()
    # OpenSSH returns 255 for connection errors; the stderr message
    # distinguishes boot-time transients from hard auth failures.
    if proc.returncode == 255:
        if "permission denied" in stderr or "publickey" in stderr:
            return "auth"
        if (
            "connection refused" in stderr
            or "no route to host" in stderr
            or "name or service not known" in stderr
            or "connection reset" in stderr
            or "host is unreachable" in stderr
        ):
            return "not_ready"
        if "connection timed out" in stderr or "connect timeout" in stderr:
            return "timeout"
        return "other"
    return "other"


def wait_for_ssh_ready(
    node: NodeRecord,
    *,
    max_attempts: int = 6,
    backoff_base: int = 5,
    max_wait: int = 180,
) -> None:
    """Wait for a freshly-provisioned node to accept SSH connections.

    Exponential backoff with a hard total-wait budget. Raises:
      - ``SshAuthError``: auth failed (fatal, do not retry)
      - ``SshTimeoutError``: budget exhausted without a successful connect
      - ``SshNotReadyError``: generic transient error that survived all attempts

    Typical call site is right after a provision step, before the first
    bootstrap command. Replaces the pattern of just running a 600s SSH
    command and hoping the pod came up in time.
    """
    import time

    from crucible.core.errors import (
        SshAuthError,
        SshNotReadyError,
        SshTimeoutError,
    )
    from crucible.core.log import log_info, log_warn

    if not node.get("ssh_host"):
        raise SshNotReadyError(
            f"{node.get('name', '?')}: no ssh_host set — cannot wait for SSH readiness"
        )

    started = time.monotonic()
    last_classification = "other"
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        elapsed = time.monotonic() - started
        remaining = max_wait - elapsed
        if remaining <= 0:
            break
        # Short per-attempt timeout so a wedged connection doesn't eat
        # the whole budget.
        attempt_timeout = min(10, int(remaining) + 1)
        proc = None
        try:
            proc = _run(
                ssh_base(node) + ["echo ready"],
                check=False,
                timeout=attempt_timeout,
            )
        except subprocess.TimeoutExpired:
            last_classification = "timeout"
            last_error = f"ssh attempt {attempt} timed out after {attempt_timeout}s"
            log_warn(f"{node['name']}: {last_error}")
        else:
            if proc.returncode == 0 and "ready" in (proc.stdout or ""):
                if attempt > 1:
                    log_info(
                        f"{node['name']}: SSH ready after {attempt} attempt(s) "
                        f"({elapsed:.1f}s)"
                    )
                return
            last_classification = _classify_ssh_failure(proc)
            last_error = (proc.stderr or "").strip() or f"rc={proc.returncode}"
            if last_classification == "auth":
                raise SshAuthError(
                    f"{node['name']}: SSH auth failed — {last_error}"
                )
            log_warn(
                f"{node['name']}: SSH attempt {attempt}/{max_attempts} "
                f"({last_classification}): {last_error}"
            )

        # Exponential backoff, capped by remaining budget.
        sleep_s = backoff_base * (2 ** (attempt - 1))
        remaining = max_wait - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(sleep_s, max(1, int(remaining))))

    total = time.monotonic() - started
    msg = (
        f"{node['name']}: SSH not ready after {total:.1f}s "
        f"({max_attempts} attempts); last error: {last_error}"
    )
    if last_classification == "timeout":
        raise SshTimeoutError(msg)
    raise SshNotReadyError(msg)


# ---------------------------------------------------------------------------
# Repo / env sync
# ---------------------------------------------------------------------------

def sync_repo(
    node: NodeRecord,
    *,
    project_root: Path,
    sync_excludes: list[str],
    enforce_clean: bool = True,
    auto_commit: bool = False,
) -> str | None:
    """Rsync the project directory to a remote node.

    When *enforce_clean* is True (default), verifies the working tree is
    committed before syncing.  Returns the git SHA that was synced, or
    None if enforcement is disabled / not in a git repo.
    """
    git_sha: str | None = None
    if enforce_clean:
        from crucible.runner.fingerprint import ensure_clean_commit

        git_sha = ensure_clean_commit(project_root, auto_commit=auto_commit)

    workspace = node.get("workspace_path", "/workspace/project")
    destination = f"{node.get('user', 'root')}@{node['ssh_host']}:{workspace}/"
    cmd = rsync_base(node)
    for item in sync_excludes:
        cmd.extend(["--exclude", item])
    cmd.extend([str(project_root) + "/", destination])
    _run(cmd, check=True)
    return git_sha


def sync_taps(
    node: NodeRecord,
    *,
    expected_shas: dict[str, str] | None = None,
) -> None:
    """Rsync crucible-hub taps to a remote node.

    Syncs each tap directory under ``~/.crucible-hub/taps/`` to
    ``/workspace/<tap_name>/`` on the remote node, making tap architectures,
    launchers, and data available for external project training.

    When *expected_shas* is provided, verifies each tap's local git SHA
    matches before syncing.  Mismatches are logged as warnings.

    Best-effort: logs a warning on failure instead of aborting bootstrap.
    """
    taps_dir = Path.home() / ".crucible-hub" / "taps"
    if not taps_dir.exists():
        return
    user = node.get("user", "root")
    host = node.get("ssh_host")
    if not host:
        return
    # Verify tap SHAs if expected
    if expected_shas:
        from crucible.core.log import log_warn
        from crucible.runner.fingerprint import safe_git_sha

        for tap_name, expected in expected_shas.items():
            tap_dir = taps_dir / tap_name
            if tap_dir.is_dir():
                actual = safe_git_sha(tap_dir)
                if actual and actual != expected:
                    log_warn(
                        f"sync_taps: tap {tap_name} SHA mismatch "
                        f"(local={actual[:8]}, expected={expected[:8]})"
                    )
    for tap in sorted(taps_dir.iterdir()):
        if not tap.is_dir() or tap.name.startswith("."):
            continue
        remote_path = f"/workspace/{tap.name}"
        destination = f"{user}@{host}:{remote_path}/"
        cmd = rsync_base(node)
        cmd.extend([
            "--exclude", ".git",
            "--exclude", "__pycache__",
            "--exclude", "*.pyc",
            "--exclude", "wandb",
            "--exclude", "data/",
            "--exclude", "checkpoints/",
            "--exclude", "*.h5",
            "--exclude", "*.pt",
            "--exclude", "*.ckpt",
            "--exclude", "*.safetensors",
        ])
        cmd.extend([str(tap) + "/", destination])
        try:
            _run(cmd, check=True)
        except Exception as exc:
            from crucible.core.log import log_warn
            log_warn(f"Tap sync failed for {tap.name} on {node['name']}: {exc}")


def sync_env_file(
    node: NodeRecord,
    *,
    project_root: Path,
) -> None:
    """Rsync the environment file to the remote node."""
    env_source = node.get("env_source", ".env.local")
    source = project_root / env_source
    if not source.exists():
        source = project_root / ".env.local"
    if not source.exists():
        source = project_root / ".env"
    if not source.exists():
        return
    workspace = node.get("workspace_path", "/workspace/project")
    destination = f"{node.get('user', 'root')}@{node['ssh_host']}:{workspace}/{env_source}"
    _run(rsync_base(node) + [str(source), destination], check=True)


# ---------------------------------------------------------------------------
# External project env var forwarding
# ---------------------------------------------------------------------------

ENV_FORWARD_DENYLIST = frozenset({
    "RUNPOD_API_KEY", "ANTHROPIC_API_KEY", "SSH_PRIVATE_KEY",
    "AWS_SECRET_ACCESS_KEY", "GCP_SERVICE_ACCOUNT_KEY",
    "OPENAI_API_KEY", "HF_TOKEN",
})

_SENSITIVE_PATTERNS = ("_SECRET", "_PRIVATE", "_CREDENTIAL")


def write_remote_env(
    node: NodeRecord,
    env_forward: list[str],
    env_set: dict[str, str],
    workspace: str,
    *,
    local_env: dict[str, str] | None = None,
) -> None:
    """Write env vars to a .env file on the pod with safe quoting.

    *env_forward*: keys to read from *local_env* (or os.environ).
    *env_set*: explicit key-value pairs to write.
    Raises ValueError if a denylisted key is requested.
    """
    import logging
    import os

    if local_env is not None:
        source = local_env
    else:
        # Build source from os.environ + local .env files
        source = dict(os.environ)
        # Also read .env files from project root (they may not be in os.environ)
        for env_file in [".env", ".env.local", ".env.runpod.local"]:
            env_path = Path.cwd() / env_file
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in source:
                        source[key] = val
    lines: list[str] = []

    for key in env_forward:
        if key in ENV_FORWARD_DENYLIST:
            raise ValueError(
                f"Refusing to forward denylisted key {key!r}. "
                f"Remove it from env_forward in the project spec."
            )
        if any(pat in key.upper() for pat in _SENSITIVE_PATTERNS):
            logging.warning(
                "Forwarding potentially sensitive key %r to pod. "
                "Ensure this is intentional.", key,
            )
        val = source.get(key, "")
        if val:
            lines.append(f"export {key}={shlex.quote(val)}")

    for key, val in env_set.items():
        lines.append(f"export {key}={shlex.quote(val)}")

    if not lines:
        return

    env_content = "\n".join(lines) + "\n"
    ws = shlex.quote(workspace)
    # Write via heredoc to avoid quoting issues
    heredoc = f"cat > {ws}/.env << 'CRUCIBLE_ENV_EOF'\n{env_content}CRUCIBLE_ENV_EOF"
    remote_exec(node, heredoc)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def local_git_sha(project_root: Path) -> str | None:
    """Return the HEAD commit SHA of the local repo, or None."""
    proc = _run(["git", "rev-parse", "HEAD"], cwd=project_root, check=False)
    value = (proc.stdout or "").strip()
    return value or None
