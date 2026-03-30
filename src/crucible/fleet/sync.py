"""SSH/rsync helpers: ssh_base, rsync_base, remote_exec, sync_repo, sync_env_file."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any


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

def ssh_base(node: dict[str, Any]) -> list[str]:
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


def scp_to_node(node: dict[str, Any], local_path: str, remote_path: str) -> None:
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


def rsync_base(node: dict[str, Any]) -> list[str]:
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
    node: dict[str, Any],
    command: str,
    *,
    check: bool = True,
    timeout: int | None = 120,
) -> subprocess.CompletedProcess[str]:
    """Execute a shell command on a remote node via SSH."""
    return _run(ssh_base(node) + [command], check=check, timeout=timeout)


def remote_python(node: dict[str, Any], code: str) -> subprocess.CompletedProcess[str]:
    """Execute a Python snippet on a remote node."""
    py = shlex.quote(node.get("python_bin", "python3"))
    workspace = shlex.quote(node.get("workspace_path", "/workspace/project"))
    command = f"cd {workspace} && {py} - <<'PY'\n{code}\nPY"
    return remote_exec(node, command)


def checked_remote_exec(
    node: dict[str, Any],
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

def ssh_ok(node: dict[str, Any]) -> bool:
    """Return True if we can reach the node over SSH."""
    if not node.get("ssh_host"):
        return False
    proc = remote_exec(node, "echo ready", check=False)
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Repo / env sync
# ---------------------------------------------------------------------------

def sync_repo(
    node: dict[str, Any],
    *,
    project_root: Path,
    sync_excludes: list[str],
) -> None:
    """Rsync the project directory to a remote node."""
    workspace = node.get("workspace_path", "/workspace/project")
    destination = f"{node.get('user', 'root')}@{node['ssh_host']}:{workspace}/"
    cmd = rsync_base(node)
    for item in sync_excludes:
        cmd.extend(["--exclude", item])
    cmd.extend([str(project_root) + "/", destination])
    _run(cmd, check=True)


def sync_env_file(
    node: dict[str, Any],
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
    node: dict[str, Any],
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
        for env_file in [".env", ".env.local", ".env.runpod.local", ".env.lewm"]:
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
