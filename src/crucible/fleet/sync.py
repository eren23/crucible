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
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        capture_output=capture_output,
        text=True,
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
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={node.get('connect_timeout', 12)}",
        "-i", ssh_key,
        f"{node.get('user', 'root')}@{node['ssh_host']}",
        "-p", str(node.get("ssh_port", 22)),
    ]


def rsync_base(node: dict[str, Any]) -> list[str]:
    """Build the base ``rsync`` command list for a node."""
    ssh_key = str(Path(node.get("ssh_key", "~/.ssh/id_ed25519")).expanduser())
    ssh_cmd = " ".join([
        "ssh",
        "-o", "StrictHostKeyChecking=no",
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
) -> subprocess.CompletedProcess[str]:
    """Execute a shell command on a remote node via SSH."""
    return _run(ssh_base(node) + [command], check=check)


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
) -> subprocess.CompletedProcess[str]:
    """Execute a command on a remote node; raise RuntimeError on failure."""
    proc = remote_exec(node, command, check=False)
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
# Git helpers
# ---------------------------------------------------------------------------

def local_git_sha(project_root: Path) -> str | None:
    """Return the HEAD commit SHA of the local repo, or None."""
    proc = _run(["git", "rev-parse", "HEAD"], cwd=project_root, check=False)
    value = (proc.stdout or "").strip()
    return value or None
