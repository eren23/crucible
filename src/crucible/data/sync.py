"""Sync local data to fleet nodes via SSH/rsync.

Provides two main operations:

* ``sync_data_to_node`` -- push the local data directory to a remote node.
* ``probe_data_on_node`` -- check which shards already exist on a node.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from crucible.core.log import log_error, log_info, log_step, log_success, log_warn
from crucible.core.types import NodeRecord


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sync_data_to_node(
    node: NodeRecord,
    data_root: Path,
    remote_prefix: str = "data",
    *,
    excludes: list[str] | None = None,
    dry_run: bool = False,
    bwlimit: str | None = None,
) -> bool:
    """Rsync the local *data_root* to *node*.

    Parameters
    ----------
    node:
        Fleet node record.  Must contain ``ssh_host`` and ``user``; optionally
        ``ssh_port``, ``ssh_key``, and ``workspace_path``.
    data_root:
        Local directory to sync (e.g. ``<project>/data``).
    remote_prefix:
        Name of the data directory on the remote side, placed under the
        node's ``workspace_path``.
    excludes:
        Extra rsync ``--exclude`` patterns (e.g. ``[".DS_Store"]``).
    dry_run:
        If ``True``, pass ``--dry-run`` to rsync.
    bwlimit:
        Optional bandwidth limit for rsync (e.g. ``"50m"``).

    Returns
    -------
    ``True`` if rsync exited successfully, ``False`` otherwise.
    """
    ssh_host = node.get("ssh_host", "")
    user = node.get("user", "root")
    port = node.get("ssh_port", 22)
    ssh_key = node.get("ssh_key", "")
    workspace = node.get("workspace_path", "/workspace")

    if not ssh_host:
        log_error(f"Node {node.get('name', '?')} has no ssh_host configured")
        return False

    remote_dest = f"{user}@{ssh_host}:{workspace}/{remote_prefix}/"
    local_src = str(data_root).rstrip("/") + "/"

    log_step(f"Syncing {local_src} -> {remote_dest}")

    cmd = _build_rsync_cmd(
        local_src=local_src,
        remote_dest=remote_dest,
        port=port,
        ssh_key=ssh_key,
        excludes=excludes,
        dry_run=dry_run,
        bwlimit=bwlimit,
    )

    log_info(f"rsync command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        log_success(f"Data synced to {node.get('name', ssh_host)}")
        return True
    else:
        log_error(
            f"rsync to {node.get('name', ssh_host)} failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
        return False


def probe_data_on_node(
    node: NodeRecord,
    remote_prefix: str = "data",
) -> dict[str, Any]:
    """Check what data is already present on *node*.

    Runs ``find`` on the remote via SSH and returns a summary.

    Returns
    -------
    Dict with keys:
        ``reachable`` (bool), ``datasets`` (dict mapping dataset dir names to
        train/val shard counts), ``total_files`` (int).
    """
    ssh_host = node.get("ssh_host", "")
    user = node.get("user", "root")
    port = node.get("ssh_port", 22)
    ssh_key = node.get("ssh_key", "")
    workspace = node.get("workspace_path", "/workspace")

    if not ssh_host:
        return {"reachable": False, "error": "no ssh_host configured", "datasets": {}}

    remote_data = f"{workspace}/{remote_prefix}"
    ssh_cmd = _build_ssh_cmd(user, ssh_host, port, ssh_key)
    find_cmd = ssh_cmd + [f"find {remote_data}/datasets -name '*.bin' -type f 2>/dev/null || true"]

    log_info(f"Probing data on {node.get('name', ssh_host)}:{remote_data}")

    try:
        result = subprocess.run(
            find_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        log_warn(f"SSH probe timed out for {node.get('name', ssh_host)}")
        return {"reachable": False, "error": "timeout", "datasets": {}}

    if result.returncode != 0:
        log_warn(f"SSH probe failed for {node.get('name', ssh_host)}: {result.stderr.strip()}")
        return {"reachable": False, "error": result.stderr.strip(), "datasets": {}}

    datasets: dict[str, dict[str, int]] = {}
    total_files = 0

    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        total_files += 1
        # Expected path: .../datasets/<variant_name>/<shard>.bin
        parts = Path(line).parts
        try:
            ds_idx = parts.index("datasets")
            ds_name = parts[ds_idx + 1]
        except (ValueError, IndexError):
            continue

        if ds_name not in datasets:
            datasets[ds_name] = {"train": 0, "val": 0}

        filename = parts[-1]
        if "_train_" in filename:
            datasets[ds_name]["train"] += 1
        elif "_val_" in filename:
            datasets[ds_name]["val"] += 1

    log_info(
        f"Node {node.get('name', ssh_host)}: {total_files} shard files across "
        f"{len(datasets)} dataset(s)"
    )

    return {
        "reachable": True,
        "datasets": datasets,
        "total_files": total_files,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_ssh_cmd(
    user: str,
    host: str,
    port: int,
    ssh_key: str,
) -> list[str]:
    """Build the base SSH command list."""
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
    ]
    if ssh_key:
        cmd.extend(["-i", str(Path(ssh_key).expanduser())])
    cmd.append(f"{user}@{host}")
    return cmd


def _build_rsync_cmd(
    *,
    local_src: str,
    remote_dest: str,
    port: int,
    ssh_key: str,
    excludes: list[str] | None,
    dry_run: bool,
    bwlimit: str | None,
) -> list[str]:
    """Build a full rsync command list."""
    ssh_inner = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {port}"
    if ssh_key:
        ssh_inner += f" -i {Path(ssh_key).expanduser()}"

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "--partial",
        "-e", ssh_inner,
    ]

    if dry_run:
        cmd.append("--dry-run")

    if bwlimit:
        cmd.extend(["--bwlimit", bwlimit])

    # Default excludes.
    default_excludes = [".DS_Store", "__pycache__", "*.tmp"]
    for excl in default_excludes:
        cmd.extend(["--exclude", excl])

    if excludes:
        for excl in excludes:
            cmd.extend(["--exclude", excl])

    cmd.extend([local_src, remote_dest])
    return cmd
