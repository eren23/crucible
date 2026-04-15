"""Generic SSH provider for manually specified hosts.

Provision and destroy are no-ops -- the user manages hosts externally
(e.g. bare metal, cloud VMs, containers) and registers them in a
``nodes.json`` file.  The provider validates SSH connectivity and
reports health.

This provider is a first-class peer to RunPod: the bootstrap,
dispatch, collect, and autonomous-researcher flows all work against
it with zero RunPod-specific code paths. What SSH can't do:

- **Provision new machines.** You manage ``nodes.json`` by hand.
- **Destroy machines.** ``crucible fleet destroy`` just removes the
  entry from inventory; the actual box keeps running.
- **Orphan detection.** There's no central API to query, so
  ``cleanup_orphans`` raises ``FleetError`` when called against
  an SSH provider.

Everything else — SSH readiness, bootstrap state tracking, per-step
timeouts, dispatch, collect — works identically to RunPod because the
machinery is provider-agnostic.

See ``examples/ssh_local/`` and ``docs/ssh-provider.md`` for a full
walkthrough.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.core.errors import FleetError
from crucible.core.log import log_info, log_warn, utc_now_iso
from crucible.fleet.provider import FleetProvider
from crucible.fleet.sync import ssh_ok, wait_for_ssh_ready


class SSHProvider(FleetProvider):
    """Generic SSH provider that reads nodes from a JSON file.

    Provision and destroy are no-ops -- hosts are managed externally.

    :param ssh_key: path to the private key used for all hosts (each
        node can override via its own ``ssh_key`` field in nodes.json)
    :param defaults: dict merged into every loaded node record — useful
        for default ``workspace_path``, ``python_bin``, etc.
    :param initial_connect: optional overrides for the backoff params
        passed to :func:`wait_for_ssh_ready`. When None, uses the
        defaults (6 attempts, 5s base, 180s budget).
    """

    provider_name = "ssh"

    def __init__(
        self,
        ssh_key: str = "~/.ssh/id_ed25519",
        defaults: dict[str, Any] | None = None,
        *,
        initial_connect: dict[str, int] | None = None,
    ) -> None:
        self.ssh_key = ssh_key
        self.defaults = defaults or {}
        self.initial_connect = initial_connect or {}

    # -- FleetProvider interface ------------------------------------------

    def provision(
        self,
        *,
        count: int,
        name_prefix: str,
        start_index: int = 1,
        replacement: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        log_warn(
            "SSH provider does not support provisioning. "
            "Add hosts to nodes.json manually."
        )
        return []

    def destroy(
        self,
        nodes: list[dict[str, Any]],
        *,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        log_warn(
            "SSH provider does not power off hosts — just removing from "
            "inventory. Your machines are still running."
        )
        if selected_names:
            return [n for n in nodes if n["name"] not in selected_names]
        return []

    def refresh(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        refreshed: list[dict[str, Any]] = []
        for node in nodes:
            updated = dict(node)
            if ssh_ok(node):
                updated["state"] = "ready"
                updated["last_seen_at"] = utc_now_iso()
            else:
                updated["state"] = "unreachable"
            refreshed.append(updated)
        return refreshed

    def wait_ready(
        self,
        nodes: list[dict[str, Any]],
        *,
        timeout_seconds: int = 300,
        poll_seconds: int = 10,
        stalled_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        """Wait for every node in *nodes* to accept SSH.

        Delegates to ``wait_for_ssh_ready`` (from fleet.sync) for each
        node so the SSH provider picks up exponential backoff,
        auth-failure fast-fail, and per-attempt timeout classification
        for free. Any ``SshAuthError`` raised by a node is re-raised
        immediately (fatal), but transient errors for individual nodes
        don't stop the loop — their state is set to ``unreachable``
        and we continue.
        """
        from crucible.core.errors import (
            SshAuthError,
            SshNotReadyError,
            SshTimeoutError,
        )

        # Use the caller's timeout_seconds as the initial-connect budget
        # per node unless the provider was constructed with explicit
        # overrides.
        max_attempts = int(self.initial_connect.get("max_attempts", 6))
        backoff_base = int(self.initial_connect.get("backoff_base", 5))
        max_wait = int(self.initial_connect.get("max_wait", timeout_seconds))

        current: list[dict[str, Any]] = []
        ready_count = 0
        for node in nodes:
            updated = dict(node)
            try:
                wait_for_ssh_ready(
                    node,
                    max_attempts=max_attempts,
                    backoff_base=backoff_base,
                    max_wait=max_wait,
                )
                updated["state"] = "ready"
                updated["last_seen_at"] = utc_now_iso()
                ready_count += 1
            except SshAuthError:
                # Fatal — bubble up immediately so the user can fix the key
                raise
            except (SshNotReadyError, SshTimeoutError) as exc:
                log_warn(f"{node['name']}: {exc}")
                updated["state"] = "unreachable"
            current.append(updated)

        log_info(f"SSH ready {ready_count}/{len(current)}")

        if stalled_seconds is not None and ready_count < len(current):
            pending_names = [n["name"] for n in current if n.get("state") != "ready"]
            raise TimeoutError(
                f"SSH readiness stalled: {', '.join(pending_names)}"
            )

        return current

    # -- Utility ----------------------------------------------------------

    @classmethod
    def load_from_file(cls, path: Path) -> list[dict[str, Any]]:
        """Load node records from a JSON file."""
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise FleetError(f"Nodes file must contain a JSON list: {path}")
        # Ensure each node has the provider tag
        for node in data:
            node.setdefault("provider", "ssh")
        return data
