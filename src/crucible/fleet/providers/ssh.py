"""Generic SSH provider for manually specified hosts.

Provision and destroy are no-ops -- the user manages hosts externally
(e.g. bare metal, cloud VMs, containers) and registers them in a
``nodes.json`` file.  The provider validates SSH connectivity and
reports health.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from crucible.core.errors import FleetError
from crucible.core.log import log_info, log_warn, utc_now_iso
from crucible.fleet.provider import FleetProvider
from crucible.fleet.sync import ssh_ok


class SSHProvider(FleetProvider):
    """Generic SSH provider that reads nodes from a JSON file.

    Provision and destroy are no-ops -- hosts are managed externally.
    """

    provider_name = "ssh"

    def __init__(
        self,
        ssh_key: str = "~/.ssh/id_ed25519",
        defaults: dict[str, Any] | None = None,
    ) -> None:
        self.ssh_key = ssh_key
        self.defaults = defaults or {}

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
        log_warn("SSH provider does not support destroying nodes.")
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
        deadline = time.time() + timeout_seconds
        current = nodes
        last_ready = -1
        last_progress = time.time()
        while time.time() < deadline:
            current = self.refresh(current)
            pending = [n for n in current if n.get("state") != "ready"]
            ready = len(current) - len(pending)
            if ready != last_ready:
                log_info(f"SSH ready {ready}/{len(current)}")
                last_ready = ready
                last_progress = time.time()
            if not pending:
                return current
            if (
                stalled_seconds is not None
                and pending
                and time.time() - last_progress >= stalled_seconds
            ):
                names = ", ".join(n["name"] for n in pending)
                raise TimeoutError(
                    f"SSH readiness stalled for {stalled_seconds}s: {names}",
                )
            time.sleep(poll_seconds)
        if any(n.get("state") != "ready" for n in current):
            names = ", ".join(n["name"] for n in current if n.get("state") != "ready")
            raise TimeoutError(f"Timed out waiting for SSH readiness: {names}")
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
