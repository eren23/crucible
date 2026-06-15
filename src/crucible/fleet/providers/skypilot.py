"""SkyPilot fleet provider (GCP-first, CLI-wrapping).

Provisions GPU VMs via the ``sky`` CLI — the provider shells out exactly
like the codebase shells to ssh/rsync, so Crucible core gains no Python
dependency on skypilot. The provider's only job is VM lifecycle plus
populating the SSH fields on each NodeRecord; bootstrap/dispatch/collect
run over SSH, provider-agnostic.

SkyPilot generates its own key and writes a ``Host <cluster>`` block to
~/.ssh/config — we parse User + IdentityFile from there (not provider.ssh_key).
"""
from __future__ import annotations

import ipaddress
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from crucible.core.errors import FleetError, PartialProvisionError
from crucible.core.log import log_info, log_warn, utc_now_iso
from crucible.core.types import NodeRecord
from crucible.fleet.provider import FleetProvider
from crucible.fleet.sync import ssh_ok, wait_for_ssh_ready


def _sanitize_cluster_name(raw: str) -> str:
    """Lowercase; non [a-z0-9-] -> '-'; collapse repeats; strip dashes."""
    s = re.sub(r"[^a-z0-9-]+", "-", raw.lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "crucible"


def _sky(args: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """Run a ``sky`` CLI subcommand. Isolated for test monkeypatching."""
    return subprocess.run(
        ["sky", *args], capture_output=True, text=True, timeout=timeout
    )


def _is_ip(token: str) -> bool:
    try:
        ipaddress.ip_address(token)
        return True
    except ValueError:
        return False


def _status_ip(cluster: str) -> str | None:
    """Return the head-node IP for *cluster*, or None if unavailable.

    ``sky status --ip`` prints just the IP on success, but may emit
    deprecation banners or ``not found`` text. We return the last
    *address-shaped* line so a banner never masquerades as an IP; a
    non-zero exit or no address line yields ``None`` (surfaced loudly
    upstream rather than written to ``ssh_host``).
    """
    proc = _sky(["status", "--ip", cluster], timeout=120)
    if proc.returncode != 0:
        return None
    for ln in reversed(proc.stdout.splitlines()):
        tok = ln.strip()
        if _is_ip(tok):
            return tok
    return None


def _read_ssh_config_text() -> str:
    p = Path.home() / ".ssh" / "config"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _parse_ssh_config(cluster: str, text: str) -> dict[str, str]:
    """Parse the ``Host <cluster>`` block -> {user, identity_file, hostname}.

    Handles OpenSSH syntax variants: ``Key Value`` and ``Key=Value``,
    tab or space indentation, multi-pattern ``Host a b`` lines, and
    surrounding quotes on values (e.g. a quoted ``IdentityFile``).
    """
    out: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"[ \t=]+", s, maxsplit=1)
        if len(parts) != 2:
            continue
        key, val = parts[0].lower(), parts[1].strip().strip('"')
        if key == "host":
            in_block = cluster in val.split()
            continue
        if in_block:
            if key == "user":
                out["user"] = val
            elif key == "identityfile":
                out["identity_file"] = val
            elif key == "hostname":
                out["hostname"] = val
    return out


class SkyPilotProvider(FleetProvider):
    """Fleet provider backed by the SkyPilot ``sky`` CLI."""

    provider_name = "skypilot"

    def __init__(
        self,
        ssh_key: str = "",
        defaults: dict[str, Any] | None = None,
        *,
        project_name: str = "",
        interruptible: bool = True,
        gpu_type_ids: list[str] | None = None,
        gpu_count: int = 1,
        initial_connect: dict[str, int] | None = None,
    ) -> None:
        self.ssh_key = ssh_key
        self.defaults = defaults or {}
        self.project_name = project_name
        self.interruptible = interruptible
        self.gpu_type_ids = gpu_type_ids or []
        self.gpu_count = gpu_count
        self.initial_connect = initial_connect or {}

    # -- helpers ----------------------------------------------------------

    def build_cluster_name(self, name_prefix: str, index: int) -> str:
        return _sanitize_cluster_name(
            f"crucible-{self.project_name}-{name_prefix}-{index}"
        )

    def _accelerators(self) -> str | None:
        explicit = self.defaults.get("accelerators")
        if explicit:
            return str(explicit)
        if self.gpu_type_ids:
            return f"{self.gpu_type_ids[0]}:{self.gpu_count}"
        return None

    def _build_launch_cmd(self, cluster: str) -> list[str]:
        cmd = ["launch", "-c", cluster, "-y",
               "--cloud", str(self.defaults.get("cloud", "gcp"))]
        accel = self._accelerators()
        if accel:
            cmd += ["--gpus", accel]
        if self.interruptible:
            cmd += ["--use-spot"]
        if self.defaults.get("region"):
            cmd += ["--region", str(self.defaults["region"])]
        if self.defaults.get("disk_size"):
            cmd += ["--disk-size", str(self.defaults["disk_size"])]
        idle = self.defaults.get("idle_minutes_to_autostop")
        if idle is not None:
            cmd += ["--idle-minutes-to-autostop", str(idle)]
        if self.defaults.get("cpus"):
            cmd += ["--cpus", str(self.defaults["cpus"])]
        if self.defaults.get("memory"):
            cmd += ["--memory", str(self.defaults["memory"])]
        cmd += ["--", "true"]
        return cmd

    def _require_sky(self) -> None:
        if shutil.which("sky") is None:
            raise FleetError(
                "sky CLI not found on $PATH. Install with "
                "`pip install skypilot[gcp]` and run `sky check`."
            )

    # -- FleetProvider interface (filled in by later tasks) ---------------

    def provision(self, *, count: int, name_prefix: str, start_index: int = 1,
                  replacement: bool = False, **kwargs: Any) -> list[NodeRecord]:
        self._require_sky()
        created: list[NodeRecord] = []
        for i in range(count):
            cluster = self.build_cluster_name(name_prefix, start_index + i)
            proc = _sky(self._build_launch_cmd(cluster))
            if proc.returncode != 0:
                if created:
                    log_warn(
                        f"sky launch failed after creating "
                        f"{[n['name'] for n in created]} — run `sky down` to clean up."
                    )
                raise FleetError(
                    f"sky launch failed for {cluster!r}: {proc.stderr[-500:]}"
                )
            created.append(self._node_record_for(cluster, replacement=replacement))
            log_info(f"provisioned skypilot cluster {cluster}")
        return created

    def _node_record_for(self, cluster: str, *, replacement: bool) -> NodeRecord:
        ip = _status_ip(cluster)
        if not ip:
            raise FleetError(
                f"sky launch succeeded but no IP for cluster {cluster!r}"
            )
        cfg = _parse_ssh_config(cluster, _read_ssh_config_text())
        rec: NodeRecord = {
            "name": cluster,
            "node_id": cluster,
            "provider": "skypilot",
            # I2: use the `sky status --ip` value so provision and refresh
            # agree on ssh_host (refresh re-derives it from the same source).
            "ssh_host": ip,
            "ssh_port": 22,
            "user": cfg.get("user", "gcpuser"),
            "ssh_key": cfg.get("identity_file", self.ssh_key),
            "state": "new",
            "interruptible": self.interruptible,
            "gpu_count": self.gpu_count,
            "workspace_path": str(self.defaults.get("workspace_path", "/workspace/project")),
            "python_bin": str(self.defaults.get("python_bin", "python3")),
            "env_source": str(self.defaults.get("env_source", ".env.local")),
            "replacement": replacement,
            "last_seen_at": utc_now_iso(),
        }
        accel = self._accelerators()
        if accel:
            rec["gpu"] = accel
        return rec

    def refresh(self, nodes: list[NodeRecord]) -> list[NodeRecord]:
        refreshed: list[NodeRecord] = []
        for node in nodes:
            updated: NodeRecord = dict(node)  # type: ignore[assignment]
            cluster = node.get("node_id") or node.get("name", "")
            ip = _status_ip(cluster)
            if not ip:
                # I1: a failed `sky status` is ambiguous (transient CLI/API
                # error vs. genuinely gone). Use the recoverable "unreachable"
                # — never the BAD_API_STATE "lost", which would auto-evict a
                # live cluster. (Follow-up: positively detect absence -> "lost"
                # once the GCP spike pins `sky status` output.)
                updated["state"] = "unreachable"
            else:
                updated["ssh_host"] = ip
                if ssh_ok(updated):
                    updated["state"] = "ready"
                    updated["last_seen_at"] = utc_now_iso()
                else:
                    updated["state"] = "unreachable"
            refreshed.append(updated)
        return refreshed

    def wait_ready(self, nodes: list[NodeRecord], *, timeout_seconds: int = 900,
                   poll_seconds: int = 15, stalled_seconds: int | None = None,
                   ) -> list[NodeRecord]:
        from crucible.core.errors import (
            SshAuthError,
            SshNotReadyError,
            SshTimeoutError,
        )

        max_attempts = int(self.initial_connect.get("max_attempts", 6))
        backoff_base = int(self.initial_connect.get("backoff_base", 5))
        max_wait = int(self.initial_connect.get("max_wait", timeout_seconds))

        current: list[NodeRecord] = []
        ready_count = 0
        for node in nodes:
            updated: NodeRecord = dict(node)  # type: ignore[assignment]
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
                raise
            except (SshNotReadyError, SshTimeoutError) as exc:
                log_warn(f"{node['name']}: {exc}")
                updated["state"] = "unreachable"
            current.append(updated)

        log_info(f"SSH ready {ready_count}/{len(current)}")
        if stalled_seconds is not None and ready_count < len(current):
            pending = [n["name"] for n in current if n.get("state") != "ready"]
            raise TimeoutError(f"SSH readiness stalled: {', '.join(pending)}")
        return current

    def destroy(self, nodes: list[NodeRecord], *,
                selected_names: set[str] | None = None) -> list[NodeRecord]:
        survivors: list[NodeRecord] = []
        for node in nodes:
            name = node["name"]
            if selected_names is not None and name not in selected_names:
                survivors.append(node)
                continue
            cluster = node.get("node_id") or name
            proc = _sky(["down", "-y", cluster], timeout=600)
            if proc.returncode != 0:
                log_warn(f"sky down failed for {cluster!r}: {proc.stderr[-300:]}")
        return survivors
