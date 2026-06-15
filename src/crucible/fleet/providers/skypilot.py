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

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from crucible.core.errors import FleetError
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


def _status_ip(cluster: str) -> str | None:
    """Return the head-node IP for *cluster*, or None if unavailable."""
    proc = _sky(["status", "--ip", cluster], timeout=120)
    if proc.returncode != 0:
        return None
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _read_ssh_config_text() -> str:
    p = Path.home() / ".ssh" / "config"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _parse_ssh_config(cluster: str, text: str) -> dict[str, str]:
    """Parse the ``Host <cluster>`` block -> {user, identity_file, hostname}."""
    out: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("host "):
            in_block = s.split(None, 1)[1].strip() == cluster
            continue
        if in_block:
            key, _, val = s.partition(" ")
            k, v = key.strip().lower(), val.strip()
            if k == "user":
                out["user"] = v
            elif k == "identityfile":
                out["identity_file"] = v
            elif k == "hostname":
                out["hostname"] = v
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
        raise NotImplementedError

    def refresh(self, nodes: list[NodeRecord]) -> list[NodeRecord]:
        raise NotImplementedError

    def wait_ready(self, nodes: list[NodeRecord], *, timeout_seconds: int = 900,
                   poll_seconds: int = 15, stalled_seconds: int | None = None,
                   ) -> list[NodeRecord]:
        raise NotImplementedError

    def destroy(self, nodes: list[NodeRecord], *,
                selected_names: set[str] | None = None) -> list[NodeRecord]:
        raise NotImplementedError
