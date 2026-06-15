# SkyPilot Fleet Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `skypilot` fleet provider (core builtin) that provisions GPU VMs on GCP via the `sky` CLI, hands Crucible SSH endpoints, and tears them down.

**Architecture:** `SkyPilotProvider(FleetProvider)` shells out to the `sky` CLI (no `import sky`, so zero new core Python dependency) for VM lifecycle, and populates the SSH fields on each `NodeRecord` (`ssh_host`/`ssh_port`/`user`/`ssh_key`) so the existing provider-agnostic bootstrap/dispatch/collect run over SSH. SkyPilot generates its own key and writes a `Host <cluster>` block to `~/.ssh/config`, which we parse for `user`/`identity_file`.

**Tech Stack:** Python, pytest + monkeypatch, the `sky` CLI (SkyPilot ~0.12), GCP.

> Run all pytest with the project venv: `PYTHONPATH=src .venv/bin/python -m pytest ...`. Commit per task on the `feat/skypilot-provider` branch.

---

## Background facts (verified against current code)

- `FleetProvider` ABC (`src/crucible/fleet/provider.py`): required `provision`, `destroy`, `refresh`, `wait_ready`; optional `stop`/`start` (default no-op). Exact signatures used below.
- Registry (`src/crucible/fleet/provider_registry.py`): `register_provider(name, factory)`; built-ins registered at module bottom. `FleetManager._build_provider` calls the factory with `ssh_key, image_name, gpu_type_ids, gpu_count, interruptible, defaults, network_volume_id, template_id, project_name` — the factory keeps what it needs and absorbs the rest via `**kwargs`.
- `NodeRecord` (`src/crucible/core/types.py:137`, `TypedDict, total=False`): `name, node_id, gpu, gpu_count, interruptible, ssh_host, ssh_port, user, ssh_key, workspace_path, python_bin, env_source, state, provider, last_seen_at, replacement` (set only what we have).
- SSH helpers (`src/crucible/fleet/sync.py`): `ssh_ok(node)`, `wait_for_ssh_ready(node, *, max_attempts, backoff_base, max_wait)`. The `ssh.py` provider's `wait_ready` is the proven template (mirrored here).
- Test convention (`tests/test_ssh_provider.py`): construct the provider directly; `monkeypatch.setattr(<module>, "<helper>", fake)` to replace module-level helpers; NodeRecords are plain dicts.

## Out of scope (named follow-ups)

`stop`/`start`; orphan cleanup / cluster listing; `PartialProvisionError`-style auto-recovery (provision is fail-fast here); managed jobs (spot auto-recovery); multi-node clusters; non-GCP clouds (config-agnostic, only GCP dogfooded); extracting a shared `ssh_wait_ready` helper (we duplicate `ssh.py`'s ~30-line loop and note it).

---

## Task 1 (MANUAL, not committed): de-risk SSH-endpoint extraction on GCP

Confirms the version-dependent details the typed provider depends on. **Requires** `sky` installed + GCP creds. Skip if env not set up, but then treat the parsing in Task 3 as spike-unconfirmed.

- [ ] **Step 1: Validate SkyPilot + GCP**

```bash
pip install "skypilot[gcp]"
sky check
```
Expected: `GCP: enabled`.

- [ ] **Step 2: Launch a tiny spot node and inspect the endpoint**

```bash
sky launch -c crucible-spike -y --cloud gcp --gpus L4:1 --use-spot --idle-minutes-to-autostop 15 -- true
sky status --ip crucible-spike
sed -n '/Host crucible-spike/,/^Host /p' ~/.ssh/config
```
Record verbatim: the exact `sky status --ip` output (is it just an IP on the last line?), and the `Host crucible-spike` block's `HostName`, `User`, `IdentityFile`. These pin the parsing in Task 3.

- [ ] **Step 3: Confirm raw ssh works, then tear down**

```bash
ssh crucible-spike 'echo ready'
sky down -y crucible-spike
```
Expected: prints `ready`; cluster removed. Note the `User`/`IdentityFile` values — they feed the Task 3 defaults.

---

## Task 2: Module scaffolding + pure helpers

**Files:**
- Create: `src/crucible/fleet/providers/skypilot.py`
- Test: `tests/test_skypilot_provider.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skypilot_provider.py`:

```python
"""Tests for the SkyPilot fleet provider (CLI-wrapping, mocked)."""
from __future__ import annotations

import pytest

from crucible.core.errors import FleetError
from crucible.fleet.providers.skypilot import (
    SkyPilotProvider,
    _parse_ssh_config,
    _sanitize_cluster_name,
)


class TestConstruction:
    def test_defaults(self):
        p = SkyPilotProvider()
        assert p.provider_name == "skypilot"
        assert p.defaults == {}
        assert p.interruptible is True
        assert p.gpu_count == 1

    def test_custom(self):
        p = SkyPilotProvider(
            defaults={"cloud": "gcp", "region": "us-central1"},
            project_name="demo",
            interruptible=False,
            gpu_type_ids=["L4"],
            gpu_count=2,
        )
        assert p.project_name == "demo"
        assert p.interruptible is False
        assert p.gpu_type_ids == ["L4"]
        assert p.gpu_count == 2


class TestClusterName:
    def test_encodes_project_and_index(self):
        p = SkyPilotProvider(project_name="My Proj")
        name = p.build_cluster_name("day", 3)
        assert name == "crucible-my-proj-day-3"

    def test_sanitize_strips_and_collapses(self):
        assert _sanitize_cluster_name("Crucible__Foo  Bar!!") == "crucible-foo-bar"


class TestAccelerators:
    def test_explicit_accelerators_wins(self):
        p = SkyPilotProvider(defaults={"accelerators": "A100:1"}, gpu_type_ids=["L4"])
        assert p._accelerators() == "A100:1"

    def test_built_from_gpu_type_and_count(self):
        p = SkyPilotProvider(gpu_type_ids=["L4"], gpu_count=2)
        assert p._accelerators() == "L4:2"

    def test_none_when_no_gpu(self):
        p = SkyPilotProvider()
        assert p._accelerators() is None


class TestBuildLaunchCmd:
    def test_spot_gpu_region_autostop_flags(self):
        p = SkyPilotProvider(
            defaults={"cloud": "gcp", "region": "us-central1",
                      "disk_size": 100, "idle_minutes_to_autostop": 30},
            gpu_type_ids=["L4"], gpu_count=1, interruptible=True,
        )
        cmd = p._build_launch_cmd("crucible-demo-day-1")
        assert cmd[:4] == ["launch", "-c", "crucible-demo-day-1", "-y"]
        assert "--cloud" in cmd and "gcp" in cmd
        assert "--gpus" in cmd and "L4:1" in cmd
        assert "--use-spot" in cmd
        assert "--region" in cmd and "us-central1" in cmd
        assert "--disk-size" in cmd and "100" in cmd
        assert "--idle-minutes-to-autostop" in cmd and "30" in cmd
        assert cmd[-2:] == ["--", "true"]

    def test_no_spot_when_not_interruptible(self):
        p = SkyPilotProvider(gpu_type_ids=["L4"], interruptible=False)
        assert "--use-spot" not in p._build_launch_cmd("c")


class TestParseSshConfig:
    def test_extracts_user_identity_hostname(self):
        text = (
            "Host other\n"
            "    HostName 9.9.9.9\n"
            "Host crucible-demo\n"
            "    HostName 1.2.3.4\n"
            "    User gcpuser\n"
            "    IdentityFile ~/.sky/generated/ssh/crucible-demo\n"
            "Host another\n"
            "    User nope\n"
        )
        got = _parse_ssh_config("crucible-demo", text)
        assert got == {
            "hostname": "1.2.3.4",
            "user": "gcpuser",
            "identity_file": "~/.sky/generated/ssh/crucible-demo",
        }

    def test_missing_host_returns_empty(self):
        assert _parse_ssh_config("nope", "Host x\n  User y\n") == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'crucible.fleet.providers.skypilot'`.

- [ ] **Step 3: Create the module with helpers + an instantiable (stubbed) class**

Create `src/crucible/fleet/providers/skypilot.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py -v`
Expected: PASS (all the construction / cluster-name / accelerators / launch-cmd / ssh-config tests).

- [ ] **Step 5: Commit**

```bash
git add src/crucible/fleet/providers/skypilot.py tests/test_skypilot_provider.py
git commit -m "feat(skypilot): provider scaffolding + pure helpers (cluster name, launch cmd, ssh-config parse)"
```

---

## Task 3: Implement `provision` (+ preflight, node-record build)

**Files:**
- Modify: `src/crucible/fleet/providers/skypilot.py`
- Test: `tests/test_skypilot_provider.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_skypilot_provider.py`:

```python
import subprocess as _sp
from crucible.fleet.providers import skypilot as _sky_mod


def _ok_proc(stdout="", stderr="", rc=0):
    return _sp.CompletedProcess(args=["sky"], returncode=rc, stdout=stdout, stderr=stderr)


class TestProvision:
    def test_success_builds_node_record(self, monkeypatch):
        monkeypatch.setattr(_sky_mod.shutil, "which", lambda _n: "/usr/bin/sky")
        monkeypatch.setattr(_sky_mod, "_sky", lambda args, **kw: _ok_proc())
        monkeypatch.setattr(_sky_mod, "_status_ip", lambda c: "1.2.3.4")
        monkeypatch.setattr(
            _sky_mod, "_read_ssh_config_text",
            lambda: ("Host crucible-demo-day-1\n  HostName 1.2.3.4\n"
                     "  User gcpuser\n  IdentityFile ~/.sky/k\n"),
        )
        p = SkyPilotProvider(project_name="demo", gpu_type_ids=["L4"], gpu_count=1,
                             defaults={"workspace_path": "/ws"})
        nodes = p.provision(count=1, name_prefix="day")
        assert len(nodes) == 1
        n = nodes[0]
        assert n["name"] == "crucible-demo-day-1"
        assert n["node_id"] == "crucible-demo-day-1"
        assert n["provider"] == "skypilot"
        assert n["ssh_host"] == "1.2.3.4"
        assert n["ssh_port"] == 22
        assert n["user"] == "gcpuser"
        assert n["ssh_key"] == "~/.sky/k"
        assert n["workspace_path"] == "/ws"
        assert n["state"] == "new"

    def test_launch_failure_raises(self, monkeypatch):
        monkeypatch.setattr(_sky_mod.shutil, "which", lambda _n: "/usr/bin/sky")
        monkeypatch.setattr(_sky_mod, "_sky", lambda args, **kw: _ok_proc(stderr="boom", rc=1))
        p = SkyPilotProvider(project_name="demo", gpu_type_ids=["L4"])
        with pytest.raises(FleetError, match="sky launch failed"):
            p.provision(count=1, name_prefix="day")

    def test_missing_sky_binary_raises(self, monkeypatch):
        monkeypatch.setattr(_sky_mod.shutil, "which", lambda _n: None)
        p = SkyPilotProvider()
        with pytest.raises(FleetError, match="sky CLI not found"):
            p.provision(count=1, name_prefix="day")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py::TestProvision -v`
Expected: FAIL — `provision` raises `NotImplementedError`.

- [ ] **Step 3: Implement `provision` + `_node_record_for`**

In `src/crucible/fleet/providers/skypilot.py`, replace the `provision` stub with:

```python
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
            "ssh_host": cfg.get("hostname", ip),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/crucible/fleet/providers/skypilot.py tests/test_skypilot_provider.py
git commit -m "feat(skypilot): provision via sky launch + SSH-endpoint NodeRecord"
```

---

## Task 4: Implement `refresh`

**Files:**
- Modify: `src/crucible/fleet/providers/skypilot.py`
- Test: `tests/test_skypilot_provider.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_skypilot_provider.py`:

```python
class TestRefresh:
    def test_ip_and_ssh_ok_becomes_ready(self, monkeypatch):
        monkeypatch.setattr(_sky_mod, "_status_ip", lambda c: "5.6.7.8")
        monkeypatch.setattr(_sky_mod, "ssh_ok", lambda node: True)
        p = SkyPilotProvider()
        out = p.refresh([{"name": "c", "node_id": "c", "state": "new"}])
        assert out[0]["ssh_host"] == "5.6.7.8"
        assert out[0]["state"] == "ready"
        assert "last_seen_at" in out[0]

    def test_ip_but_ssh_fails_becomes_unreachable(self, monkeypatch):
        monkeypatch.setattr(_sky_mod, "_status_ip", lambda c: "5.6.7.8")
        monkeypatch.setattr(_sky_mod, "ssh_ok", lambda node: False)
        p = SkyPilotProvider()
        out = p.refresh([{"name": "c", "node_id": "c", "state": "ready"}])
        assert out[0]["state"] == "unreachable"

    def test_no_ip_becomes_lost(self, monkeypatch):
        monkeypatch.setattr(_sky_mod, "_status_ip", lambda c: None)
        p = SkyPilotProvider()
        out = p.refresh([{"name": "c", "node_id": "c", "state": "ready"}])
        assert out[0]["state"] == "lost"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py::TestRefresh -v`
Expected: FAIL — `refresh` raises `NotImplementedError`.

- [ ] **Step 3: Implement `refresh`**

Replace the `refresh` stub with:

```python
    def refresh(self, nodes: list[NodeRecord]) -> list[NodeRecord]:
        refreshed: list[NodeRecord] = []
        for node in nodes:
            updated: NodeRecord = dict(node)  # type: ignore[assignment]
            cluster = node.get("node_id") or node["name"]
            ip = _status_ip(cluster)
            if not ip:
                updated["state"] = "lost"
            else:
                updated["ssh_host"] = ip
                if ssh_ok(updated):
                    updated["state"] = "ready"
                    updated["last_seen_at"] = utc_now_iso()
                else:
                    updated["state"] = "unreachable"
            refreshed.append(updated)
        return refreshed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/crucible/fleet/providers/skypilot.py tests/test_skypilot_provider.py
git commit -m "feat(skypilot): refresh via sky status --ip + ssh probe"
```

---

## Task 5: Implement `wait_ready` + `destroy`

**Files:**
- Modify: `src/crucible/fleet/providers/skypilot.py`
- Test: `tests/test_skypilot_provider.py`

> `wait_ready` duplicates `ssh.py`'s proven loop (delegating to `wait_for_ssh_ready`). Follow-up: extract a shared `ssh_wait_ready` helper so both providers share it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_skypilot_provider.py`:

```python
from crucible.core.errors import SshAuthError, SshNotReadyError


class TestWaitReady:
    def test_all_ready(self, monkeypatch):
        monkeypatch.setattr(_sky_mod, "wait_for_ssh_ready", lambda node, **kw: None)
        p = SkyPilotProvider()
        out = p.wait_ready([{"name": "a", "ssh_host": "1.2.3.4"}],
                           timeout_seconds=5, poll_seconds=1)
        assert out[0]["state"] == "ready"
        assert "last_seen_at" in out[0]

    def test_auth_error_is_fatal(self, monkeypatch):
        def boom(node, **kw):
            raise SshAuthError("bad key")
        monkeypatch.setattr(_sky_mod, "wait_for_ssh_ready", boom)
        p = SkyPilotProvider()
        with pytest.raises(SshAuthError):
            p.wait_ready([{"name": "a", "ssh_host": "1.2.3.4"}],
                         timeout_seconds=5, poll_seconds=1)

    def test_transient_marks_unreachable(self, monkeypatch):
        def boom(node, **kw):
            raise SshNotReadyError("not yet")
        monkeypatch.setattr(_sky_mod, "wait_for_ssh_ready", boom)
        p = SkyPilotProvider()
        out = p.wait_ready([{"name": "a", "ssh_host": "1.2.3.4"}],
                           timeout_seconds=5, poll_seconds=1)
        assert out[0]["state"] == "unreachable"


class TestDestroy:
    def test_selected_names_downs_only_those(self, monkeypatch):
        downed: list[str] = []
        monkeypatch.setattr(_sky_mod, "_sky",
                            lambda args, **kw: (downed.append(args[-1]) or _ok_proc()))
        p = SkyPilotProvider()
        nodes = [
            {"name": "a", "node_id": "a"},
            {"name": "b", "node_id": "b"},
            {"name": "c", "node_id": "c"},
        ]
        survivors = p.destroy(nodes, selected_names={"b"})
        assert [n["name"] for n in survivors] == ["a", "c"]
        assert downed == ["b"]

    def test_no_names_downs_all(self, monkeypatch):
        downed: list[str] = []
        monkeypatch.setattr(_sky_mod, "_sky",
                            lambda args, **kw: (downed.append(args[-1]) or _ok_proc()))
        p = SkyPilotProvider()
        survivors = p.destroy([{"name": "a", "node_id": "a"},
                               {"name": "b", "node_id": "b"}])
        assert survivors == []
        assert sorted(downed) == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py::TestWaitReady tests/test_skypilot_provider.py::TestDestroy -v`
Expected: FAIL — both raise `NotImplementedError`.

- [ ] **Step 3: Implement `wait_ready` + `destroy`**

Replace the `wait_ready` and `destroy` stubs with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/crucible/fleet/providers/skypilot.py tests/test_skypilot_provider.py
git commit -m "feat(skypilot): wait_ready (ssh backoff) + destroy via sky down"
```

---

## Task 6: Register the provider + optional dependency

**Files:**
- Modify: `src/crucible/fleet/provider_registry.py`
- Modify: `pyproject.toml`
- Test: `tests/test_skypilot_provider.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_skypilot_provider.py`:

```python
class TestRegistry:
    def test_build_provider_returns_skypilot(self):
        from crucible.fleet.provider_registry import build_provider, list_providers
        assert "skypilot" in list_providers()
        p = build_provider(
            "skypilot",
            ssh_key="~/.ssh/id",
            gpu_type_ids=["L4"],
            gpu_count=1,
            interruptible=True,
            defaults={"cloud": "gcp"},
            project_name="demo",
            image_name="ignored",
            network_volume_id="ignored",
            template_id="ignored",
        )
        assert isinstance(p, SkyPilotProvider)
        assert p.project_name == "demo"
        assert p.gpu_type_ids == ["L4"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py::TestRegistry -v`
Expected: FAIL — `PluginError: Unknown fleet provider 'skypilot'`.

- [ ] **Step 3: Register the factory**

In `src/crucible/fleet/provider_registry.py`, add a factory above the `# Register built-ins` section (after `_ssh_factory`):

```python
def _skypilot_factory(
    *,
    ssh_key: str = "",
    gpu_type_ids: list[str] | None = None,
    defaults: JsonDict | None = None,
    gpu_count: int = 1,
    project_name: str = "",
    **kwargs: Any,
) -> FleetProvider:
    from crucible.fleet.providers.skypilot import SkyPilotProvider
    return SkyPilotProvider(
        ssh_key=ssh_key,
        defaults=defaults or {},
        project_name=project_name,
        interruptible=bool(kwargs.get("interruptible", True)),
        gpu_type_ids=gpu_type_ids,
        gpu_count=gpu_count,
    )
```

And add to the register block at the bottom:

```python
register_provider("skypilot", _skypilot_factory)
```

- [ ] **Step 4: Add the optional dependency**

In `pyproject.toml`, under `[project.optional-dependencies]` (create the table if it does not exist), add:

```toml
skypilot = ["skypilot[gcp]>=0.12"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add src/crucible/fleet/provider_registry.py pyproject.toml tests/test_skypilot_provider.py
git commit -m "feat(skypilot): register provider + optional skypilot[gcp] extra"
```

---

## Task 7: Docs

**Files:**
- Create: `docs/skypilot-provider.md`

- [ ] **Step 1: Write the doc**

Create `docs/skypilot-provider.md` covering: install (`pip install skypilot[gcp]`), GCP auth (`GOOGLE_APPLICATION_CREDENTIALS` / `gcloud auth application-default login`) + `sky check`, the `crucible.yaml` `provider.type: skypilot` block with the `provider.defaults` keys (`cloud`, `region`, `accelerators`, `disk_size`, `idle_minutes_to_autostop`, `cpus`, `memory`), how cluster names encode the project (`crucible-<project>-<prefix>-<idx>`), the spot/autostop cost-control note, the no-spot-auto-recovery caveat, and the out-of-scope list (stop/start, orphan cleanup, managed jobs, non-GCP). Mirror the structure of `docs/ssh-provider.md`.

- [ ] **Step 2: Commit**

```bash
git add docs/skypilot-provider.md
git commit -m "docs(skypilot): provider setup + config reference"
```

---

## Task 8 (MANUAL, not committed): end-to-end GCP dogfood

Environment-gated; requires GCP creds + quota + spend. Skip in CI.

- [ ] **Step 1: Point a scratch project at skypilot**

In a scratch project's `crucible.yaml`:

```yaml
provider:
  type: skypilot
  interruptible: true
  gpu_types: [L4]
  gpu_count: 1
  defaults:
    cloud: gcp
    region: us-central1
    idle_minutes_to_autostop: 30
    workspace_path: /workspace/project
```

- [ ] **Step 2: Run the lifecycle**

Drive `provision_nodes` → `fleet_refresh` → `bootstrap_nodes` → one tiny experiment via `dispatch_experiments` → `collect_results` → `destroy_nodes` (MCP or `crucible fleet` CLI).
Expected: a cluster comes up on GCP, code syncs, a smoke experiment runs, results collect, and `sky status` shows the cluster gone after destroy.

---

## Verification (whole plan)

1. `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py -v` → all pass (mocked, no cloud).
2. `PYTHONPATH=src .venv/bin/python -m pytest tests/test_provider_registry.py tests/test_ssh_provider.py -v` → still green (no regressions).
3. `build_provider("skypilot", ...)` returns a `SkyPilotProvider`; `provider.type: skypilot` resolves through `FleetManager`.
4. (If env set up) Task 8 runs a real loop on GCP and tears down cleanly.
5. `git log --oneline -6` shows the per-task commits.
```
