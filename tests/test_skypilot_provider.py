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
