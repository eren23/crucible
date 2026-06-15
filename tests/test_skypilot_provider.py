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
