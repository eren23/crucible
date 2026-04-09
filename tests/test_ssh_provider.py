"""Tests for the SSH provider (Phase 3b).

Covers the no-op provision/destroy semantics and the upgraded
wait_ready that uses wait_for_ssh_ready's exponential backoff.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.core.errors import (
    FleetError,
    SshAuthError,
    SshNotReadyError,
    SshTimeoutError,
)
from crucible.fleet.providers.ssh import SSHProvider


# ---------------------------------------------------------------------------
# Construction + constants
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self):
        p = SSHProvider()
        assert p.provider_name == "ssh"
        assert p.ssh_key == "~/.ssh/id_ed25519"
        assert p.defaults == {}
        assert p.initial_connect == {}

    def test_custom_ssh_key(self):
        p = SSHProvider(ssh_key="/custom/key")
        assert p.ssh_key == "/custom/key"

    def test_initial_connect_overrides(self):
        p = SSHProvider(
            initial_connect={"max_attempts": 3, "backoff_base": 1, "max_wait": 15}
        )
        assert p.initial_connect["max_attempts"] == 3
        assert p.initial_connect["max_wait"] == 15


# ---------------------------------------------------------------------------
# No-op provision/destroy
# ---------------------------------------------------------------------------


class TestNoopOperations:
    def test_provision_returns_empty_and_warns(self, caplog):
        p = SSHProvider()
        result = p.provision(count=5, name_prefix="x")
        assert result == []

    def test_destroy_with_selected_names_removes_them(self):
        p = SSHProvider()
        nodes = [
            {"name": "a", "state": "ready"},
            {"name": "b", "state": "ready"},
            {"name": "c", "state": "ready"},
        ]
        remaining = p.destroy(nodes, selected_names={"b"})
        assert [n["name"] for n in remaining] == ["a", "c"]

    def test_destroy_without_names_returns_empty_inventory(self):
        p = SSHProvider()
        result = p.destroy([{"name": "a"}, {"name": "b"}])
        assert result == []

    def test_provider_has_no_list_all_pods(self):
        """Confirmed: cleanup_orphans() will raise FleetError for SSH."""
        p = SSHProvider()
        assert not hasattr(p, "list_all_pods")


# ---------------------------------------------------------------------------
# refresh — uses ssh_ok to classify state
# ---------------------------------------------------------------------------


class TestRefresh:
    def test_reachable_nodes_become_ready(self, monkeypatch):
        from crucible.fleet.providers import ssh as ssh_module

        monkeypatch.setattr(ssh_module, "ssh_ok", lambda node: True)
        p = SSHProvider()
        result = p.refresh([{"name": "x", "ssh_host": "1.2.3.4", "state": "new"}])
        assert result[0]["state"] == "ready"
        assert "last_seen_at" in result[0]

    def test_unreachable_nodes_become_unreachable(self, monkeypatch):
        from crucible.fleet.providers import ssh as ssh_module

        monkeypatch.setattr(ssh_module, "ssh_ok", lambda node: False)
        p = SSHProvider()
        result = p.refresh([{"name": "y", "ssh_host": "1.2.3.4", "state": "ready"}])
        assert result[0]["state"] == "unreachable"


# ---------------------------------------------------------------------------
# wait_ready — delegates to wait_for_ssh_ready with backoff
# ---------------------------------------------------------------------------


class TestWaitReady:
    def test_all_nodes_ready(self, monkeypatch):
        from crucible.fleet.providers import ssh as ssh_module

        called: list[str] = []

        def fake_wait(node, **kwargs):
            called.append(node["name"])
            # Success — no exception

        monkeypatch.setattr(ssh_module, "wait_for_ssh_ready", fake_wait)
        p = SSHProvider(initial_connect={"max_attempts": 2, "backoff_base": 0, "max_wait": 5})
        nodes = [
            {"name": "a", "ssh_host": "1.2.3.4"},
            {"name": "b", "ssh_host": "5.6.7.8"},
        ]
        result = p.wait_ready(nodes, timeout_seconds=5, poll_seconds=1)
        assert called == ["a", "b"]
        assert all(n["state"] == "ready" for n in result)
        assert all("last_seen_at" in n for n in result)

    def test_auth_error_is_fatal(self, monkeypatch):
        from crucible.fleet.providers import ssh as ssh_module

        def fake_wait(node, **kwargs):
            raise SshAuthError(f"{node['name']}: auth failed")

        monkeypatch.setattr(ssh_module, "wait_for_ssh_ready", fake_wait)
        p = SSHProvider()
        with pytest.raises(SshAuthError):
            p.wait_ready(
                [{"name": "x", "ssh_host": "1.2.3.4"}],
                timeout_seconds=5,
                poll_seconds=1,
            )

    def test_transient_failure_marks_unreachable(self, monkeypatch):
        from crucible.fleet.providers import ssh as ssh_module

        def fake_wait(node, **kwargs):
            raise SshNotReadyError(f"{node['name']}: not ready")

        monkeypatch.setattr(ssh_module, "wait_for_ssh_ready", fake_wait)
        p = SSHProvider()
        result = p.wait_ready(
            [{"name": "x", "ssh_host": "1.2.3.4"}],
            timeout_seconds=5,
            poll_seconds=1,
        )
        assert result[0]["state"] == "unreachable"

    def test_mixed_ready_and_unreachable(self, monkeypatch):
        from crucible.fleet.providers import ssh as ssh_module

        def fake_wait(node, **kwargs):
            if node["name"] == "bad":
                raise SshTimeoutError(f"{node['name']}: timed out")
            # good returns without raising

        monkeypatch.setattr(ssh_module, "wait_for_ssh_ready", fake_wait)
        p = SSHProvider()
        result = p.wait_ready(
            [
                {"name": "good", "ssh_host": "1.2.3.4"},
                {"name": "bad", "ssh_host": "5.6.7.8"},
            ],
            timeout_seconds=5,
            poll_seconds=1,
        )
        states = {n["name"]: n["state"] for n in result}
        assert states == {"good": "ready", "bad": "unreachable"}

    def test_stalled_seconds_raises_when_any_unreachable(self, monkeypatch):
        from crucible.fleet.providers import ssh as ssh_module

        def fake_wait(node, **kwargs):
            raise SshNotReadyError("nope")

        monkeypatch.setattr(ssh_module, "wait_for_ssh_ready", fake_wait)
        p = SSHProvider()
        with pytest.raises(TimeoutError, match="stalled"):
            p.wait_ready(
                [{"name": "x", "ssh_host": "1.2.3.4"}],
                timeout_seconds=5,
                poll_seconds=1,
                stalled_seconds=10,
            )

    def test_initial_connect_overrides_forwarded(self, monkeypatch):
        from crucible.fleet.providers import ssh as ssh_module

        captured_kwargs: dict[str, Any] = {}

        def fake_wait(node, **kwargs):
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(ssh_module, "wait_for_ssh_ready", fake_wait)
        p = SSHProvider(
            initial_connect={
                "max_attempts": 3,
                "backoff_base": 1,
                "max_wait": 20,
            }
        )
        p.wait_ready(
            [{"name": "x", "ssh_host": "1.2.3.4"}],
            timeout_seconds=100,  # overridden by explicit max_wait
            poll_seconds=1,
        )
        assert captured_kwargs["max_attempts"] == 3
        assert captured_kwargs["backoff_base"] == 1
        assert captured_kwargs["max_wait"] == 20


# ---------------------------------------------------------------------------
# load_from_file
# ---------------------------------------------------------------------------


class TestLoadFromFile:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        p = SSHProvider.load_from_file(tmp_path / "nope.json")
        assert p == []

    def test_valid_file_loads_and_tags(self, tmp_path: Path):
        nodes_file = tmp_path / "nodes.json"
        nodes_file.write_text(json.dumps([
            {"name": "a", "ssh_host": "1.2.3.4"},
            {"name": "b", "ssh_host": "5.6.7.8", "provider": "custom"},
        ]))
        loaded = SSHProvider.load_from_file(nodes_file)
        assert loaded[0]["provider"] == "ssh"       # auto-tagged
        assert loaded[1]["provider"] == "custom"    # preserved

    def test_non_list_content_raises(self, tmp_path: Path):
        nodes_file = tmp_path / "nodes.json"
        nodes_file.write_text('{"not": "a list"}')
        with pytest.raises(FleetError, match="JSON list"):
            SSHProvider.load_from_file(nodes_file)
