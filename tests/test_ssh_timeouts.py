"""Tests for SSH timeout + per-step backoff (Phase 2c).

Covers:
- FleetSSHConfig parsing from yaml
- _resolve_step_timeout precedence (explicit > config > default > None)
- _classify_ssh_failure heuristics for the four error classes
- wait_for_ssh_ready exponential backoff, budget exhaustion, auth fatality
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from crucible.core.config import (
    FleetConfig,
    FleetSSHConfig,
    FleetSSHInitialConnectConfig,
    load_config,
)
from crucible.core.errors import SshAuthError, SshNotReadyError, SshTimeoutError


# ---------------------------------------------------------------------------
# FleetSSHConfig parsing
# ---------------------------------------------------------------------------


class TestFleetSSHConfigParsing:
    def test_defaults_when_yaml_omits_fleet_section(self, tmp_path: Path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text("name: test\n")
        cfg = load_config(yaml_path)
        assert isinstance(cfg.fleet, FleetConfig)
        assert cfg.fleet.ssh.initial_connect.max_attempts == 6
        assert cfg.fleet.ssh.initial_connect.backoff_base == 5
        assert cfg.fleet.ssh.initial_connect.max_wait == 180
        # Default step timeouts include all the core steps
        assert cfg.fleet.ssh.step_timeouts["default"] == 300
        assert cfg.fleet.ssh.step_timeouts["pip_install"] == 900

    def test_user_overrides_merge_with_defaults(self, tmp_path: Path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text(
            yaml.safe_dump({
                "name": "test",
                "fleet": {
                    "ssh": {
                        "initial_connect": {"max_attempts": 10, "max_wait": 300},
                        "step_timeouts": {"default": 500, "my_custom_step": 999},
                    },
                },
            })
        )
        cfg = load_config(yaml_path)
        assert cfg.fleet.ssh.initial_connect.max_attempts == 10
        assert cfg.fleet.ssh.initial_connect.max_wait == 300
        # Unspecified fields keep defaults
        assert cfg.fleet.ssh.initial_connect.backoff_base == 5
        # User override wins
        assert cfg.fleet.ssh.step_timeouts["default"] == 500
        assert cfg.fleet.ssh.step_timeouts["my_custom_step"] == 999
        # Default entries still present for non-overridden steps
        assert cfg.fleet.ssh.step_timeouts["pip_install"] == 900


# ---------------------------------------------------------------------------
# _resolve_step_timeout
# ---------------------------------------------------------------------------


class TestResolveStepTimeout:
    def test_explicit_wins(self):
        from crucible.fleet.bootstrap import _resolve_step_timeout
        assert _resolve_step_timeout("pip_install", 42) == 42

    def test_config_lookup_by_step_name(self, tmp_path: Path, monkeypatch):
        from crucible.fleet import bootstrap as bootstrap_module

        fake_cfg = MagicMock()
        fake_cfg.fleet.ssh.step_timeouts = {"default": 300, "pip_install": 900}
        monkeypatch.setattr(
            "crucible.core.config.load_config", lambda: fake_cfg
        )
        assert bootstrap_module._resolve_step_timeout("pip_install", None) == 900

    def test_falls_back_to_default_entry(self, monkeypatch):
        from crucible.fleet import bootstrap as bootstrap_module
        fake_cfg = MagicMock()
        fake_cfg.fleet.ssh.step_timeouts = {"default": 300}
        monkeypatch.setattr("crucible.core.config.load_config", lambda: fake_cfg)
        assert bootstrap_module._resolve_step_timeout("unknown_step", None) == 300

    def test_returns_none_when_config_missing(self, monkeypatch):
        from crucible.fleet import bootstrap as bootstrap_module

        def boom() -> Any:
            raise RuntimeError("no config")

        monkeypatch.setattr("crucible.core.config.load_config", boom)
        # Must not raise — callers fall back to checked_remote_exec's default
        assert bootstrap_module._resolve_step_timeout("pip_install", None) is None


# ---------------------------------------------------------------------------
# _classify_ssh_failure
# ---------------------------------------------------------------------------


def _fake_proc(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ssh"], returncode=returncode, stdout="", stderr=stderr
    )


class TestClassifySshFailure:
    def test_timeout_exception_classified_as_timeout(self):
        from crucible.fleet.sync import _classify_ssh_failure
        exc = subprocess.TimeoutExpired(cmd="ssh", timeout=5)
        assert _classify_ssh_failure(None, exc) == "timeout"

    def test_permission_denied_is_auth(self):
        from crucible.fleet.sync import _classify_ssh_failure
        proc = _fake_proc(255, "Permission denied (publickey).")
        assert _classify_ssh_failure(proc) == "auth"

    def test_connection_refused_is_not_ready(self):
        from crucible.fleet.sync import _classify_ssh_failure
        proc = _fake_proc(255, "ssh: connect to host 1.2.3.4 port 22: Connection refused")
        assert _classify_ssh_failure(proc) == "not_ready"

    def test_host_unreachable_is_not_ready(self):
        from crucible.fleet.sync import _classify_ssh_failure
        proc = _fake_proc(255, "ssh: Could not resolve hostname abc: Name or service not known")
        assert _classify_ssh_failure(proc) == "not_ready"

    def test_timeout_in_stderr_is_timeout(self):
        from crucible.fleet.sync import _classify_ssh_failure
        proc = _fake_proc(255, "ssh: connect to host foo port 22: Connection timed out")
        assert _classify_ssh_failure(proc) == "timeout"

    def test_rc_zero_is_other(self):
        from crucible.fleet.sync import _classify_ssh_failure
        # rc=0 shouldn't be classified as a failure at all, but the function
        # is defensive
        assert _classify_ssh_failure(_fake_proc(0)) == "other"

    def test_unknown_rc_is_other(self):
        from crucible.fleet.sync import _classify_ssh_failure
        assert _classify_ssh_failure(_fake_proc(1, "command not found")) == "other"


# ---------------------------------------------------------------------------
# wait_for_ssh_ready
# ---------------------------------------------------------------------------


class TestWaitForSshReady:
    def test_success_on_first_attempt(self, monkeypatch):
        from crucible.fleet import sync as sync_module

        calls: list[int] = []

        def fake_run(cmd, check, timeout):
            calls.append(1)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ready\n", stderr=""
            )

        monkeypatch.setattr(sync_module, "_run", fake_run)
        node = {"name": "n1", "ssh_host": "1.2.3.4", "user": "root"}
        sync_module.wait_for_ssh_ready(node, max_attempts=3, backoff_base=1, max_wait=10)
        assert len(calls) == 1

    def test_success_after_transient_failures(self, monkeypatch):
        from crucible.fleet import sync as sync_module

        attempt_count = {"n": 0}

        def fake_run(cmd, check, timeout):
            attempt_count["n"] += 1
            if attempt_count["n"] < 3:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=255,
                    stdout="", stderr="Connection refused",
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ready\n", stderr="",
            )

        monkeypatch.setattr(sync_module, "_run", fake_run)
        # Use tiny backoff so the test runs fast
        node = {"name": "n1", "ssh_host": "1.2.3.4", "user": "root"}
        sync_module.wait_for_ssh_ready(
            node, max_attempts=5, backoff_base=0, max_wait=30
        )
        assert attempt_count["n"] == 3

    def test_auth_failure_is_fatal_immediately(self, monkeypatch):
        from crucible.fleet import sync as sync_module

        def fake_run(cmd, check, timeout):
            return subprocess.CompletedProcess(
                args=cmd, returncode=255,
                stdout="", stderr="Permission denied (publickey)",
            )

        monkeypatch.setattr(sync_module, "_run", fake_run)
        node = {"name": "n1", "ssh_host": "1.2.3.4", "user": "root"}
        with pytest.raises(SshAuthError):
            sync_module.wait_for_ssh_ready(
                node, max_attempts=5, backoff_base=0, max_wait=30
            )

    def test_budget_exhausted_raises_not_ready(self, monkeypatch):
        from crucible.fleet import sync as sync_module

        def fake_run(cmd, check, timeout):
            return subprocess.CompletedProcess(
                args=cmd, returncode=255,
                stdout="", stderr="Connection refused",
            )

        monkeypatch.setattr(sync_module, "_run", fake_run)
        node = {"name": "n1", "ssh_host": "1.2.3.4", "user": "root"}
        with pytest.raises(SshNotReadyError, match="not ready"):
            sync_module.wait_for_ssh_ready(
                node, max_attempts=2, backoff_base=0, max_wait=5
            )

    def test_no_host_raises_immediately(self):
        from crucible.fleet import sync as sync_module

        node = {"name": "n1"}  # no ssh_host
        with pytest.raises(SshNotReadyError, match="no ssh_host"):
            sync_module.wait_for_ssh_ready(node)

    def test_timeout_classification_raises_timeout_error(self, monkeypatch):
        from crucible.fleet import sync as sync_module

        def fake_run(cmd, check, timeout):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(sync_module, "_run", fake_run)
        node = {"name": "n1", "ssh_host": "1.2.3.4", "user": "root"}
        with pytest.raises(SshTimeoutError):
            sync_module.wait_for_ssh_ready(
                node, max_attempts=2, backoff_base=0, max_wait=5
            )
