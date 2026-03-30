"""Tests for crucible.fleet.sync — SSH/rsync helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crucible.fleet.sync import (
    rsync_base,
    ssh_base,
    ssh_ok,
    sync_env_file,
    sync_repo,
)


def _make_node(**overrides: Any) -> dict[str, Any]:
    base = {
        "name": "test-node",
        "ssh_host": "10.0.0.1",
        "ssh_port": 22,
        "user": "root",
        "ssh_key": "~/.ssh/id_ed25519",
        "workspace_path": "/workspace/project",
        "connect_timeout": 12,
    }
    base.update(overrides)
    return base


class TestSshBase:
    def test_builds_command(self):
        node = _make_node()
        cmd = ssh_base(node)
        assert "ssh" in cmd
        assert "-o" in cmd
        assert "StrictHostKeyChecking=no" in cmd
        assert "UserKnownHostsFile=/dev/null" in cmd
        assert "BatchMode=yes" in cmd
        assert "root@10.0.0.1" in cmd
        assert "-p" in cmd
        assert "22" in cmd

    def test_custom_port(self):
        node = _make_node(ssh_port=2222)
        cmd = ssh_base(node)
        assert "2222" in cmd

    def test_custom_user(self):
        node = _make_node(user="ubuntu")
        cmd = ssh_base(node)
        assert "ubuntu@10.0.0.1" in cmd


class TestRsyncBase:
    def test_builds_command(self):
        node = _make_node()
        cmd = rsync_base(node)
        assert cmd[0] == "rsync"
        assert "-az" in cmd
        assert "-e" in cmd
        # The -e value should contain ssh options
        e_idx = cmd.index("-e")
        ssh_cmd = cmd[e_idx + 1]
        assert "StrictHostKeyChecking=no" in ssh_cmd
        assert "UserKnownHostsFile=/dev/null" in ssh_cmd


class TestSyncRepo:
    @patch("crucible.fleet.sync._run")
    def test_includes_excludes(self, mock_run, tmp_path: Path):
        node = _make_node()
        sync_repo(node, project_root=tmp_path, sync_excludes=[".git", ".venv", "__pycache__"])
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        # Verify each exclude is present
        exclude_items = []
        for i, arg in enumerate(cmd):
            if arg == "--exclude" and i + 1 < len(cmd):
                exclude_items.append(cmd[i + 1])
        assert ".git" in exclude_items
        assert ".venv" in exclude_items
        assert "__pycache__" in exclude_items

    @patch("crucible.fleet.sync._run")
    def test_destination_includes_workspace(self, mock_run, tmp_path: Path):
        node = _make_node(workspace_path="/custom/workspace")
        sync_repo(node, project_root=tmp_path, sync_excludes=[])
        cmd = mock_run.call_args[0][0]
        assert any("/custom/workspace/" in str(arg) for arg in cmd)


class TestSyncEnvFile:
    @patch("crucible.fleet.sync._run")
    def test_syncs_env_file(self, mock_run, tmp_path: Path):
        env_file = tmp_path / ".env.local"
        env_file.write_text("WANDB_API_KEY=xxx")
        node = _make_node(env_source=".env.local")
        sync_env_file(node, project_root=tmp_path)
        mock_run.assert_called_once()

    @patch("crucible.fleet.sync._run")
    def test_fallback_to_dot_env(self, mock_run, tmp_path: Path):
        """Falls back to .env when env_source doesn't exist."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val")
        node = _make_node(env_source=".env.nonexistent")
        # .env.nonexistent doesn't exist, .env.local doesn't exist, falls to .env
        sync_env_file(node, project_root=tmp_path)
        mock_run.assert_called_once()

    @patch("crucible.fleet.sync._run")
    def test_no_env_file_noop(self, mock_run, tmp_path: Path):
        """No env file at all → no rsync call."""
        node = _make_node(env_source=".env.nonexistent")
        sync_env_file(node, project_root=tmp_path)
        mock_run.assert_not_called()


class TestSshOk:
    @patch("crucible.fleet.sync.remote_exec")
    def test_returns_true_on_success(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0)
        assert ssh_ok(_make_node()) is True

    @patch("crucible.fleet.sync.remote_exec")
    def test_returns_false_on_failure(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=1)
        assert ssh_ok(_make_node()) is False

    def test_returns_false_without_host(self):
        assert ssh_ok({"name": "no-host"}) is False
