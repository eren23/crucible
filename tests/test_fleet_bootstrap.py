"""Tests for crucible.fleet.bootstrap — node bootstrap sequence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from crucible.fleet.bootstrap import (
    BOOTSTRAP_ATTEMPTS,
    bootstrap_node,
    bootstrap_node_worker,
)


def _make_node(**overrides: Any) -> dict[str, Any]:
    base = {
        "name": "test-node",
        "ssh_host": "10.0.0.1",
        "ssh_port": 22,
        "user": "root",
        "ssh_key": "~/.ssh/id_ed25519",
        "workspace_path": "/workspace/project",
        "python_bin": "python3",
        "state": "pending",
        "env_ready": False,
        "dataset_ready": False,
    }
    base.update(overrides)
    return base


class TestBootstrapNode:
    @patch("crucible.fleet.bootstrap.local_git_sha", return_value="abc123")
    @patch("crucible.fleet.bootstrap.checked_remote_exec")
    @patch("crucible.fleet.bootstrap.sync_env_file")
    @patch("crucible.fleet.bootstrap.sync_repo")
    def test_full_sequence(self, mock_sync, mock_env, mock_exec, mock_sha, tmp_path: Path):
        """Bootstrap runs sync → env → python check → torch check → pip → data."""
        mock_exec.return_value = MagicMock(stdout="1\n")

        node = _make_node()
        result = bootstrap_node(
            node,
            project_root=tmp_path,
            sync_excludes=[".git"],
            train_shards=1,
        )

        # Verify sync_repo and sync_env_file called first
        mock_sync.assert_called_once()
        mock_env.assert_called_once()
        # Verify remote commands were executed (python_version, torch_import, pip_install, data_probe)
        assert mock_exec.call_count >= 3
        # Verify node state updated
        assert result["state"] == "ready"
        assert result["env_ready"] is True
        assert result["git_sha"] == "abc123"

    @patch("crucible.fleet.bootstrap.local_git_sha", return_value="abc123")
    @patch("crucible.fleet.bootstrap.checked_remote_exec")
    @patch("crucible.fleet.bootstrap.sync_env_file")
    @patch("crucible.fleet.bootstrap.sync_repo")
    def test_skip_install_and_data(self, mock_sync, mock_env, mock_exec, mock_sha, tmp_path: Path):
        """skip_install and skip_data reduce the number of remote calls."""
        node = _make_node()
        result = bootstrap_node(
            node,
            project_root=tmp_path,
            sync_excludes=[],
            train_shards=1,
            skip_install=True,
            skip_data=True,
        )

        # Only python_version and torch_import (no pip, no data)
        assert mock_exec.call_count == 2
        assert result["state"] == "ready"

    @patch("crucible.fleet.bootstrap.local_git_sha", return_value=None)
    @patch("crucible.fleet.bootstrap.checked_remote_exec")
    @patch("crucible.fleet.bootstrap.sync_env_file")
    @patch("crucible.fleet.bootstrap.sync_repo")
    def test_null_git_sha(self, mock_sync, mock_env, mock_exec, mock_sha, tmp_path: Path):
        """Node records null git_sha when local repo has no HEAD."""
        mock_exec.return_value = MagicMock(stdout="1\n")
        node = _make_node()
        result = bootstrap_node(
            node,
            project_root=tmp_path,
            sync_excludes=[],
            train_shards=1,
            skip_install=True,
            skip_data=True,
        )
        assert result["git_sha"] is None


class TestBootstrapNodeWorker:
    @patch("crucible.fleet.bootstrap.upsert_node_record")
    @patch("crucible.fleet.bootstrap.bootstrap_node")
    def test_retries_on_failure(self, mock_bootstrap, mock_upsert, tmp_path: Path):
        """Worker retries up to BOOTSTRAP_ATTEMPTS times."""
        nodes_file = tmp_path / "nodes.json"
        nodes_file.write_text("[]")

        # Fail twice, succeed third time
        mock_bootstrap.side_effect = [
            RuntimeError("SSH timeout"),
            RuntimeError("pip failed"),
            _make_node(state="ready", env_ready=True),
        ]

        with patch("crucible.fleet.bootstrap.time.sleep"):
            result = bootstrap_node_worker(
                _make_node(),
                nodes_file=nodes_file,
                project_root=tmp_path,
                sync_excludes=[],
                train_shards=1,
            )
        assert mock_bootstrap.call_count == 3
        assert result["state"] == "ready"

    @patch("crucible.fleet.bootstrap.bootstrap_node")
    def test_gives_up_after_max_attempts(self, mock_bootstrap, tmp_path: Path):
        """Worker raises after exhausting all retry attempts."""
        nodes_file = tmp_path / "nodes.json"
        nodes_file.write_text("[]")

        mock_bootstrap.side_effect = RuntimeError("persistent failure")

        with patch("crucible.fleet.bootstrap.time.sleep"), \
             pytest.raises(RuntimeError, match="persistent failure"):
            bootstrap_node_worker(
                _make_node(),
                nodes_file=nodes_file,
                project_root=tmp_path,
                sync_excludes=[],
                train_shards=1,
            )
        assert mock_bootstrap.call_count == BOOTSTRAP_ATTEMPTS
