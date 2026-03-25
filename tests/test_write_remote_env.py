"""Tests for write_remote_env from crucible.fleet.sync."""
from unittest.mock import patch, MagicMock

import pytest

from crucible.fleet.sync import write_remote_env, ENV_FORWARD_DENYLIST


class TestDenylist:
    def test_blocks_runpod_key(self):
        with pytest.raises(ValueError, match="Refusing to forward denylisted key"):
            write_remote_env(
                node={"ssh_host": "x", "name": "n"},
                env_forward=["RUNPOD_API_KEY"],
                env_set={},
                workspace="/workspace",
                local_env={"RUNPOD_API_KEY": "secret"},
            )

    def test_blocks_anthropic_key(self):
        with pytest.raises(ValueError, match="denylisted"):
            write_remote_env(
                node={"ssh_host": "x", "name": "n"},
                env_forward=["ANTHROPIC_API_KEY"],
                env_set={},
                workspace="/workspace",
                local_env={"ANTHROPIC_API_KEY": "secret"},
            )

    def test_all_denylist_entries_blocked(self):
        for key in ENV_FORWARD_DENYLIST:
            with pytest.raises(ValueError):
                write_remote_env(
                    node={"ssh_host": "x", "name": "n"},
                    env_forward=[key],
                    env_set={},
                    workspace="/workspace",
                    local_env={key: "val"},
                )


class TestEnvForwarding:
    @patch("crucible.fleet.sync.remote_exec")
    def test_forwards_requested_keys(self, mock_exec):
        write_remote_env(
            node={"ssh_host": "x", "name": "n"},
            env_forward=["WANDB_API_KEY"],
            env_set={},
            workspace="/workspace",
            local_env={"WANDB_API_KEY": "mykey123", "OTHER": "ignore"},
        )
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][1]
        assert "WANDB_API_KEY" in cmd
        assert "mykey123" in cmd
        assert "OTHER" not in cmd

    @patch("crucible.fleet.sync.remote_exec")
    def test_env_set_written(self, mock_exec):
        write_remote_env(
            node={"ssh_host": "x", "name": "n"},
            env_forward=[],
            env_set={"FOO": "bar", "BAZ": "qux"},
            workspace="/workspace",
            local_env={},
        )
        cmd = mock_exec.call_args[0][1]
        assert "FOO" in cmd
        assert "bar" in cmd
        assert "BAZ" in cmd

    @patch("crucible.fleet.sync.remote_exec")
    def test_empty_does_not_call_remote(self, mock_exec):
        write_remote_env(
            node={"ssh_host": "x", "name": "n"},
            env_forward=[],
            env_set={},
            workspace="/workspace",
            local_env={},
        )
        mock_exec.assert_not_called()

    @patch("crucible.fleet.sync.remote_exec")
    def test_missing_key_skipped(self, mock_exec):
        write_remote_env(
            node={"ssh_host": "x", "name": "n"},
            env_forward=["NONEXISTENT"],
            env_set={"ONLY": "this"},
            workspace="/workspace",
            local_env={},
        )
        cmd = mock_exec.call_args[0][1]
        assert "NONEXISTENT" not in cmd
        assert "ONLY" in cmd
