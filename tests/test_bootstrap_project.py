"""Tests for bootstrap_project from crucible.fleet.bootstrap."""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call

import pytest


def _make_spec(**overrides):
    defaults = dict(
        name="testproj", repo="https://r.git", branch="main", shallow=True,
        workspace="/workspace/test", python="3.10",
        install=["numpy"], install_torch="torch==2.6.0", install_flags="--index-url https://x",
        setup=["echo setup"], setup_timeout=60,
        train="python train.py", timeout=0,
        env_forward=["WANDB_API_KEY"], env_set={"FOO": "bar"},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_node(**overrides):
    defaults = dict(name="node-01", ssh_host="1.2.3.4", ssh_port=22)
    defaults.update(overrides)
    return defaults


class TestBootstrapProject:
    @patch("crucible.fleet.sync.write_remote_env")
    @patch("crucible.fleet.bootstrap.bootstrap_step")
    def test_existing_repo_triggers_update(self, mock_step, mock_env):
        # repo_check returns "exists"
        mock_step.return_value = MagicMock(stdout="exists\n")

        from crucible.fleet.bootstrap import bootstrap_project
        result = bootstrap_project(_make_node(), _make_spec())

        # Should have called repo_check then repo_update (not repo_clone)
        step_labels = [c.args[1] for c in mock_step.call_args_list]
        assert "repo_check" in step_labels
        assert "repo_update" in step_labels
        assert "repo_clone" not in step_labels

    @patch("crucible.fleet.sync.write_remote_env")
    @patch("crucible.fleet.bootstrap.bootstrap_step")
    def test_new_repo_triggers_clone(self, mock_step, mock_env):
        mock_step.return_value = MagicMock(stdout="missing\n")

        from crucible.fleet.bootstrap import bootstrap_project
        result = bootstrap_project(_make_node(), _make_spec())

        step_labels = [c.args[1] for c in mock_step.call_args_list]
        assert "repo_check" in step_labels
        assert "repo_clone" in step_labels
        assert "repo_update" not in step_labels

    @patch("crucible.fleet.sync.write_remote_env")
    @patch("crucible.fleet.bootstrap.bootstrap_step")
    def test_venv_skipped_if_exists(self, mock_step, mock_env):
        # repo_check=exists, venv_check=exists
        def side_effect(node, label, cmd):
            if "venv_check" in label:
                return MagicMock(stdout="exists\n")
            return MagicMock(stdout="exists\n")
        mock_step.side_effect = side_effect

        from crucible.fleet.bootstrap import bootstrap_project
        result = bootstrap_project(_make_node(), _make_spec())

        step_labels = [c.args[1] for c in mock_step.call_args_list]
        assert "create_venv" not in step_labels

    @patch("crucible.fleet.sync.write_remote_env")
    @patch("crucible.fleet.bootstrap.bootstrap_step")
    def test_node_marked_ready(self, mock_step, mock_env):
        mock_step.return_value = MagicMock(stdout="exists\n")

        from crucible.fleet.bootstrap import bootstrap_project
        node = _make_node()
        result = bootstrap_project(node, _make_spec())

        assert result["state"] == "ready"
        assert result["env_ready"] is True
        assert result["project"] == "testproj"

    @patch("crucible.fleet.sync.write_remote_env")
    @patch("crucible.fleet.bootstrap.bootstrap_step")
    def test_env_forwarding_called(self, mock_step, mock_env):
        mock_step.return_value = MagicMock(stdout="exists\n")

        from crucible.fleet.bootstrap import bootstrap_project
        spec = _make_spec()
        bootstrap_project(_make_node(), spec)

        mock_env.assert_called_once()
        kwargs = mock_env.call_args
        assert kwargs[1]["env_forward"] == ["WANDB_API_KEY"]
        assert kwargs[1]["env_set"] == {"FOO": "bar"}

    @patch("crucible.fleet.sync.write_remote_env")
    @patch("crucible.fleet.bootstrap.bootstrap_step")
    def test_no_python_skips_venv(self, mock_step, mock_env):
        mock_step.return_value = MagicMock(stdout="exists\n")

        from crucible.fleet.bootstrap import bootstrap_project
        spec = _make_spec(python="")
        bootstrap_project(_make_node(), spec)

        step_labels = [c.args[1] for c in mock_step.call_args_list]
        assert "venv_check" not in step_labels
        assert "create_venv" not in step_labels
