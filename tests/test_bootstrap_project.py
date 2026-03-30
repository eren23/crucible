"""Tests for bootstrap_project from crucible.fleet.bootstrap."""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call

import pytest


def _make_spec(**overrides):
    defaults = dict(
        name="testproj", repo="https://r.git", branch="main", shallow=True,
        workspace="/workspace/test", python="3.10",
        install=["numpy"], install_torch="torch==2.6.0", install_flags="--index-url https://x",
        system_packages=[],
        setup=["echo setup"], setup_timeout=60,
        train="python train.py", timeout=0,
        env_forward=["WANDB_API_KEY"], env_set={"FOO": "bar"},
        launcher="", launcher_entry="", local_files=[],
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
        def side_effect(node, label, cmd, **kwargs):
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

    @patch("crucible.fleet.sync.write_remote_env")
    @patch("crucible.fleet.bootstrap.bootstrap_step")
    def test_install_and_setup_steps_use_project_timeout(self, mock_step, mock_env):
        def side_effect(node, label, cmd, **kwargs):
            if label in {"repo_check", "venv_check"}:
                return MagicMock(stdout="missing\n")
            return MagicMock(stdout="ok\n")

        mock_step.side_effect = side_effect

        from crucible.fleet.bootstrap import bootstrap_project

        spec = _make_spec(setup_timeout=900)
        bootstrap_project(_make_node(), spec)

        timeouts_by_label = {
            c.args[1]: c.kwargs.get("timeout")
            for c in mock_step.call_args_list
            if c.args[1] in {
                "system_tools",
                "repo_clone",
                "install_uv",
                "create_venv",
                "install_torch",
                "install_numpy",
                "setup_0",
            }
        }

        assert timeouts_by_label == {
            "system_tools": 900,
            "repo_clone": 900,
            "install_uv": 900,
            "create_venv": 900,
            "install_torch": 900,
            "install_numpy": 900,
            "setup_0": 900,
        }

    @patch("crucible.fleet.sync.write_remote_env")
    @patch("crucible.fleet.bootstrap.bootstrap_step")
    def test_system_tools_step_installs_generic_packages(self, mock_step, mock_env):
        mock_step.return_value = MagicMock(stdout="exists\n")

        from crucible.fleet.bootstrap import bootstrap_project

        bootstrap_project(_make_node(), _make_spec(system_packages=["ffmpeg"]))

        system_tools_call = next(c for c in mock_step.call_args_list if c.args[1] == "system_tools")
        command = system_tools_call.args[2]
        assert "apt-get install -y git rsync curl ffmpeg" in command
        assert "apk add --no-cache git rsync curl ffmpeg" in command
