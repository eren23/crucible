"""Tests for fleet/project_runner.py — launch, check, collect."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from crucible.core.errors import RunnerError
from crucible.fleet.project_runner import (
    launch_project,
    check_project_running,
    collect_project_result,
)


def _make_spec(**overrides):
    defaults = dict(
        name="testproj", repo="https://r.git", workspace="/workspace/test",
        python="3.10", train="python train.py", branch="main", shallow=True,
        install=[], install_torch="", install_flags="", setup=[],
        env_forward=[], env_set={}, timeout=0,
        metrics=SimpleNamespace(source="stdout", primary="val_loss", direction="minimize"),
        pod=SimpleNamespace(image="", gpu_type="", container_disk=0, volume_disk=0),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_node(**overrides):
    defaults = dict(name="node-01", ssh_host="1.2.3.4", ssh_port=22)
    defaults.update(overrides)
    return defaults


class TestLaunchProject:
    @patch("crucible.fleet.project_runner.remote_exec")
    def test_parses_pid(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0, stdout="1234\n", stderr="")
        result = launch_project(_make_node(), _make_spec(), "run_1")
        assert result["pid"] == 1234
        assert result["run_id"] == "run_1"
        assert result["node"] == "node-01"

    @patch("crucible.fleet.project_runner.remote_exec")
    def test_raises_on_failure(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=1, stdout="error", stderr="fail")
        with pytest.raises(RunnerError, match="Failed to launch"):
            launch_project(_make_node(), _make_spec(), "run_x")

    @patch("crucible.fleet.project_runner.remote_exec")
    def test_raises_on_bad_pid(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0, stdout="not_a_number\n", stderr="")
        with pytest.raises(RunnerError, match="Could not parse PID"):
            launch_project(_make_node(), _make_spec(), "run_y")

    @patch("crucible.fleet.project_runner.remote_exec")
    def test_overrides_passed(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0, stdout="999\n", stderr="")
        launch_project(_make_node(), _make_spec(), "run_z", overrides={"LR": "0.001"})
        cmd = mock_exec.call_args[0][1]
        assert "LR" in cmd
        assert "0.001" in cmd

    @patch("crucible.fleet.project_runner.remote_exec")
    def test_missing_env_file_does_not_block_launch(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0, stdout="111\n", stderr="")
        launch_project(_make_node(), _make_spec(env_set={}), "run_envless")
        cmd = mock_exec.call_args[0][1]
        assert "if [ -f /workspace/test/.env ]; then source /workspace/test/.env; fi" in cmd
        assert "nohup bash -c" in cmd


class TestCheckProjectRunning:
    @patch("crucible.fleet.project_runner.remote_exec")
    def test_running(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0, stdout="running\n")
        assert check_project_running(_make_node(), 1234) is True

    @patch("crucible.fleet.project_runner.remote_exec")
    def test_stopped(self, mock_exec):
        mock_exec.return_value = MagicMock(returncode=0, stdout="stopped\n")
        assert check_project_running(_make_node(), 1234) is False


class TestCollectProjectResult:
    @patch("crucible.fleet.project_runner._run")
    @patch("crucible.fleet.project_runner.rsync_base", return_value=["rsync"])
    @patch("crucible.fleet.project_runner.check_project_running", return_value=False)
    def test_parses_stdout_metrics(self, mock_check, mock_rsync, mock_run, tmp_path):
        # Write a fake log file
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "run_1.log"
        log_file.write_text(
            "step:100/200 train_loss:0.500000\n"
            "metric:val_loss=0.400000\n"
            "final_generic val_loss:0.350000 val_bpb:0.350000\n"
        )

        # Mock rsync to copy our log file
        def fake_run(cmd, **kw):
            return MagicMock(returncode=0)
        mock_run.side_effect = fake_run

        # Pre-create the log so collect finds it
        local_log = tmp_path / "collect_logs"
        local_log.mkdir()
        (local_log / "run_1.log").write_text(log_file.read_text())

        result = collect_project_result(
            node=_make_node(),
            spec=_make_spec(),
            run_id="run_1",
            pid=999,
            local_logs_dir=local_log,
        )
        assert result["status"] == "completed"
        assert result["result"]["val_loss"] == pytest.approx(0.35)

    @patch("crucible.fleet.project_runner._run")
    @patch("crucible.fleet.project_runner.rsync_base", return_value=["rsync"])
    @patch("crucible.fleet.project_runner.check_project_running", return_value=True)
    def test_running_status(self, mock_check, mock_rsync, mock_run, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        result = collect_project_result(
            node=_make_node(), spec=_make_spec(),
            run_id="run_2", pid=888, local_logs_dir=logs,
        )
        assert result["status"] == "running"
