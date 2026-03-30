"""Fleet-style external project roundtrip tests with mocked transport."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import yaml

from crucible.core.config import ProjectConfig, WandbConfig
from crucible.mcp.tools import collect_project_results, run_project


def _write_project_spec(project_root: Path, name: str) -> None:
    specs_dir = project_root / ".crucible" / "projects"
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / f"{name}.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "repo": "repo/demo",
                "workspace": "/workspace/demo",
                "train": "python train.py --epochs 1",
                "metrics": {"source": "stdout", "primary": "val_loss", "direction": "minimize"},
            }
        ),
        encoding="utf-8",
    )


def test_run_project_and_collect_results_roundtrip(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_project_spec(project_root, "demo")

    cfg = ProjectConfig(
        project_root=project_root,
        nodes_file="nodes.json",
        logs_dir="logs",
        fleet_results_file="experiments_fleet.jsonl",
        wandb=WandbConfig(project="demo-wandb"),
    )

    (project_root / "nodes.json").write_text(
        json.dumps(
            [
                {
                    "name": "node-1",
                    "project": "demo",
                    "env_ready": True,
                    "ssh_host": "10.0.0.5",
                    "ssh_port": 22,
                }
            ]
        ),
        encoding="utf-8",
    )

    observed_commands: list[str] = []

    def fake_remote_exec(node, command, *, check=True):
        observed_commands.append(command)
        if "start_new_session=True" in command:
            return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout="4242\n", stderr="")
        if "kill -0 4242" in command:
            return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout="stopped\n", stderr="")
        return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, capture_output=True, check=True, cwd=None):
        if cmd and cmd[0] == "rsync":
            local_log = Path(cmd[-1])
            local_log.parent.mkdir(parents=True, exist_ok=True)
            local_log.write_text(
                "step:1/1 train_loss:1.000000\n"
                "metric:val_loss=0.250000\n"
                "metric:accuracy=0.750000\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with (
        patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=False),
        patch("crucible.mcp.tools._get_config", return_value=cfg),
        patch("crucible.fleet.project_runner.remote_exec", side_effect=fake_remote_exec),
        patch("crucible.fleet.project_runner._run", side_effect=fake_run),
        patch("crucible.runner.wandb_logger.fetch_wandb_run_info", return_value={
            "url": "https://wandb.ai/team/demo-wandb/runs/abc123",
            "metrics": {"val_loss": 0.2, "accuracy": 0.8},
        }),
    ):
        launched = run_project({"project_name": "demo", "overrides": {"EPOCHS": "1"}})
        assert "error" not in launched
        assert launched["nodes"][0]["status"] == "launched"

        run_id = launched["run_id"]
        collected = collect_project_results({"run_id": run_id})

    assert any("python -c" in command for command in observed_commands)
    assert any("start_new_session=True" in command for command in observed_commands)
    assert any("if [ -f /workspace/demo/.env ]; then source /workspace/demo/.env; fi" in command for command in observed_commands)
    assert any("export EPOCHS=1" in command for command in observed_commands)
    assert any("export WANDB_RUN_NAME=" in command for command in observed_commands)
    assert any("export WANDB_PROJECT=demo-wandb" in command for command in observed_commands)

    assert collected["run_id"] == run_id
    assert collected["status"] == "completed"
    assert collected["metrics"]["val_loss"] == 0.2
    assert collected["metrics"]["accuracy"] == 0.8
    assert collected["wandb"]["run_name"] == run_id
    assert collected["contract_status"] == "compliant"
    assert (project_root / "logs" / f"{run_id}.log").exists()

    fleet_results = project_root / "experiments_fleet.jsonl"
    assert fleet_results.exists()
    persisted = [json.loads(line) for line in fleet_results.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert persisted[-1]["id"] == run_id
    assert persisted[-1]["config"]["EPOCHS"] == "1"

    project_runs = project_root / ".crucible" / "projects" / "runs.jsonl"
    assert project_runs.exists()
    records = [json.loads(line) for line in project_runs.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = records[-1]
    assert latest["run_id"] == run_id
    assert latest["status"] == "completed"
    assert latest["overrides"]["EPOCHS"] == "1"
    assert latest["resolved_overrides"]["WANDB_RUN_NAME"] == run_id


def test_multi_node_launch_uses_launch_id_and_per_node_run_ids(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_project_spec(project_root, "demo")

    cfg = ProjectConfig(
        project_root=project_root,
        nodes_file="nodes.json",
        logs_dir="logs",
        fleet_results_file="experiments_fleet.jsonl",
        wandb=WandbConfig(project="demo-wandb"),
    )

    (project_root / "nodes.json").write_text(
        json.dumps(
            [
                {"name": "node-1", "project": "demo", "env_ready": True, "ssh_host": "10.0.0.5", "ssh_port": 22},
                {"name": "node-2", "project": "demo", "env_ready": True, "ssh_host": "10.0.0.6", "ssh_port": 22},
            ]
        ),
        encoding="utf-8",
    )

    def fake_remote_exec(node, command, *, check=True):
        if "start_new_session=True" in command:
            pid = "4242" if node["name"] == "node-1" else "4343"
            return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout=f"{pid}\n", stderr="")
        if "kill -0" in command:
            return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout="stopped\n", stderr="")
        return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout="", stderr="")

    def fake_run(cmd, *, capture_output=True, check=True, cwd=None):
        if cmd and cmd[0] == "rsync":
            local_log = Path(cmd[-1])
            local_log.parent.mkdir(parents=True, exist_ok=True)
            local_log.write_text(
                "step:1/1 train_loss:1.000000\n"
                "final_generic val_loss:0.250000 val_bpb:0.250000\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with (
        patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=False),
        patch("crucible.mcp.tools._get_config", return_value=cfg),
        patch("crucible.fleet.project_runner.remote_exec", side_effect=fake_remote_exec),
        patch("crucible.fleet.project_runner._run", side_effect=fake_run),
        patch("crucible.runner.wandb_logger.fetch_wandb_run_info", return_value={}),
    ):
        launched = run_project({"project_name": "demo"})
        assert "launch_id" in launched
        assert "run_id" not in launched
        assert len(launched["nodes"]) == 2
        run_ids = {node["run_id"] for node in launched["nodes"]}
        assert len(run_ids) == 2

        collected = collect_project_results({"launch_id": launched["launch_id"]})

    assert collected["summary"]["completed"] == 2
    assert len(collected["runs"]) == 2
