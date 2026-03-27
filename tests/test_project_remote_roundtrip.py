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
        if "nohup bash -c" in command:
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
        patch("crucible.runner.wandb.fetch_wandb_run_info", return_value={
            "url": "https://wandb.ai/team/demo-wandb/runs/abc123",
            "metrics": {"val_loss": 0.2, "accuracy": 0.8},
        }),
    ):
        launched = run_project({"project_name": "demo", "overrides": {"EPOCHS": "1"}})
        assert "error" not in launched
        assert launched["nodes"][0]["status"] == "launched"

        run_id = launched["run_id"]
        collected = collect_project_results({"run_id": run_id})

    assert any("nohup bash -c" in command for command in observed_commands)
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
