"""Tests for crucible.runner.experiment — experiment execution."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crucible.runner.experiment import (
    _resolve_python,
    _resolve_script,
)


class TestResolveScript:
    def test_from_project_config(self, project_dir: Path):
        """Finds script from crucible.yaml training config."""
        train_script = project_dir / "train.py"
        train_script.write_text("# training script")

        from crucible.core.config import load_config
        cfg = load_config(project_dir / "crucible.yaml")

        result = _resolve_script("torch", project_config=cfg, project_root=project_dir)
        assert result == train_script

    def test_fallback_train_backend(self, tmp_path: Path):
        """Falls back to train_{backend}.py convention."""
        script = tmp_path / "train_torch.py"
        script.write_text("# training")

        result = _resolve_script("torch", project_root=tmp_path)
        assert result == script

    def test_fallback_train_py(self, tmp_path: Path):
        """Falls back to train.py."""
        script = tmp_path / "train.py"
        script.write_text("# training")

        result = _resolve_script("torch", project_root=tmp_path)
        assert result == script

    def test_not_found_raises(self, tmp_path: Path):
        """Raises FileNotFoundError when no script found."""
        with pytest.raises(FileNotFoundError, match="No training script found"):
            _resolve_script("torch", project_root=tmp_path)


class TestResolvePython:
    def test_venv_python(self, tmp_path: Path):
        """Returns venv python when it exists."""
        venv_python = tmp_path / ".venv" / "bin" / "python3"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("#!/usr/bin/env python3")

        result = _resolve_python(tmp_path)
        assert result == str(venv_python)

    def test_fallback_to_sys(self, tmp_path: Path):
        """Returns sys.executable when no venv."""
        result = _resolve_python(tmp_path)
        assert result == sys.executable


class TestRunExperiment:
    @patch("crucible.runner.experiment.subprocess.Popen")
    @patch("crucible.runner.experiment.WandbLogger")
    def test_merges_preset_and_config(self, mock_wandb_cls, mock_popen, project_dir: Path):
        """run_experiment merges preset values with config overrides."""
        from crucible.runner.experiment import run_experiment

        # Create a training script
        train_script = project_dir / "train.py"
        train_script.write_text("print('done')")

        # Mock the process
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = ""
        mock_proc.stdout.read.return_value = ""
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        # Mock wandb
        mock_wandb = MagicMock()
        mock_wandb.enabled = False
        mock_wandb_cls.create.return_value = mock_wandb

        # Mock selectors to avoid hanging
        with patch("crucible.runner.experiment.selectors") as mock_sel:
            mock_selector = MagicMock()
            mock_selector.select.return_value = []
            mock_sel.DefaultSelector.return_value = mock_selector

            result = run_experiment(
                config={"LR": "0.001"},
                name="test-exp",
                backend="torch",
                preset="smoke",
                project_root=str(project_dir),
                stream_output=False,
                timeout_seconds=5,
            )

        # Verify environment includes preset values + config
        env_passed = mock_popen.call_args.kwargs.get("env") or mock_popen.call_args[1].get("env", {})
        assert env_passed.get("LR") == "0.001"
        assert "RUN_ID" in env_passed
        assert env_passed.get("RUN_BACKEND") == "torch"
        assert env_passed.get("RUN_PRESET") == "smoke"
        # Smoke preset should set MAX_WALLCLOCK_SECONDS
        assert "MAX_WALLCLOCK_SECONDS" in env_passed
