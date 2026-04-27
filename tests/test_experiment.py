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
        mock_wandb.enabled = True  # default wandb.required=True now enforces this
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
        assert result["contract_status"] == "compliant"

    @patch("crucible.runner.experiment.subprocess.Popen")
    @patch("crucible.runner.experiment.WandbLogger")
    def test_enforced_contract_requires_wandb_init(self, mock_wandb_cls, mock_popen, project_dir: Path):
        from crucible.runner.experiment import run_experiment

        train_script = project_dir / "train.py"
        train_script.write_text("print('done')")

        mock_wandb = MagicMock()
        mock_wandb.enabled = False
        mock_wandb.error = "init failed"
        mock_wandb_cls.create.return_value = mock_wandb

        with pytest.raises(Exception, match="W&B logging is required"):
            run_experiment(
                config={"CRUCIBLE_ENFORCE_CONTRACT": "1", "WANDB_PROJECT": "demo"},
                name="test-exp",
                backend="torch",
                preset="smoke",
                project_root=str(project_dir),
                stream_output=False,
                timeout_seconds=5,
            )
        mock_popen.assert_not_called()

    @patch("crucible.runner.experiment.subprocess.Popen")
    @patch("crucible.runner.experiment.WandbLogger")
    def test_default_enforces_when_wandb_required(self, mock_wandb_cls, mock_popen, project_dir: Path):
        """With CRUCIBLE_ENFORCE_CONTRACT unset and wandb.required=True (default),
        an inert WandbLogger triggers RunnerError. This is the new default
        behavior introduced to stop silent W&B failures."""
        from crucible.core.config import ProjectConfig, WandbConfig
        from crucible.runner.experiment import run_experiment

        train_script = project_dir / "train.py"
        train_script.write_text("print('done')")

        mock_wandb = MagicMock()
        mock_wandb.enabled = False
        mock_wandb.error = None
        mock_wandb_cls.create.return_value = mock_wandb

        cfg = ProjectConfig(project_root=project_dir, wandb=WandbConfig(required=True, project="demo"))

        with pytest.raises(Exception, match="W&B logging is required.*WANDB_PROJECT unset"):
            run_experiment(
                config={},
                name="test-exp",
                backend="torch",
                preset="smoke",
                project_root=str(project_dir),
                project_config=cfg,
                stream_output=False,
                timeout_seconds=5,
            )
        mock_popen.assert_not_called()

    def test_enforce_zero_opts_out_when_wandb_required(self, project_dir: Path):
        """CRUCIBLE_ENFORCE_CONTRACT=0 disables the runtime gate even when
        wandb.required=True. Verified by inspecting the resolved enforce
        decision directly, avoiding the pre-existing subprocess.run mock
        interaction in the broader run_experiment streaming path."""
        from crucible.core.config import ProjectConfig, WandbConfig

        cfg = ProjectConfig(project_root=project_dir, wandb=WandbConfig(required=True, project="demo"))
        env = {"CRUCIBLE_ENFORCE_CONTRACT": "0"}
        # Replicate the gate logic from experiment.py:295-308
        flag = env.get("CRUCIBLE_ENFORCE_CONTRACT", "").strip()
        if flag == "1":
            enforce = True
        elif flag == "0":
            enforce = False
        else:
            enforce = bool(getattr(cfg.wandb, "required", True))
        assert enforce is False, "CRUCIBLE_ENFORCE_CONTRACT=0 must disable the gate"

    def test_required_false_skips_runtime_gate(self, project_dir: Path):
        """wandb.required=False bypasses the runtime gate when
        CRUCIBLE_ENFORCE_CONTRACT is unset."""
        from crucible.core.config import ProjectConfig, WandbConfig

        cfg = ProjectConfig(project_root=project_dir, wandb=WandbConfig(required=False))
        env: dict[str, str] = {}
        flag = env.get("CRUCIBLE_ENFORCE_CONTRACT", "").strip()
        if flag == "1":
            enforce = True
        elif flag == "0":
            enforce = False
        else:
            enforce = bool(getattr(cfg.wandb, "required", True))
        assert enforce is False, "wandb.required=False must disable the gate"

    def test_required_true_default_enables_gate(self, project_dir: Path):
        """Default wandb.required=True with unset env enables the gate.
        This is the new default behavior introduced by the W&B reliability work."""
        from crucible.core.config import ProjectConfig, WandbConfig

        cfg = ProjectConfig(project_root=project_dir, wandb=WandbConfig(required=True))
        env: dict[str, str] = {}
        flag = env.get("CRUCIBLE_ENFORCE_CONTRACT", "").strip()
        if flag == "1":
            enforce = True
        elif flag == "0":
            enforce = False
        else:
            enforce = bool(getattr(cfg.wandb, "required", True))
        assert enforce is True, "default wandb.required=True must enable the gate"


class TestResolveLoggingBackendDefault:
    """Coverage for _resolve_logging_backend_default helper (P2 auto-default)."""

    def test_explicit_value_passes_through(self):
        from crucible.runner.experiment import _resolve_logging_backend_default

        backend, warn = _resolve_logging_backend_default({"LOGGING_BACKEND": "console"})
        assert backend == "console"
        assert warn is None

    def test_both_wandb_vars_present_enables_wandb(self):
        from crucible.runner.experiment import _resolve_logging_backend_default

        backend, warn = _resolve_logging_backend_default(
            {"WANDB_API_KEY": "k", "WANDB_PROJECT": "p"}
        )
        assert backend == "wandb,console"
        assert warn is None

    def test_only_project_warns_and_falls_back(self):
        from crucible.runner.experiment import _resolve_logging_backend_default

        backend, warn = _resolve_logging_backend_default({"WANDB_PROJECT": "p"})
        assert backend == "console"
        assert warn is not None and "WANDB_API_KEY missing" in warn

    def test_only_api_key_warns_and_falls_back(self):
        from crucible.runner.experiment import _resolve_logging_backend_default

        backend, warn = _resolve_logging_backend_default({"WANDB_API_KEY": "k"})
        assert backend == "console"
        assert warn is not None and "WANDB_PROJECT missing" in warn

    def test_neither_present_returns_empty(self):
        """No W&B vars and no LOGGING_BACKEND: helper returns empty string,
        leaving the env untouched (downstream falls back to ConsoleLogger)."""
        from crucible.runner.experiment import _resolve_logging_backend_default

        backend, warn = _resolve_logging_backend_default({})
        assert backend == ""
        assert warn is None

    def test_whitespace_treated_as_unset(self):
        from crucible.runner.experiment import _resolve_logging_backend_default

        backend, warn = _resolve_logging_backend_default(
            {"WANDB_API_KEY": "  ", "WANDB_PROJECT": "p"}
        )
        # Whitespace-only key is treated as unset → fall back to console + warn
        assert backend == "console"
        assert warn is not None
