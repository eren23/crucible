"""Tests for CLI contract enforcement on direct local runs."""
from __future__ import annotations

import argparse

import pytest

from crucible.core.config import ProjectConfig, ProviderConfig, WandbConfig
from crucible.core.errors import ConfigError
from crucible.cli.run_commands import _handle_run


def test_run_experiment_cli_rejects_local_execution(monkeypatch):
    cfg = ProjectConfig(
        provider=ProviderConfig(type="runpod"),
        wandb=WandbConfig(project="demo"),
    )
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setattr("crucible.cli.run_commands.load_config", lambda: cfg)

    args = argparse.Namespace(
        run_command="experiment",
        overrides=[],
        name="demo",
        preset="smoke",
        backend="torch",
        timeout=5,
    )

    with pytest.raises(ConfigError, match="allow_local_dev=true"):
        _handle_run(args)
