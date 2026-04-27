"""Tests for shared RunPod + W&B experiment contract enforcement."""
from __future__ import annotations

from crucible.core.config import (
    ExecutionPolicyConfig,
    ProjectConfig,
    ProviderConfig,
    WandbConfig,
)
from crucible.core.errors import ConfigError
from crucible.core.experiment_contract import (
    contract_metadata,
    validate_experiment_contract,
)


def test_validate_remote_contract_passes_with_runpod_and_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    cfg = ProjectConfig(
        provider=ProviderConfig(type="runpod"),
        wandb=WandbConfig(project="demo-project"),
    )
    result = validate_experiment_contract(
        cfg,
        action="enqueue",
        execution_mode="remote",
    )
    assert result["execution_provider"] == "runpod"
    assert result["wandb"]["project"] == "demo-project"


def test_validate_local_contract_blocks_by_default(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    cfg = ProjectConfig(
        provider=ProviderConfig(type="runpod"),
        wandb=WandbConfig(project="demo-project"),
    )
    try:
        validate_experiment_contract(
            cfg,
            action="local run",
            execution_mode="local",
        )
    except ConfigError as exc:
        assert "allow_local_dev=true" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_validate_contract_rejects_wrong_provider(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    cfg = ProjectConfig(
        provider=ProviderConfig(type="ssh"),
        wandb=WandbConfig(project="demo-project"),
    )
    try:
        validate_experiment_contract(
            cfg,
            action="enqueue",
            execution_mode="remote",
        )
    except ConfigError as exc:
        assert "provider.type='runpod'" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_validate_contract_rejects_missing_wandb_project(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    cfg = ProjectConfig(provider=ProviderConfig(type="runpod"))
    try:
        validate_experiment_contract(
            cfg,
            action="enqueue",
            execution_mode="remote",
        )
    except ConfigError as exc:
        assert "wandb.project" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_contract_metadata_includes_remote_node(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    cfg = ProjectConfig(
        provider=ProviderConfig(type="runpod"),
        wandb=WandbConfig(project="demo-project"),
        execution_policy=ExecutionPolicyConfig(require_remote=True),
    )
    meta = contract_metadata(cfg, remote_node="gpu-1")
    assert meta["remote_node"] == "gpu-1"
    assert meta["contract_status"] == "compliant"


def test_validate_contract_rejects_missing_wandb_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = ProjectConfig(
        provider=ProviderConfig(type="runpod"),
        wandb=WandbConfig(project="demo-project", mode="online"),
    )
    try:
        validate_experiment_contract(
            cfg,
            action="enqueue",
            execution_mode="remote",
        )
    except ConfigError as exc:
        assert "WANDB_API_KEY" in str(exc)
    else:
        raise AssertionError("expected ConfigError when WANDB_API_KEY missing")


def test_validate_contract_allows_disabled_wandb_mode(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = ProjectConfig(
        provider=ProviderConfig(type="runpod"),
        wandb=WandbConfig(project="demo-project", mode="disabled"),
    )
    # mode=disabled means the API key is not needed even if wandb.required=true
    result = validate_experiment_contract(
        cfg,
        action="enqueue",
        execution_mode="remote",
    )
    assert result["wandb"]["mode"] == "disabled"


def test_validate_contract_skips_wandb_when_not_required(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    cfg = ProjectConfig(
        provider=ProviderConfig(type="runpod"),
        wandb=WandbConfig(required=False),
    )
    result = validate_experiment_contract(
        cfg,
        action="enqueue",
        execution_mode="remote",
    )
    # wandb.required=false: missing project + key is not an error
    assert result["wandb"]["required"] is False
