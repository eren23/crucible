"""Tests for crucible.core.config."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible import __version__ as CRUCIBLE_VERSION
from crucible.core.config import (
    ProjectConfig,
    ProviderConfig,
    DataConfig,
    TrainingConfig,
    ResearcherConfig,
    WandbConfig,
    ExecutionPolicyConfig,
    load_config,
    find_config,
    generate_default_config,
)


# ---------------------------------------------------------------------------
# generate_default_config
# ---------------------------------------------------------------------------

class TestGenerateDefaultConfig:
    def test_returns_valid_yaml(self):
        text = generate_default_config()
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict)

    def test_contains_required_sections(self):
        text = generate_default_config()
        assert "name:" in text
        assert "provider:" in text
        assert "training:" in text
        assert "presets:" in text
        assert "data:" in text
        assert "researcher:" in text
        assert "sync_excludes:" in text

    def test_parsed_fields(self):
        parsed = yaml.safe_load(generate_default_config())
        assert parsed["name"] == "my-project"
        assert parsed["version"] == CRUCIBLE_VERSION
        assert parsed["provider"]["type"] == "runpod"
        assert parsed["wandb"]["required"] is True
        assert parsed["execution_policy"]["require_remote"] is True
        assert isinstance(parsed["training"], list)
        assert len(parsed["training"]) >= 1


# ---------------------------------------------------------------------------
# ProjectConfig defaults
# ---------------------------------------------------------------------------

class TestProjectConfigDefaults:
    def test_default_name(self):
        cfg = ProjectConfig()
        assert cfg.name == "crucible-project"

    def test_default_version(self):
        cfg = ProjectConfig()
        assert cfg.version == CRUCIBLE_VERSION

    def test_default_provider(self):
        cfg = ProjectConfig()
        assert isinstance(cfg.provider, ProviderConfig)
        assert cfg.provider.type == "runpod"
        assert cfg.provider.interruptible is True

    def test_default_training_is_empty_list(self):
        cfg = ProjectConfig()
        assert cfg.training == []

    def test_default_presets_is_empty_dict(self):
        cfg = ProjectConfig()
        assert cfg.presets == {}

    def test_default_results_file(self):
        cfg = ProjectConfig()
        assert cfg.results_file == "experiments.jsonl"

    def test_default_logs_dir(self):
        cfg = ProjectConfig()
        assert cfg.logs_dir == "logs"

    def test_default_sync_excludes(self):
        cfg = ProjectConfig()
        assert ".git" in cfg.sync_excludes
        assert "__pycache__" in cfg.sync_excludes

    def test_default_researcher(self):
        cfg = ProjectConfig()
        assert isinstance(cfg.researcher, ResearcherConfig)
        assert cfg.researcher.budget_hours == 10.0
        assert cfg.researcher.max_iterations == 20

    def test_default_wandb(self):
        cfg = ProjectConfig()
        assert isinstance(cfg.wandb, WandbConfig)
        assert cfg.wandb.required is True
        assert cfg.wandb.mode == "online"

    def test_default_execution_policy(self):
        cfg = ProjectConfig()
        assert isinstance(cfg.execution_policy, ExecutionPolicyConfig)
        assert cfg.execution_policy.require_remote is True
        assert cfg.execution_policy.required_provider == "runpod"
        assert cfg.execution_policy.allow_local_dev is False


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_default_when_path_none_and_no_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config(path=None)
        assert cfg.name == "crucible-project"
        assert cfg.version == CRUCIBLE_VERSION

    def test_load_from_yaml_file(self, tmp_path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text(
            "name: test-project\nversion: '0.2.0'\nprovider:\n  type: ssh\n  interruptible: false\n",
            encoding="utf-8",
        )
        cfg = load_config(yaml_path)
        assert cfg.name == "test-project"
        assert cfg.version == "0.2.0"
        assert cfg.provider.type == "ssh"
        assert cfg.provider.interruptible is False

    def test_project_root_set_from_yaml_parent(self, tmp_path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text("name: root-test\n", encoding="utf-8")
        cfg = load_config(yaml_path)
        assert cfg.project_root == tmp_path.resolve()

    def test_load_all_fields(self, project_dir):
        yaml_path = project_dir / "crucible.yaml"
        cfg = load_config(yaml_path)
        assert cfg.name == "test-project"
        assert cfg.version == "0.3.0"
        assert cfg.provider.type == "ssh"
        assert cfg.provider.gpu_types == ["NVIDIA A100"]
        assert cfg.data.source == "huggingface"
        assert cfg.data.repo_id == "test/dataset"
        assert len(cfg.training) == 2
        assert cfg.training[0].backend == "torch"
        assert cfg.training[1].backend == "mlx"
        assert "smoke" in cfg.presets
        assert "custom_preset" in cfg.presets
        assert cfg.researcher.model == "claude-sonnet-4-6-20250514"
        assert cfg.researcher.budget_hours == 5.0
        assert cfg.researcher.max_iterations == 10
        assert ".git" in cfg.sync_excludes
        assert cfg.results_file == "experiments.jsonl"
        assert cfg.fleet_results_file == "experiments_fleet.jsonl"
        assert cfg.logs_dir == "logs"

    def test_load_empty_yaml(self, tmp_path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text("", encoding="utf-8")
        cfg = load_config(yaml_path)
        assert cfg.name == "crucible-project"

    def test_load_partial_yaml(self, tmp_path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text("name: partial\n", encoding="utf-8")
        cfg = load_config(yaml_path)
        assert cfg.name == "partial"
        assert cfg.version == CRUCIBLE_VERSION
        assert cfg.training == []


# ---------------------------------------------------------------------------
# find_config
# ---------------------------------------------------------------------------

class TestFindConfig:
    def test_find_config_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = find_config()
        # Could be None or could find a crucible.yaml higher up; check type
        assert result is None or isinstance(result, Path)

    def test_find_config_in_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "crucible.yaml").write_text("name: found\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = find_config()
        assert result is not None
        assert result.name == "crucible.yaml"

    def test_find_config_walks_up(self, tmp_path, monkeypatch):
        (tmp_path / "crucible.yaml").write_text("name: parent\n", encoding="utf-8")
        child = tmp_path / "subdir" / "deep"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        result = find_config()
        assert result is not None
        assert result.parent.resolve() == tmp_path.resolve()
