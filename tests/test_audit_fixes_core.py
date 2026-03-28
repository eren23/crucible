"""Tests for audit fix Tiers 1-2 and 5: bugs, silent failures, validations.

All tests are non-torch and run without GPU.
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# 1.1 — tap.py atomic write fd handling
# ---------------------------------------------------------------------------

class TestTapAtomicWrite:
    """Verify tap.py atomic writes work even on error paths."""

    def test_save_taps_yaml_writes_correctly(self, tmp_path: Path):
        from crucible.core.tap import TapManager

        hub = tmp_path / "hub"
        hub.mkdir()
        (hub / "taps").mkdir()
        (hub / "plugins").mkdir()
        tm = TapManager(hub)
        tm._save_taps_yaml([{"name": "test", "url": "http://example.com"}])
        assert tm._taps_file.exists()
        data = yaml.safe_load(tm._taps_file.read_text())
        assert data[0]["name"] == "test"

    def test_save_installed_yaml_writes_correctly(self, tmp_path: Path):
        from crucible.core.tap import TapManager

        hub = tmp_path / "hub"
        hub.mkdir()
        (hub / "taps").mkdir()
        (hub / "plugins").mkdir()
        tm = TapManager(hub)
        tm._save_installed_yaml([{"name": "pkg", "type": "optimizers"}])
        assert tm._installed_file.exists()
        data = yaml.safe_load(tm._installed_file.read_text())
        assert data[0]["name"] == "pkg"


# ---------------------------------------------------------------------------
# 2.1 — plugin_registry.py warns on broken plugin
# ---------------------------------------------------------------------------

class TestPluginRegistryWarning:
    """Broken plugins should log a warning and not silently vanish."""

    def test_broken_plugin_warns(self, tmp_path: Path, capfd: pytest.CaptureFixture[str]):
        from crucible.core.plugin_registry import PluginRegistry

        reg = PluginRegistry("test_type")
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        (plugin_dir / "broken.py").write_text("raise ImportError('intentional')\n")
        (plugin_dir / "good.py").write_text("x = 1\n")

        loaded = reg.load_plugins(plugin_dir, source="test")
        assert "good" in loaded
        assert "broken" not in loaded

        captured = capfd.readouterr()
        assert "Failed to load test_type plugin broken.py" in captured.err

    def test_good_plugins_still_load(self, tmp_path: Path):
        from crucible.core.plugin_registry import PluginRegistry

        reg = PluginRegistry("test_good")
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        (plugin_dir / "alpha.py").write_text("val = 42\n")
        (plugin_dir / "beta.py").write_text("val = 99\n")

        loaded = reg.load_plugins(plugin_dir, source="test")
        assert set(loaded) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# 2.2 — config.py warns when no crucible.yaml found
# ---------------------------------------------------------------------------

class TestConfigFallbackWarning:
    def test_missing_config_warns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]):
        monkeypatch.chdir(tmp_path)
        from crucible.core.config import load_config
        cfg = load_config()
        captured = capfd.readouterr()
        assert "No crucible.yaml found" in captured.err


# ---------------------------------------------------------------------------
# 2.3 — config.py warns on broken project spec
# ---------------------------------------------------------------------------

class TestProjectSpecWarning:
    def test_broken_spec_warns(self, tmp_path: Path, capfd: pytest.CaptureFixture[str]):
        specs_dir = tmp_path / ".crucible" / "projects"
        specs_dir.mkdir(parents=True)
        (specs_dir / "good.yaml").write_text(yaml.dump({"name": "good", "repo": "x"}))
        (specs_dir / "broken.yaml").write_text("{{invalid yaml::")

        from crucible.core.config import list_project_specs
        results = list_project_specs(tmp_path)

        assert len(results) == 1
        assert results[0]["name"] == "good"
        captured = capfd.readouterr()
        assert "Failed to parse project spec broken.yaml" in captured.err


# ---------------------------------------------------------------------------
# 5.2 — finding.py body warning (non-blocking)
# ---------------------------------------------------------------------------

class TestFindingBodyWarning:
    def test_missing_body_warns_but_passes(self, capfd: pytest.CaptureFixture[str]):
        from crucible.core.finding import validate_finding
        errors = validate_finding({"title": "Test finding"})
        assert len(errors) == 0  # no blocking errors
        captured = capfd.readouterr()
        assert "no 'body'" in captured.err

    def test_with_body_no_warning(self, capfd: pytest.CaptureFixture[str]):
        from crucible.core.finding import validate_finding
        errors = validate_finding({"title": "Test", "body": "Details here"})
        assert len(errors) == 0
        captured = capfd.readouterr()
        assert "body" not in captured.err


# ---------------------------------------------------------------------------
# 5.5 — redact.py broader patterns
# ---------------------------------------------------------------------------

class TestRedactBroaderPatterns:
    def test_short_anthropic_key_redacted(self):
        from crucible.core.redact import redact_secrets
        # 17 chars after sk- (would fail with old 20-char minimum)
        result = redact_secrets({"key": "sk-abcdefghijklmnopq"})
        assert result["key"] == "<REDACTED>"

    def test_wandb_v2_key_redacted(self):
        from crucible.core.redact import redact_secrets
        result = redact_secrets({"key": "wandb_v2_abcdefghijklmnopq"})
        assert result["key"] == "<REDACTED>"

    def test_wandb_v1_still_works(self):
        from crucible.core.redact import redact_secrets
        result = redact_secrets({"key": "wandb_v1_abcdefghijklmnopq"})
        assert result["key"] == "<REDACTED>"

    def test_runpod_key_still_works(self):
        from crucible.core.redact import redact_secrets
        result = redact_secrets({"key": "rpd_abcdefghijklmnopq"})
        assert result["key"] == "<REDACTED>"

    def test_hf_key_still_works(self):
        from crucible.core.redact import redact_secrets
        result = redact_secrets({"key": "hf_abcdefghijklmnopq"})
        assert result["key"] == "<REDACTED>"

    def test_non_secret_unchanged(self):
        from crucible.core.redact import redact_secrets
        result = redact_secrets({"name": "hello world"})
        assert result["name"] == "hello world"


# ---------------------------------------------------------------------------
# 5.3 — hyperparams.py bounds validation
# ---------------------------------------------------------------------------

class TestHyperparamsBounds:
    def test_zero_layers_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NUM_LAYERS", "0")
        from crucible.training.hyperparams import Hyperparameters
        # Need to reload to pick up env var at class level
        Hyperparameters.num_layers = 0
        with pytest.raises(ValueError, match="NUM_LAYERS"):
            Hyperparameters()
        Hyperparameters.num_layers = 9  # restore

    def test_negative_dim_raises(self, monkeypatch: pytest.MonkeyPatch):
        from crucible.training.hyperparams import Hyperparameters
        Hyperparameters.model_dim = -1
        with pytest.raises(ValueError, match="MODEL_DIM"):
            Hyperparameters()
        Hyperparameters.model_dim = 512  # restore

    def test_valid_params_pass(self):
        from crucible.training.hyperparams import Hyperparameters
        hp = Hyperparameters()  # should not raise with defaults
        assert hp.num_layers >= 1
        assert hp.model_dim >= 1


# ---------------------------------------------------------------------------
# 5.4 — manifest.py key validation
# ---------------------------------------------------------------------------

class TestManifestKeyValidation:
    def test_missing_name_raises(self):
        from crucible.data.manifest import resolve_shard_paths
        with pytest.raises(ValueError, match="missing required 'name'"):
            resolve_shard_paths({"stats": {"files_train": 1}}, remote_prefix="datasets")
