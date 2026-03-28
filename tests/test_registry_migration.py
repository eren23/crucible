"""Tests for data_adapters and objectives migration to PluginRegistry.

Verifies backward compatibility: same public API, dict aliases work,
new features (list_detailed, source kwarg) available.
"""
from __future__ import annotations

import pytest

from crucible.core.errors import PluginError


# ===================================================================
# Data Adapter Registry Migration
# ===================================================================

class TestDataAdapterMigration:
    def test_registry_dict_alias_is_same_object(self):
        from crucible.training.data_adapters import DATA_ADAPTER_REGISTRY, _ADAPTER_REGISTRY
        assert DATA_ADAPTER_REGISTRY is _ADAPTER_REGISTRY._registry

    def test_builtins_in_dict(self):
        from crucible.training.data_adapters import DATA_ADAPTER_REGISTRY
        assert "token" in DATA_ADAPTER_REGISTRY
        assert "image_folder" in DATA_ADAPTER_REGISTRY
        assert "synthetic_images" in DATA_ADAPTER_REGISTRY
        assert "synthetic_video" in DATA_ADAPTER_REGISTRY

    def test_dict_subscript_works(self):
        from crucible.training.data_adapters import DATA_ADAPTER_REGISTRY
        cls = DATA_ADAPTER_REGISTRY["token"]
        assert cls is not None

    def test_dict_iteration_works(self):
        from crucible.training.data_adapters import DATA_ADAPTER_REGISTRY
        names = sorted(DATA_ADAPTER_REGISTRY)
        assert "token" in names
        assert len(names) >= 4

    def test_list_data_adapters(self):
        from crucible.training.data_adapters import list_data_adapters
        names = list_data_adapters()
        assert "token" in names
        assert "image_folder" in names

    def test_list_data_adapters_detailed(self):
        from crucible.training.data_adapters import list_data_adapters_detailed
        detailed = list_data_adapters_detailed()
        sources = {d["name"]: d["source"] for d in detailed}
        assert sources["token"] == "builtin"
        assert sources["image_folder"] == "builtin"

    def test_register_with_source_kwarg(self):
        from crucible.training.data_adapters import (
            register_data_adapter, _ADAPTER_REGISTRY, list_data_adapters_detailed
        )

        class FakeAdapter:
            pass

        try:
            register_data_adapter("test_fake_adapter", FakeAdapter, source="local")
            detailed = list_data_adapters_detailed()
            sources = {d["name"]: d["source"] for d in detailed}
            assert sources["test_fake_adapter"] == "local"
        finally:
            _ADAPTER_REGISTRY.unregister("test_fake_adapter")

    def test_register_without_source_uses_builtin(self):
        """Plain register_data_adapter("name", cls) should default to builtin."""
        from crucible.training.data_adapters import register_data_adapter, _ADAPTER_REGISTRY

        class FakeAdapter2:
            pass

        try:
            # This should work because no "test_fake2" is registered yet
            register_data_adapter("test_fake2", FakeAdapter2)
            detailed = _ADAPTER_REGISTRY.list_plugins_detailed()
            sources = {d["name"]: d["source"] for d in detailed}
            assert sources["test_fake2"] == "builtin"
        finally:
            _ADAPTER_REGISTRY.unregister("test_fake2")

    def test_build_data_adapter_unknown_raises_keyerror(self):
        from crucible.training.data_adapters import build_data_adapter
        with pytest.raises(KeyError, match="Unknown data adapter"):
            build_data_adapter("nonexistent_adapter_xyz")


# ===================================================================
# Objectives Registry Migration
# ===================================================================

class TestObjectivesMigration:
    def test_registry_dict_alias_is_same_object(self):
        from crucible.training.objectives import OBJECTIVE_REGISTRY, _OBJECTIVE_REGISTRY
        assert OBJECTIVE_REGISTRY is _OBJECTIVE_REGISTRY._registry

    def test_builtins_in_dict(self):
        from crucible.training.objectives import OBJECTIVE_REGISTRY
        assert "cross_entropy" in OBJECTIVE_REGISTRY
        assert "mse" in OBJECTIVE_REGISTRY
        assert "kl_divergence" in OBJECTIVE_REGISTRY
        assert "composite" in OBJECTIVE_REGISTRY
        assert "diffusion" in OBJECTIVE_REGISTRY
        assert "jepa" in OBJECTIVE_REGISTRY

    def test_dict_subscript_works(self):
        from crucible.training.objectives import OBJECTIVE_REGISTRY
        cls = OBJECTIVE_REGISTRY["cross_entropy"]
        assert cls is not None

    def test_dict_iteration_works(self):
        from crucible.training.objectives import OBJECTIVE_REGISTRY
        names = sorted(OBJECTIVE_REGISTRY)
        assert "cross_entropy" in names
        assert len(names) >= 6

    def test_list_objectives(self):
        from crucible.training.objectives import list_objectives
        names = list_objectives()
        assert "cross_entropy" in names
        assert "diffusion" in names

    def test_list_objectives_detailed(self):
        from crucible.training.objectives import list_objectives_detailed
        detailed = list_objectives_detailed()
        sources = {d["name"]: d["source"] for d in detailed}
        assert sources["cross_entropy"] == "builtin"

    def test_register_with_source_kwarg(self):
        from crucible.training.objectives import register_objective, _OBJECTIVE_REGISTRY

        class FakeObjective:
            pass

        try:
            register_objective("test_fake_obj", FakeObjective, source="local")
            detailed = _OBJECTIVE_REGISTRY.list_plugins_detailed()
            sources = {d["name"]: d["source"] for d in detailed}
            assert sources["test_fake_obj"] == "local"
        finally:
            _OBJECTIVE_REGISTRY.unregister("test_fake_obj")

    def test_build_objective_unknown_raises_keyerror(self):
        from crucible.training.objectives import build_objective
        with pytest.raises(KeyError, match="Unknown objective"):
            build_objective("nonexistent_obj_xyz")


# ===================================================================
# Config migration — new TrainingConfig fields
# ===================================================================

class TestConfigTrainingFields:
    def test_training_config_new_fields_default_empty(self):
        from crucible.core.config import TrainingConfig
        tc = TrainingConfig()
        assert tc.optimizer == ""
        assert tc.lr_schedule == ""
        assert tc.logging_backends == ""
        assert tc.callbacks == ""

    def test_training_config_loads_from_yaml(self, tmp_path):
        from crucible.core.config import load_config
        yaml_content = """\
name: test
training:
  - backend: torch
    optimizer: lion
    lr_schedule: cosine
    logging_backends: wandb,console
    callbacks: grad_clip,nan_detector
"""
        (tmp_path / "crucible.yaml").write_text(yaml_content)
        config = load_config(tmp_path / "crucible.yaml")
        assert len(config.training) == 1
        tc = config.training[0]
        assert tc.optimizer == "lion"
        assert tc.lr_schedule == "cosine"
        assert tc.logging_backends == "wandb,console"
        assert tc.callbacks == "grad_clip,nan_detector"

    def test_training_config_missing_fields_default(self, tmp_path):
        from crucible.core.config import load_config
        yaml_content = """\
name: test
training:
  - backend: torch
"""
        (tmp_path / "crucible.yaml").write_text(yaml_content)
        config = load_config(tmp_path / "crucible.yaml")
        tc = config.training[0]
        assert tc.optimizer == ""
        assert tc.lr_schedule == ""


class TestConfigPluginsSection:
    def test_plugins_config_defaults(self):
        from crucible.core.config import PluginsConfig
        pc = PluginsConfig()
        assert pc.discover is True
        assert pc.local_dir == "plugins"
        assert pc.hub_discover is True

    def test_plugins_config_from_yaml(self, tmp_path):
        from crucible.core.config import load_config
        yaml_content = """\
name: test
plugins:
  discover: false
  local_dir: my_plugins
  hub_discover: false
"""
        (tmp_path / "crucible.yaml").write_text(yaml_content)
        config = load_config(tmp_path / "crucible.yaml")
        assert config.plugins.discover is False
        assert config.plugins.local_dir == "my_plugins"
        assert config.plugins.hub_discover is False

    def test_plugins_config_missing_section_defaults(self, tmp_path):
        from crucible.core.config import load_config
        (tmp_path / "crucible.yaml").write_text("name: test\n")
        config = load_config(tmp_path / "crucible.yaml")
        assert config.plugins.discover is True
        assert config.plugins.local_dir == "plugins"
