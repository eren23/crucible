"""Tests for crucible.runner.presets."""
from __future__ import annotations

import pytest

from crucible.core.config import ProjectConfig
from crucible.runner.presets import (
    PRESET_DEFAULTS,
    CLI_TIMEOUT_DEFAULTS,
    get_preset,
    list_presets,
    get_timeout_hint,
)


# ---------------------------------------------------------------------------
# get_preset
# ---------------------------------------------------------------------------

class TestGetPreset:
    def test_builtin_smoke(self):
        cfg = ProjectConfig()  # empty presets
        preset = get_preset("smoke", project_config=cfg)
        assert preset["MAX_WALLCLOCK_SECONDS"] == "60"
        assert preset["ITERATIONS"] == "400"

    def test_builtin_proxy(self):
        cfg = ProjectConfig()
        preset = get_preset("proxy", project_config=cfg)
        assert preset["MAX_WALLCLOCK_SECONDS"] == "1800"
        assert preset["ITERATIONS"] == "6000"

    def test_builtin_medium(self):
        cfg = ProjectConfig()
        preset = get_preset("medium", project_config=cfg)
        assert preset["MAX_WALLCLOCK_SECONDS"] == "3600"

    def test_builtin_promotion(self):
        cfg = ProjectConfig()
        preset = get_preset("promotion", project_config=cfg)
        assert preset["ITERATIONS"] == "100000"

    def test_builtin_overnight(self):
        cfg = ProjectConfig()
        preset = get_preset("overnight", project_config=cfg)
        assert preset["ITERATIONS"] == "200000"

    def test_unknown_preset_raises(self):
        cfg = ProjectConfig()
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent", project_config=cfg)

    def test_yaml_override_merges(self):
        cfg = ProjectConfig(
            presets={
                "smoke": {"MAX_WALLCLOCK_SECONDS": "120", "CUSTOM_KEY": "custom_val"},
            }
        )
        preset = get_preset("smoke", project_config=cfg)
        # Overridden
        assert preset["MAX_WALLCLOCK_SECONDS"] == "120"
        # Added
        assert preset["CUSTOM_KEY"] == "custom_val"
        # Inherited from builtin
        assert preset["ITERATIONS"] == "400"

    def test_yaml_only_preset(self):
        cfg = ProjectConfig(
            presets={
                "custom": {"MAX_WALLCLOCK_SECONDS": "999", "ITERATIONS": "10"},
            }
        )
        preset = get_preset("custom", project_config=cfg)
        assert preset["MAX_WALLCLOCK_SECONDS"] == "999"
        assert preset["ITERATIONS"] == "10"

    def test_all_builtin_presets_have_required_keys(self):
        required_keys = {"MAX_WALLCLOCK_SECONDS", "ITERATIONS", "TRAIN_BATCH_TOKENS"}
        for name, preset in PRESET_DEFAULTS.items():
            for key in required_keys:
                assert key in preset, f"Preset {name} missing {key}"


# ---------------------------------------------------------------------------
# list_presets
# ---------------------------------------------------------------------------

class TestListPresets:
    def test_returns_sorted_list(self):
        cfg = ProjectConfig()
        names = list_presets(project_config=cfg)
        assert names == sorted(names)

    def test_includes_builtins(self):
        cfg = ProjectConfig()
        names = list_presets(project_config=cfg)
        for builtin in PRESET_DEFAULTS:
            assert builtin in names

    def test_includes_yaml_presets(self):
        cfg = ProjectConfig(presets={"my_custom": {"A": "1"}})
        names = list_presets(project_config=cfg)
        assert "my_custom" in names

    def test_no_duplicates(self):
        cfg = ProjectConfig(presets={"smoke": {"A": "1"}})
        names = list_presets(project_config=cfg)
        assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# get_timeout_hint
# ---------------------------------------------------------------------------

class TestGetTimeoutHint:
    def test_known_backend_preset(self):
        timeout = get_timeout_hint("mlx", "smoke")
        assert timeout == CLI_TIMEOUT_DEFAULTS["mlx"]["smoke"]

    def test_torch_proxy(self):
        timeout = get_timeout_hint("torch", "proxy")
        assert timeout == CLI_TIMEOUT_DEFAULTS["torch"]["proxy"]

    def test_unknown_backend_returns_fallback(self):
        timeout = get_timeout_hint("unknown_backend", "smoke")
        assert timeout == 3600

    def test_unknown_preset_returns_fallback(self):
        timeout = get_timeout_hint("torch", "unknown_preset")
        assert timeout == 3600
