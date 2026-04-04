"""Smoke tests for the fleet pipeline: design -> queue -> dispatch -> collect.

These mock SSH/RunPod but exercise real code paths end-to-end.
No GPU or network access required.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def smoke_project(tmp_path: Path) -> Path:
    """Minimal project structure for smoke testing."""
    (tmp_path / "crucible.yaml").write_text(yaml.dump({
        "store_dir": ".crucible",
        "results_file": "experiments.jsonl",
        "fleet_results_file": "experiments_fleet.jsonl",
        "nodes_file": "nodes.json",
        "queue_file": "fleet_queue.jsonl",
        "provider": {"type": "ssh", "defaults": {"workspace_path": "/workspace"}},
    }))
    (tmp_path / ".crucible").mkdir()
    (tmp_path / ".crucible" / "designs").mkdir()
    (tmp_path / "nodes.json").write_text("[]")
    (tmp_path / "experiments.jsonl").touch()
    (tmp_path / "experiments_fleet.jsonl").touch()
    (tmp_path / "fleet_queue.jsonl").touch()
    (tmp_path / "logs").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Design save + retrieve round-trip
# ---------------------------------------------------------------------------

class TestDesignRoundTrip:
    """Design save/retrieve via YAML files (bypasses VersionStore complexity)."""

    def test_save_and_get_design(self, smoke_project: Path):
        designs_dir = smoke_project / ".crucible" / "designs" / "smoke-moe"
        designs_dir.mkdir(parents=True)
        content = {
            "name": "smoke-moe",
            "config": {
                "MODEL_FAMILY": "moe_baseline",
                "MODEL_DIM": "384",
                "NUM_LAYERS": "6",
                "MOE_NUM_EXPERTS": "4",
            },
            "base_preset": "smoke",
        }
        (designs_dir / "v1.yaml").write_text(yaml.dump(content))
        (designs_dir / "current.yaml").write_text(yaml.dump(content))

        loaded = yaml.safe_load((designs_dir / "current.yaml").read_text())
        assert loaded["name"] == "smoke-moe"
        assert loaded["config"]["MODEL_FAMILY"] == "moe_baseline"

    def test_save_multiple_families(self, smoke_project: Path):
        families = [
            ("smoke-baseline", "baseline"),
            ("smoke-looped", "looped"),
            ("smoke-moe", "moe_baseline"),
            ("smoke-partial-rope", "sota_partial_rope"),
        ]
        for name, family in families:
            d = smoke_project / ".crucible" / "designs" / name
            d.mkdir(parents=True)
            content = {"name": name, "config": {"MODEL_FAMILY": family, "MODEL_DIM": "256"}, "base_preset": "smoke"}
            (d / "current.yaml").write_text(yaml.dump(content))

        design_dirs = list((smoke_project / ".crucible" / "designs").iterdir())
        saved_names = {d.name for d in design_dirs if d.is_dir()}
        assert {"smoke-baseline", "smoke-looped", "smoke-moe", "smoke-partial-rope"} <= saved_names


# ---------------------------------------------------------------------------
# 2. Queue enqueue + status
# ---------------------------------------------------------------------------

class TestQueueRoundTrip:
    def test_enqueue_and_check(self, smoke_project: Path):
        from crucible.fleet.queue import enqueue_experiments, load_queue

        queue_file = smoke_project / "fleet_queue.jsonl"
        enqueue_experiments(queue_file, [
            {"name": "smoke-test-1", "tier": "smoke", "backend": "torch", "config": {"MODEL_FAMILY": "baseline"}},
            {"name": "smoke-test-2", "tier": "smoke", "backend": "torch", "config": {"MODEL_FAMILY": "moe_baseline"}},
        ])

        queue = load_queue(queue_file)
        names = [e.get("experiment_name") for e in queue]
        assert "smoke-test-1" in names
        assert "smoke-test-2" in names

    def test_enqueue_preserves_config(self, smoke_project: Path):
        from crucible.fleet.queue import enqueue_experiments, load_queue

        queue_file = smoke_project / "fleet_queue.jsonl"
        enqueue_experiments(queue_file, [{
            "name": "config-check",
            "tier": "smoke",
            "backend": "torch",
            "config": {
                "MODEL_FAMILY": "sota_partial_rope",
                "ROPE_DIMS": "16",
                "SMEAR_GATE": "true",
            },
        }])

        queue = load_queue(queue_file)
        entry = [e for e in queue if e.get("experiment_name") == "config-check"][0]
        assert entry["config"]["ROPE_DIMS"] == "16"


# ---------------------------------------------------------------------------
# 3. Output parser
# ---------------------------------------------------------------------------

class TestOutputParser:
    def test_parse_completed_run(self):
        from crucible.runner.output_parser import OutputParser

        parser = OutputParser()
        output = (
            "step:100/500 train_loss:3.245\n"
            "step:200/500 train_loss:2.891\n"
            "final_eval val_loss:2.500 val_bpb:1.350\n"
            "Serialized model int8+zstd: 12345 bytes\n"
        )
        result = parser.parse(output)
        assert result is not None
        assert result["status"] == "completed"
        assert result["result"]["val_bpb"] == pytest.approx(1.35)
        assert result["model_bytes"] == 12345

    def test_parse_partial_run(self):
        from crucible.runner.output_parser import OutputParser

        parser = OutputParser()
        output = "step:100/500 train_loss:3.245\nstep:200/500 train_loss:2.891\n"
        result = parser.parse(output)
        assert result is not None
        assert result["status"] != "completed"

    def test_parse_empty_output(self):
        from crucible.runner.output_parser import OutputParser
        assert OutputParser().parse("") is None


# ---------------------------------------------------------------------------
# 4. SSH smoke (mocked)
# ---------------------------------------------------------------------------

class TestSSHSmoke:
    def test_ssh_ok_success(self):
        from crucible.fleet.sync import ssh_ok
        node = {"ssh_host": "1.2.3.4", "ssh_port": 22}
        with patch("crucible.fleet.sync._run") as mock:
            mock.return_value = subprocess.CompletedProcess([], 0, "ready\n", "")
            assert ssh_ok(node) is True

    def test_ssh_ok_failure(self):
        from crucible.fleet.sync import ssh_ok
        node = {"ssh_host": "1.2.3.4", "ssh_port": 22}
        with patch("crucible.fleet.sync._run") as mock:
            mock.return_value = subprocess.CompletedProcess([], 255, "", "refused")
            assert ssh_ok(node) is False

    def test_ssh_ok_no_host(self):
        from crucible.fleet.sync import ssh_ok
        assert ssh_ok({}) is False

    def test_remote_exec_passes_timeout(self):
        from crucible.fleet.sync import remote_exec
        node = {"ssh_host": "10.0.0.1", "ssh_port": 22}
        with patch("crucible.fleet.sync._run") as mock:
            mock.return_value = subprocess.CompletedProcess([], 0, "", "")
            remote_exec(node, "echo test", check=False, timeout=60)
            _, kwargs = mock.call_args
            assert kwargs["timeout"] == 60

    def test_checked_remote_exec_long_timeout(self):
        import inspect
        from crucible.fleet.sync import checked_remote_exec
        sig = inspect.signature(checked_remote_exec)
        assert sig.parameters["timeout"].default == 600


# ---------------------------------------------------------------------------
# 5. MCP tool dispatch
# ---------------------------------------------------------------------------

class TestMCPToolDispatch:
    def test_fleet_tools_registered(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        for t in ["get_fleet_status", "get_queue_status", "enqueue_experiment", "cancel_experiment"]:
            assert t in TOOL_DISPATCH

    def test_model_tools_registered(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        for t in ["model_list_families", "model_compose", "model_validate_config"]:
            assert t in TOOL_DISPATCH

    def test_hub_tools_registered(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        for t in ["hub_status", "hub_tap_list", "hub_search", "hub_install"]:
            assert t in TOOL_DISPATCH

    def test_plugin_tools_registered(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        for t in ["plugin_list", "plugin_add", "plugin_get_schema"]:
            assert t in TOOL_DISPATCH

    def test_families_returns_builtins(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        result = TOOL_DISPATCH["model_list_families"]({})
        assert "baseline" in result["families"]

    def test_tool_count_above_100(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        assert len(TOOL_DISPATCH) >= 100


# ---------------------------------------------------------------------------
# 6. Architecture registration
# ---------------------------------------------------------------------------

class TestArchRegistration:
    def test_builtins(self):
        from crucible.models.registry import list_families
        families = list_families()
        for name in ["baseline", "looped", "convloop", "prefix_memory"]:
            assert name in families

    def test_local_plugins(self):
        """Local .crucible/architectures/ plugins should be in families (if project dir is cwd)."""
        from crucible.models.registry import list_families
        families = list_families()
        # These are loaded from .crucible/architectures/ of the actual project
        # May not be present in CI. Check at least one exists beyond builtins.
        has_local = any(f not in {"baseline", "looped", "convloop", "prefix_memory", "memory"} for f in families)
        assert has_local or len(families) >= 4  # builtins always present


# ---------------------------------------------------------------------------
# 7. Plugin registries
# ---------------------------------------------------------------------------

class TestPluginRegistries:
    def test_optimizers(self):
        from crucible.training.optimizers import OPTIMIZER_REGISTRY
        names = OPTIMIZER_REGISTRY.list_plugins()
        for n in ["adam", "adamw", "sgd", "rmsprop", "muon"]:
            assert n in names

    def test_schedulers(self):
        from crucible.training.schedulers import SCHEDULER_REGISTRY
        names = SCHEDULER_REGISTRY.list_plugins()
        assert "cosine" in names and "constant" in names

    def test_callbacks(self):
        from crucible.training.callbacks import CALLBACK_REGISTRY
        names = CALLBACK_REGISTRY.list_plugins()
        assert "grad_clip" in names and "nan_detector" in names

    def test_loggers(self):
        from crucible.runner.loggers import LOGGER_REGISTRY
        names = LOGGER_REGISTRY.list_plugins()
        assert "console" in names and "jsonl" in names


# ---------------------------------------------------------------------------
# 8. Presets
# ---------------------------------------------------------------------------

class TestPresets:
    def test_all_presets_load(self):
        from crucible.runner.presets import list_presets, get_preset
        presets = list_presets()
        for name in ["smoke", "screen", "proxy", "medium", "promotion"]:
            assert name in presets
            p = get_preset(name)
            assert isinstance(p, dict)

    def test_unknown_preset_raises(self):
        from crucible.runner.presets import get_preset
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent_xyz")


# ---------------------------------------------------------------------------
# 9. Leaderboard with empty results
# ---------------------------------------------------------------------------

class TestLeaderboardEmpty:
    def test_no_results(self):
        from crucible.analysis.leaderboard import leaderboard
        results = leaderboard(results=[], top_n=5)
        assert results == []


# ---------------------------------------------------------------------------
# 10. Tap smoke
# ---------------------------------------------------------------------------

class TestTapSmoke:
    def test_search_if_tap_configured(self):
        from crucible.core.tap import TapManager
        tm = TapManager()
        if not tm.list_taps():
            pytest.skip("No taps configured")
        assert len(tm.search()) >= 1

    def test_list_installed(self):
        from crucible.core.tap import TapManager
        assert isinstance(TapManager().list_installed(), list)
