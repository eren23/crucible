"""Tests for MCP external project tools."""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from crucible.mcp.tools import (
    _make_launch_id,
    bootstrap_project_tool,
    collect_project_results,
    get_project_run_status,
    list_projects,
    provision_project,
    run_project,
)


def _write_spec(tmp_path, name, content):
    d = tmp_path / ".crucible" / "projects"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(content), encoding="utf-8")


class TestListProjects:
    def test_lists_specs(self, tmp_path, monkeypatch):
        _write_spec(tmp_path, "proj1", {"name": "proj1", "repo": "r1", "launcher": "demo_launcher", "train": "t1"})
        _write_spec(tmp_path, "proj2", {"name": "proj2", "repo": "r2", "train": "t2"})

        # Mock _get_config to return our tmp_path
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            result = list_projects({})

        assert len(result["projects"]) == 2
        names = {p["name"] for p in result["projects"]}
        assert names == {"proj1", "proj2"}
        proj1 = next(p for p in result["projects"] if p["name"] == "proj1")
        assert proj1["launcher"] == "demo_launcher"

    def test_empty_dir(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            result = list_projects({})
        assert result["projects"] == []


class TestCollectProjectResults:
    def test_missing_run_id(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        # Create the projects dir so load doesn't fail
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            result = collect_project_results({"run_id": "nonexistent_123"})
        assert "error" in result
        assert "No run found" in result["error"]

    def test_collect_by_launch_id(self, tmp_path):
        from crucible.core.config import ProjectConfig

        cfg = ProjectConfig(project_root=tmp_path)
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _save_project_run

            _save_project_run("run_1", {"launch_id": "launch_1", "project": "demo", "node_name": "n1", "pid": 1, "status": "launched"})
            _save_project_run("run_2", {"launch_id": "launch_1", "project": "demo", "node_name": "n2", "pid": 2, "status": "launched"})

            with patch("crucible.mcp.tools._observe_project_run", side_effect=[
                {"run_id": "run_1", "status": "completed"},
                {"run_id": "run_2", "status": "interrupted"},
            ]):
                result = collect_project_results({"launch_id": "launch_1"})

        assert result["summary"] == {"completed": 1, "interrupted": 1}
        assert {row["run_id"] for row in result["runs"]} == {"run_1", "run_2"}


class TestProjectRunStatus:
    def test_status_includes_recent_events(self, tmp_path):
        from crucible.core.config import ProjectConfig

        cfg = ProjectConfig(project_root=tmp_path)
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _save_project_run, _append_project_run_event

            _save_project_run("run_123", {"project": "demo", "node_name": "node-1", "pid": 42, "status": "launched"})
            _append_project_run_event("run_123", "launch_requested")
            _append_project_run_event("run_123", "launch_succeeded")

            with patch("crucible.mcp.tools._observe_project_run", return_value={
                "run_id": "run_123",
                "status": "running",
                "launch_id": None,
                "metrics": None,
                "log_tail": "",
                "log_path": "",
                "wandb": None,
                "contract_status": "compliant",
                "failure_class": None,
                "remote_node_state": "ready",
            }):
                result = get_project_run_status({"run_id": "run_123", "event_limit": 2})

        assert result["status"] == "running"
        assert [event["event"] for event in result["events"]] == ["launch_requested", "launch_succeeded"]


class TestProvisionProject:
    def test_uses_next_index_and_interruptible_override(self, tmp_path):
        from crucible.core.config import ProjectConfig

        cfg = ProjectConfig(project_root=tmp_path)
        _write_spec(
            tmp_path,
            "lewm",
            {
                "name": "lewm",
                "repo": "https://example.com/lewm.git",
                "pod": {
                    "gpu_type": "NVIDIA GeForce RTX 4090",
                    "container_disk": 40,
                    "volume_disk": 80,
                    "interruptible": False,
                },
            },
        )
        (tmp_path / cfg.nodes_file).write_text(
            '[{"name":"lewm-01","node_id":"n1"},{"name":"lewm-04","node_id":"n4"}]',
            encoding="utf-8",
        )
        merged_nodes = [
            {"name": "lewm-01", "node_id": "n1"},
            {"name": "lewm-04", "node_id": "n4"},
            {"name": "lewm-05", "node_id": "n5"},
        ]

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            with patch("crucible.mcp.tools._project_contract_env", return_value={}):
                with patch("crucible.fleet.manager.FleetManager.provision", return_value=merged_nodes) as mock_provision:
                    result = provision_project({"project_name": "lewm", "count": 1})

        assert result["created"] == 1
        assert result["new_nodes"] == [{"name": "lewm-05", "node_id": "n5"}]
        mock_provision.assert_called_once_with(
            count=1,
            name_prefix="lewm",
            start_index=5,
            gpu_type_id="NVIDIA GeForce RTX 4090",
            container_disk_gb=40,
            volume_gb=80,
            interruptible=False,
        )


class TestBootstrapProjectTool:
    def test_includes_node_error_details(self, tmp_path):
        from crucible.core.config import ProjectConfig

        cfg = ProjectConfig(project_root=tmp_path)
        _write_spec(
            tmp_path,
            "demo",
            {
                "name": "demo",
                "repo": "https://example.com/demo.git",
            },
        )
        (tmp_path / cfg.nodes_file).write_text(
            '[{"name":"demo-01","ssh_host":"1.2.3.4","project":"demo"}]',
            encoding="utf-8",
        )

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            with patch(
                "crucible.fleet.bootstrap.bootstrap_project",
                side_effect=RuntimeError("bootstrap:system_tools failed: unsupported_package_manager"),
            ):
                result = bootstrap_project_tool({"project_name": "demo"})

        assert result["bootstrapped"] == 0
        assert result["nodes"] == [
            {
                "name": "demo-01",
                "state": "boot_failed",
                "project": "demo",
                "error": "bootstrap:system_tools failed: unsupported_package_manager",
            }
        ]


class TestRunPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _save_project_run, _load_project_run

            _save_project_run("run_abc", {
                "node_name": "node-01",
                "ssh_host": "1.2.3.4",
                "project": "myproj",
                "pid": 1234,
            })

            loaded = _load_project_run("run_abc")
            assert loaded is not None
            assert loaded["run_id"] == "run_abc"
            assert loaded["node_name"] == "node-01"
            assert loaded["pid"] == 1234

    def test_load_returns_latest_record(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _save_project_run, _load_project_run

            _save_project_run("run_latest", {"status": "launched", "pid": 111})
            _save_project_run("run_latest", {"status": "completed", "pid": 222})

            loaded = _load_project_run("run_latest")
            assert loaded is not None
            assert loaded["status"] == "completed"
            assert loaded["pid"] == 222

    def test_load_missing_returns_none(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _load_project_run
            assert _load_project_run("nonexistent") is None


class TestLaunchIds:
    def test_make_launch_id_is_unique_across_calls(self):
        first = _make_launch_id("lewm")
        second = _make_launch_id("lewm")

        assert first != second
        assert first.startswith("lewm_")
        assert second.startswith("lewm_")


class TestRunProjectVariant:
    """Coverage for the ``variant=<name>`` arg on run_project.

    Before the 2026-04-11 fix, the ``variants:`` dict in a project yaml was
    silently dropped. Now the caller passes ``variant=<name>`` and the
    variant's env dict is merged into ``overrides`` before the caller's
    explicit ``overrides`` (so the caller's values still win when the same
    key appears in both).
    """

    def _setup_spec(self, tmp_path, **extra):
        content = {
            "name": "dummy",
            "repo": "https://example.com/r.git",
            "train": "python train.py",
            "variants": {
                "small": {
                    "WM_STEPS": "1000",
                    "WM_LR": "1e-3",
                    "WANDB_RUN_NAME": "small-run",
                },
                "large": {
                    "WM_STEPS": "15000",
                    "WM_LR": "5e-4",
                    "WANDB_RUN_NAME": "large-run",
                },
            },
        }
        content.update(extra)
        _write_spec(tmp_path, "dummy", content)

    def test_unknown_variant_is_loud_error(self, tmp_path):
        from crucible.core.config import ProjectConfig
        self._setup_spec(tmp_path)
        cfg = ProjectConfig(project_root=tmp_path)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            result = run_project({
                "project_name": "dummy",
                "variant": "nonexistent_variant",
            })

        assert "error" in result
        assert "Variant 'nonexistent_variant' not found" in result["error"]
        assert "Available variants: large, small" in result["error"]

    def _fake_nodes(self):
        return [{
            "name": "dummy-01",
            "project": "dummy",
            "env_ready": True,
            "ssh_host": "10.0.0.1",
            "ssh_port": 22,
            "workspace_path": "/workspace/project",
        }]

    def _run_and_capture(self, tmp_path, args, extra_env=None):
        """Helper: mock inventory + launch_project, call run_project, return the
        overrides dict that was passed to launch_project."""
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)

        captured: dict = {}

        def fake_launch(node, spec, run_id, *, overrides=None):
            captured["overrides"] = dict(overrides or {})
            return {"pid": 1234, "status": "launched"}

        # WANDB_PROJECT is required by the contract env check. Set a fake
        # value unless the caller specified extra_env.
        os_env_patch = {
            "WANDB_PROJECT": "fake-project-for-test",
            "WANDB_API_KEY": "fake-api-key-for-test",
            **(extra_env or {}),
        }

        # load_nodes_if_exists is imported *inside* run_project's body, so
        # patch it at the source module, not the MCP tools module.
        with patch("crucible.mcp.tools._get_config", return_value=cfg), \
             patch.dict("os.environ", os_env_patch, clear=False), \
             patch("crucible.fleet.inventory.load_nodes_if_exists", return_value=self._fake_nodes()), \
             patch("crucible.fleet.project_runner.launch_project", side_effect=fake_launch):
            result = run_project(args)
        return result, captured.get("overrides", {})

    def test_variant_dict_merged_into_overrides_before_caller(self, tmp_path):
        """Variant values apply, caller's explicit overrides win ties."""
        self._setup_spec(tmp_path)
        result, env = self._run_and_capture(tmp_path, {
            "project_name": "dummy",
            "variant": "large",
            # Caller overrides WM_LR — should win over variant's "5e-4".
            # Caller does NOT override WM_STEPS — should stay as variant's "15000".
            "overrides": {"WM_LR": "1e-4"},
        })
        assert "error" not in result, f"unexpected error: {result.get('error')}"
        # Variant values that the caller didn't override: from variant.
        assert env.get("WM_STEPS") == "15000"
        # Caller's override wins over the variant's value for the same key.
        assert env.get("WM_LR") == "1e-4"
        # Variant-supplied WANDB_RUN_NAME also flows through (caller didn't override).
        assert env.get("WANDB_RUN_NAME") == "large-run"
        # CRUCIBLE_VARIANT_NAME reflects the chosen variant name.
        assert env.get("CRUCIBLE_VARIANT_NAME") == "large"

    def test_no_variant_arg_preserves_legacy_behavior(self, tmp_path):
        """Passing no variant at all means the variants dict is not read:
        only the caller's overrides apply."""
        self._setup_spec(tmp_path)
        result, env = self._run_and_capture(tmp_path, {
            "project_name": "dummy",
            "overrides": {"WM_LR": "3e-4"},
        })
        assert "error" not in result, f"unexpected error: {result.get('error')}"
        assert env.get("WM_LR") == "3e-4"
        # No variant applied, so neither variant's WM_STEPS nor its WANDB_RUN_NAME leak in.
        assert "WM_STEPS" not in env
        assert env.get("WANDB_RUN_NAME") not in {"large-run", "small-run"}
