"""Tests verifying the judge-separation contract is enforced at MCP tool boundaries.

When ``ProjectConfig.judges`` is configured, harness_iterate and
tree_expand_grpo must reject mis-separated panels before any LLM call
happens. Unconfigured panels skip enforcement (opt-in).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.config import JudgeConfig, JudgePanel, ProjectConfig


def _make_proj(tmp_path: Path, panel: JudgePanel | None = None) -> ProjectConfig:
    proj_root = tmp_path / "proj"
    proj_root.mkdir(exist_ok=True)
    (proj_root / ".crucible").mkdir(exist_ok=True)
    return ProjectConfig(
        name="test-proj",
        project_root=proj_root,
        judges=panel or JudgePanel(),
    )


class TestHarnessIterateGate:
    def test_rejects_mis_separated_panel(self, tmp_path: Path, monkeypatch):
        from crucible.mcp.tools import harness_iterate

        bad_panel = JudgePanel(
            reward_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        config = _make_proj(tmp_path, bad_panel)
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)

        # No optimizer registered for this tree name → without the gate, we'd
        # get a "[StateError] call harness_init first". With the gate, we
        # should see the judge-separation error first.
        out = harness_iterate({"tree_name": "anything"})
        assert "error" in out
        assert "judge" in out["error"].lower()

    def test_skips_when_panel_unconfigured(self, tmp_path: Path, monkeypatch):
        from crucible.mcp.tools import harness_iterate

        config = _make_proj(tmp_path, JudgePanel())  # empty
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)

        out = harness_iterate({"tree_name": "no-such-tree"})
        # Falls through to normal "no optimizer" error, NOT a judge error.
        assert "error" in out
        assert "judge" not in out["error"].lower()


class TestHarnessOptimizerDirectGate:
    """Direct-call gate (no MCP) — closes the bypass for TUI/CLI paths."""

    def test_optimizer_run_iteration_raises_on_mis_separated_panel(
        self, tmp_path: Path,
    ):
        import pytest
        import yaml
        from crucible.core.errors import ConfigError
        from crucible.researcher.harness_optimizer import HarnessOptimizer

        bad_panel = JudgePanel(
            reward_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        config = _make_proj(tmp_path, bad_panel)

        # Minimal in-memory domain spec.
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump({
            "name": "test_domain",
            "interface": {"class_name": "Harness", "required_methods": []},
            "metrics": [{"name": "accuracy", "direction": "maximize"}],
            "baselines": [],
            "evaluation": {"backend": "harness", "tier": "harness"},
        }), encoding="utf-8")

        opt = HarnessOptimizer(
            config,
            domain_spec=spec_path,
            tree_name="t1",
            n_candidates=1,
            dry_run=True,
        )

        with pytest.raises(ConfigError, match="reward_judge"):
            opt.run_iteration()

    def test_optimizer_run_iteration_passes_when_unconfigured(
        self, tmp_path: Path,
    ):
        import yaml
        from crucible.researcher.harness_optimizer import HarnessOptimizer

        config = _make_proj(tmp_path, JudgePanel())  # empty
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump({
            "name": "test_domain",
            "interface": {"class_name": "Harness", "required_methods": []},
            "metrics": [{"name": "accuracy", "direction": "maximize"}],
            "baselines": [],
            "evaluation": {"backend": "harness", "tier": "harness"},
        }), encoding="utf-8")

        opt = HarnessOptimizer(
            config,
            domain_spec=spec_path,
            tree_name="t2",
            n_candidates=1,
            dry_run=True,
        )
        # Empty panel → gate skipped, dry-run completes without raising.
        summary = opt.run_iteration()
        assert "iteration" in summary
