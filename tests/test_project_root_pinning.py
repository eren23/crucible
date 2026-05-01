"""``CRUCIBLE_PROJECT_ROOT`` overrides ``find_config()``.

Multi-project users hit silent failures when ``find_config()`` walks up
from cwd and lands on the wrong ``crucible.yaml`` (parent dir contains
both projects, MCP server invoked from elsewhere, symlinked worktree).
The env var override pins the project root explicitly.

Also covers WANDB_RUN_NAME auto-derivation so cross-project runs in a
shared W&B project remain distinguishable.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from crucible.core.config import find_config


def _write_yaml(path: Path, name: str) -> None:
    path.write_text(f"name: {name}\n", encoding="utf-8")


class TestFindConfigOverride:
    def test_env_override_takes_precedence_over_cwd(self, tmp_path: Path):
        proj_a = tmp_path / "alpha"
        proj_b = tmp_path / "beta"
        proj_a.mkdir()
        proj_b.mkdir()
        _write_yaml(proj_a / "crucible.yaml", "alpha")
        _write_yaml(proj_b / "crucible.yaml", "beta")

        # Cwd is alpha but env points at beta — env wins.
        with patch.dict(os.environ, {"CRUCIBLE_PROJECT_ROOT": str(proj_b)}, clear=False):
            old = Path.cwd()
            os.chdir(proj_a)
            try:
                resolved = find_config()
            finally:
                os.chdir(old)

        assert resolved is not None
        assert resolved == proj_b / "crucible.yaml"

    def test_invalid_override_raises_config_error(self, tmp_path: Path):
        """A misconfigured override must raise ConfigError, not silently
        fall through to the walk-up — that would let a stale env var
        silently downgrade fleet ops to a default-name project on the
        wrong RunPod account."""
        from crucible.core.errors import ConfigError

        proj_a = tmp_path / "alpha"
        proj_a.mkdir()
        _write_yaml(proj_a / "crucible.yaml", "alpha")

        bad_root = tmp_path / "does-not-exist"
        with patch.dict(os.environ, {"CRUCIBLE_PROJECT_ROOT": str(bad_root)}, clear=False):
            old = Path.cwd()
            os.chdir(proj_a)
            try:
                with pytest.raises(ConfigError, match="CRUCIBLE_PROJECT_ROOT"):
                    find_config()
            finally:
                os.chdir(old)

    def test_falls_back_to_cwd_walkup_when_env_unset(self, tmp_path: Path):
        proj = tmp_path / "alpha"
        proj.mkdir()
        nested = proj / "src" / "deep"
        nested.mkdir(parents=True)
        _write_yaml(proj / "crucible.yaml", "alpha")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CRUCIBLE_PROJECT_ROOT", None)
            old = Path.cwd()
            os.chdir(nested)
            try:
                resolved = find_config()
            finally:
                os.chdir(old)

        assert resolved == proj / "crucible.yaml"


class TestWandbRunNameAutoDerive:
    """``run_experiment`` should derive WANDB_RUN_NAME from the project so
    two projects sharing a W&B project don't collide on the default UUID.
    Tests target the real ``derive_wandb_run_name`` helper to avoid
    silently drifting from production code."""

    def test_explicit_wandb_run_name_is_respected(self):
        from crucible.runner.experiment import derive_wandb_run_name
        assert derive_wandb_run_name(
            explicit="my-explicit-name",
            project_name="alpha", variant="", exp_id="abc123",
        ) == "my-explicit-name"

    def test_project_plus_variant_when_both_set(self):
        from crucible.runner.experiment import derive_wandb_run_name
        assert derive_wandb_run_name(
            explicit=None, project_name="alpha", variant="lr_5e-4", exp_id="abc123",
        ) == "alpha-lr_5e-4"

    def test_project_plus_exp_id_when_no_variant(self):
        from crucible.runner.experiment import derive_wandb_run_name
        assert derive_wandb_run_name(
            explicit=None, project_name="alpha", variant="", exp_id="abc123",
        ) == "alpha-abc123"

    def test_legacy_exp_id_when_project_unset(self):
        from crucible.runner.experiment import derive_wandb_run_name
        assert derive_wandb_run_name(
            explicit=None, project_name="", variant="", exp_id="abc123",
        ) == "abc123"

    def test_unsafe_chars_normalized(self):
        """Slashes and spaces in name yield clean identifier, not URL-encoded."""
        from crucible.runner.experiment import derive_wandb_run_name
        out = derive_wandb_run_name(
            explicit=None, project_name="Foo Bar/v2", variant="lr 5e-4", exp_id="abc123",
        )
        assert "/" not in out and " " not in out
        assert out == "Foo-Bar-v2-lr-5e-4"
