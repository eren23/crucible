"""Smoke test for the Phase 3 hf-collab recipe yaml.

Validates that docs/recipes/hf-collab-parameter-golf.yaml parses cleanly
and contains the expected schema fields. Catches accidental yaml
breakage on edits without requiring a real HF round-trip.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


RECIPE_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs" / "recipes" / "hf-collab-parameter-golf.yaml"
)


def _load() -> dict:
    return yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))


class TestHfCollabRecipe:
    def test_file_exists(self):
        assert RECIPE_PATH.exists(), f"recipe yaml missing at {RECIPE_PATH}"

    def test_parses_as_yaml(self):
        data = _load()
        assert isinstance(data, dict)

    def test_has_required_top_level_fields(self):
        data = _load()
        for key in ("name", "title", "goal", "environment", "steps", "gotchas", "tags"):
            assert key in data, f"missing {key!r}"
        assert data["name"] == "hf-collab-parameter-golf"

    def test_steps_reference_real_mcp_tools(self):
        data = _load()
        from crucible.mcp.tools import TOOL_DISPATCH

        for i, step in enumerate(data["steps"]):
            assert isinstance(step, dict), f"step {i} must be a dict"
            assert "tool" in step, f"step {i} missing 'tool' key"
            tool_name = step["tool"]
            assert tool_name in TOOL_DISPATCH, (
                f"step {i} references unknown tool {tool_name!r}; "
                f"fix the recipe or rename in TOOL_DISPATCH"
            )

    def test_uses_all_three_phases_of_hf_collab(self):
        # The recipe is the canonical end-to-end demo. It must exercise
        # at least one tool from each Phase so users see the full loop.
        data = _load()
        tools_used = {step["tool"] for step in data["steps"]}
        # Phase 1 — write side
        assert tools_used & {
            "hf_publish_leaderboard", "hf_push_artifact",
            "hf_publish_findings", "hf_publish_recipes",
        }, "recipe should include at least one Phase-1 write tool"
        # Phase 2 — read side
        assert tools_used & {
            "research_hf_prior_attempts", "research_hf_discussions",
            "note_post_to_hf_discussions",
        }, "recipe should include at least one Phase-2 read/comm tool"

    def test_gotchas_have_symptom_and_fix(self):
        data = _load()
        for i, g in enumerate(data["gotchas"]):
            assert "symptom" in g and "fix" in g, (
                f"gotcha {i} must have both 'symptom' and 'fix'"
            )

    def test_environment_block_lists_required_env_files(self):
        data = _load()
        env = data["environment"]
        files = env.get("required_env_files") or []
        all_keys = {k for f in files for k in f.get("keys", [])}
        # HF_TOKEN must be in the required list — the whole point of
        # this recipe is HF integration.
        assert "HF_TOKEN" in all_keys

    def test_tags_include_hf_collab_marker(self):
        data = _load()
        assert "hf-collab" in data["tags"]

    def test_required_yaml_fields_match_hfcollabconfig(self):
        # If HfCollabConfig drops a field that the recipe still tells
        # users to set, this test fails — preventing recipe drift from
        # the actual config schema.
        from dataclasses import fields

        from crucible.core.config import HfCollabConfig

        data = _load()
        env_blocks = data["environment"].get("required_yaml") or []
        hf_block = next(
            (b for b in env_blocks if b.get("block") == "hf_collab"),
            None,
        )
        assert hf_block is not None, "recipe must list an hf_collab yaml block"
        recipe_field_names = set((hf_block.get("fields") or {}).keys())
        config_field_names = {f.name for f in fields(HfCollabConfig)}
        unknown = recipe_field_names - config_field_names
        assert not unknown, (
            f"Recipe references hf_collab fields that no longer exist on "
            f"HfCollabConfig: {sorted(unknown)}. Either update the recipe "
            f"or restore the field."
        )

    def test_recipe_get_round_trip(self, tmp_path, monkeypatch):
        # Copy the recipe into a tmp project's .crucible/recipes/ and
        # confirm the actual recipe_get MCP tool can load it back. This
        # catches recipe-system invariants (e.g. step ordering, tag
        # auto-merge) that the raw yaml parse would miss.
        from crucible.core.config import ProjectConfig
        from crucible.mcp.tools import recipe_get

        project = tmp_path / "proj"
        recipes_dir = project / ".crucible" / "recipes"
        recipes_dir.mkdir(parents=True)
        target = recipes_dir / "hf-collab-parameter-golf.yaml"
        target.write_text(RECIPE_PATH.read_text(encoding="utf-8"))

        cfg = ProjectConfig()
        cfg.project_root = project
        cfg.store_dir = ".crucible"
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: cfg)

        loaded = recipe_get({"name": "hf-collab-parameter-golf"})
        # recipe_get returns either the recipe dict or {"error": ...}.
        assert "error" not in loaded, loaded.get("error")
        assert loaded["name"] == "hf-collab-parameter-golf"
        assert isinstance(loaded["steps"], list) and loaded["steps"]
