"""Tests for recipe MCP tools (save / list / get).

Covers: validation, edge cases, error handling, roundtrip persistence,
tag filtering, and file I/O robustness.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from crucible.mcp.tools import recipe_get, recipe_list, recipe_save


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_config(tmp_path: Path):
    from crucible.core.config import ProjectConfig

    return ProjectConfig(
        name="test",
        project_root=tmp_path,
        store_dir=".crucible",
    )


def _make_step(tool: str = "run_project", note: str = "run it") -> dict:
    return {"tool": tool, "args": {"project_name": "demo"}, "note": note}


def _make_recipe_args(
    name: str = "test-recipe",
    steps: list | None = None,
    **overrides,
) -> dict:
    base = {
        "name": name,
        "title": "Test Recipe",
        "goal": "Verify recipe tools",
        "project_spec": "yolo11-demo",
        "environment": {"gpu": "RTX 4090", "torch": "2.6.0+cu124", "python": "3.11"},
        "steps": steps or [_make_step()],
        "results": {"map50_95_b": 0.733},
        "gotchas": [{"issue": "torch version", "fix": "pin cu124"}],
        "tags": ["yolo", "test"],
        "created_by": "test-agent",
    }
    base.update(overrides)
    return base


@pytest.fixture
def cfg(tmp_path):
    """Provide a patched _get_config that returns a tmp_path-based config."""
    config = _fake_config(tmp_path)
    with patch("crucible.mcp.tools._get_config", return_value=config):
        yield tmp_path


# ---------------------------------------------------------------------------
# recipe_save — validation
# ---------------------------------------------------------------------------


class TestRecipeSaveValidation:
    def test_rejects_missing_name(self, cfg):
        result = recipe_save({"steps": [_make_step()]})
        assert "error" in result
        assert "name" in result["error"].lower()

    def test_rejects_empty_name(self, cfg):
        result = recipe_save({"name": "", "steps": [_make_step()]})
        assert "error" in result

    def test_rejects_name_with_spaces(self, cfg):
        result = recipe_save({"name": "has spaces", "steps": [_make_step()]})
        assert "error" in result
        assert "Invalid recipe name" in result["error"]

    def test_rejects_name_with_uppercase(self, cfg):
        result = recipe_save({"name": "HasUpper", "steps": [_make_step()]})
        assert "error" in result

    def test_rejects_name_starting_with_hyphen(self, cfg):
        result = recipe_save({"name": "-bad-start", "steps": [_make_step()]})
        assert "error" in result

    def test_rejects_name_with_special_chars(self, cfg):
        result = recipe_save({"name": "bad/name!", "steps": [_make_step()]})
        assert "error" in result

    def test_rejects_missing_steps(self, cfg):
        result = recipe_save({"name": "no-steps"})
        assert "error" in result
        assert "step" in result["error"].lower()

    def test_rejects_empty_steps(self, cfg):
        result = recipe_save({"name": "empty-steps", "steps": []})
        assert "error" in result

    def test_rejects_malformed_step_no_tool(self, cfg):
        result = recipe_save({
            "name": "bad-step",
            "steps": [{"args": {}, "note": "missing tool key"}],
        })
        assert "error" in result
        assert "tool" in result["error"]

    def test_rejects_step_thats_not_a_dict(self, cfg):
        result = recipe_save({
            "name": "bad-step-type",
            "steps": ["not a dict"],
        })
        assert "error" in result

    def test_accepts_valid_names(self, cfg):
        valid_names = ["my-recipe", "recipe123", "a", "yolo11-nano-coco128", "with_underscores"]
        for name in valid_names:
            result = recipe_save({"name": name, "steps": [_make_step()]})
            assert result.get("saved") is True, f"Name {name!r} should be valid"


# ---------------------------------------------------------------------------
# recipe_save — persistence
# ---------------------------------------------------------------------------


class TestRecipeSavePersistence:
    def test_creates_yaml_file(self, cfg):
        result = recipe_save(_make_recipe_args())
        assert result["saved"] is True

        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        assert path.exists()

        data = yaml.safe_load(path.read_text())
        assert data["name"] == "test-recipe"
        assert data["title"] == "Test Recipe"
        assert data["goal"] == "Verify recipe tools"
        assert data["project_spec"] == "yolo11-demo"

    def test_persists_environment(self, cfg):
        recipe_save(_make_recipe_args())
        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert data["environment"]["gpu"] == "RTX 4090"
        assert data["environment"]["torch"] == "2.6.0+cu124"

    def test_persists_steps(self, cfg):
        steps = [
            {"tool": "provision_project", "args": {"count": 1}, "note": "spin up"},
            {"tool": "bootstrap_project", "args": {"project_name": "demo"}, "note": "install"},
            {"tool": "run_project", "args": {"project_name": "demo"}, "note": "train"},
        ]
        recipe_save(_make_recipe_args(steps=steps))
        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert len(data["steps"]) == 3
        assert data["steps"][0]["tool"] == "provision_project"
        assert data["steps"][2]["tool"] == "run_project"

    def test_persists_gotchas(self, cfg):
        recipe_save(_make_recipe_args())
        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert len(data["gotchas"]) == 1
        assert data["gotchas"][0]["issue"] == "torch version"
        assert data["gotchas"][0]["fix"] == "pin cu124"

    def test_persists_results(self, cfg):
        recipe_save(_make_recipe_args())
        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert data["results"]["map50_95_b"] == 0.733

    def test_persists_tags(self, cfg):
        recipe_save(_make_recipe_args())
        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert data["tags"] == ["yolo", "test"]

    def test_sets_created_at(self, cfg):
        recipe_save(_make_recipe_args())
        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert "created_at" in data
        assert len(data["created_at"]) > 0

    def test_sets_created_by(self, cfg):
        recipe_save(_make_recipe_args(created_by="my-agent"))
        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert data["created_by"] == "my-agent"

    def test_defaults_created_by_to_mcp_agent(self, cfg):
        args = _make_recipe_args()
        del args["created_by"]
        recipe_save(args)
        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert data["created_by"] == "mcp-agent"

    def test_overwrites_existing_recipe(self, cfg):
        recipe_save(_make_recipe_args(title="Version 1"))
        result = recipe_save(_make_recipe_args(title="Version 2"))
        assert result["saved"] is True
        assert result["overwritten"] is True

        path = cfg / ".crucible" / "recipes" / "test-recipe.yaml"
        data = yaml.safe_load(path.read_text())
        assert data["title"] == "Version 2"

    def test_first_save_not_overwritten(self, cfg):
        result = recipe_save(_make_recipe_args())
        assert result["overwritten"] is False

    def test_minimal_recipe(self, cfg):
        result = recipe_save({"name": "minimal", "steps": [_make_step()]})
        assert result["saved"] is True

        path = cfg / ".crucible" / "recipes" / "minimal.yaml"
        data = yaml.safe_load(path.read_text())
        assert data["name"] == "minimal"
        assert data["title"] == ""
        assert data["goal"] == ""
        assert data["environment"] == {}
        assert data["gotchas"] == []
        assert data["tags"] == []

    def test_creates_recipes_dir(self, cfg):
        recipes_dir = cfg / ".crucible" / "recipes"
        assert not recipes_dir.exists()
        recipe_save(_make_recipe_args())
        assert recipes_dir.exists()

    def test_unicode_content(self, cfg):
        recipe_save(_make_recipe_args(
            name="unicode-test",
            title="Transformer training",
            goal="Fine-tune with mixed precision",
        ))
        path = cfg / ".crucible" / "recipes" / "unicode-test.yaml"
        data = yaml.safe_load(path.read_text())
        assert "Transformer" in data["title"]


# ---------------------------------------------------------------------------
# recipe_list
# ---------------------------------------------------------------------------


class TestRecipeList:
    def test_empty_when_no_dir(self, cfg):
        result = recipe_list({})
        assert result["recipes"] == []
        assert result["total"] == 0

    def test_empty_when_dir_exists_but_no_recipes(self, cfg):
        (cfg / ".crucible" / "recipes").mkdir(parents=True)
        result = recipe_list({})
        assert result["recipes"] == []
        assert result["total"] == 0

    def test_lists_saved_recipes(self, cfg):
        recipe_save(_make_recipe_args(name="alpha", title="Alpha"))
        recipe_save(_make_recipe_args(name="beta", title="Beta"))

        result = recipe_list({})
        assert result["total"] == 2
        names = [r["name"] for r in result["recipes"]]
        assert "alpha" in names
        assert "beta" in names

    def test_returns_summary_fields(self, cfg):
        recipe_save(_make_recipe_args(
            name="detailed",
            title="Detailed Recipe",
            goal="Test listing",
            tags=["gpu", "wandb"],
        ))

        result = recipe_list({})
        r = result["recipes"][0]
        assert r["name"] == "detailed"
        assert r["title"] == "Detailed Recipe"
        assert r["goal"] == "Test listing"
        assert r["tags"] == ["gpu", "wandb"]
        assert r["project_spec"] == "yolo11-demo"
        assert "created_at" in r

    def test_filter_by_tag(self, cfg):
        recipe_save(_make_recipe_args(name="yolo-run", tags=["yolo", "detection"]))
        recipe_save(_make_recipe_args(name="diffusion-run", tags=["diffusion", "ddpm"]))
        recipe_save(_make_recipe_args(name="yolo-small", tags=["yolo", "small"]))

        result = recipe_list({"tag": "yolo"})
        assert result["total"] == 2
        names = [r["name"] for r in result["recipes"]]
        assert "yolo-run" in names
        assert "yolo-small" in names
        assert "diffusion-run" not in names

    def test_filter_by_tag_no_match(self, cfg):
        recipe_save(_make_recipe_args(name="alpha", tags=["a"]))
        result = recipe_list({"tag": "nonexistent"})
        assert result["total"] == 0

    def test_skips_corrupted_yaml(self, cfg):
        recipe_save(_make_recipe_args(name="good-recipe"))

        recipes_dir = cfg / ".crucible" / "recipes"
        (recipes_dir / "bad.yaml").write_text(": : : invalid yaml [[[")

        result = recipe_list({})
        assert result["total"] == 1
        assert result["recipes"][0]["name"] == "good-recipe"

    def test_skips_non_dict_yaml(self, cfg):
        recipe_save(_make_recipe_args(name="good-recipe"))

        recipes_dir = cfg / ".crucible" / "recipes"
        (recipes_dir / "scalar.yaml").write_text("just a string")

        result = recipe_list({})
        assert result["total"] == 1

    def test_sorted_alphabetically(self, cfg):
        recipe_save(_make_recipe_args(name="charlie"))
        recipe_save(_make_recipe_args(name="alpha"))
        recipe_save(_make_recipe_args(name="bravo"))

        result = recipe_list({})
        names = [r["name"] for r in result["recipes"]]
        assert names == ["alpha", "bravo", "charlie"]


# ---------------------------------------------------------------------------
# recipe_get
# ---------------------------------------------------------------------------


class TestRecipeGet:
    def test_returns_full_data(self, cfg):
        recipe_save(_make_recipe_args(name="full-recipe"))
        result = recipe_get({"name": "full-recipe"})

        assert result["name"] == "full-recipe"
        assert result["title"] == "Test Recipe"
        assert result["goal"] == "Verify recipe tools"
        assert result["project_spec"] == "yolo11-demo"
        assert result["environment"]["gpu"] == "RTX 4090"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["tool"] == "run_project"
        assert result["results"]["map50_95_b"] == 0.733
        assert result["gotchas"][0]["issue"] == "torch version"
        assert result["tags"] == ["yolo", "test"]
        assert result["created_by"] == "test-agent"
        assert "created_at" in result

    def test_not_found(self, cfg):
        result = recipe_get({"name": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"]
        assert "recipe_list" in result["error"]

    def test_missing_name(self, cfg):
        result = recipe_get({})
        assert "error" in result
        assert "name" in result["error"].lower()

    def test_empty_name(self, cfg):
        result = recipe_get({"name": ""})
        assert "error" in result

    def test_corrupted_yaml(self, cfg):
        recipes_dir = cfg / ".crucible" / "recipes"
        recipes_dir.mkdir(parents=True)
        (recipes_dir / "broken.yaml").write_text(": : : [[[")

        result = recipe_get({"name": "broken"})
        assert "error" in result

    def test_non_dict_yaml(self, cfg):
        recipes_dir = cfg / ".crucible" / "recipes"
        recipes_dir.mkdir(parents=True)
        (recipes_dir / "scalar.yaml").write_text("just a string value")

        result = recipe_get({"name": "scalar"})
        assert "error" in result
        assert "invalid format" in result["error"]


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestRecipeRoundtrip:
    def test_save_list_get_roundtrip(self, cfg):
        """Full cycle: save -> list -> get -> verify all data intact."""
        original = _make_recipe_args(
            name="roundtrip-test",
            steps=[
                {"tool": "provision_project", "args": {"project_name": "demo", "count": 1}, "note": "create pod"},
                {"tool": "fleet_refresh", "args": {}, "note": "wait for SSH"},
                {"tool": "bootstrap_project", "args": {"project_name": "demo"}, "note": "install deps"},
                {"tool": "run_project", "args": {"project_name": "demo", "overrides": {"MODEL": "yolo11n.pt"}}, "note": "train"},
                {"tool": "collect_project_results", "args": {"run_id": "demo_123"}, "note": "get metrics"},
                {"tool": "destroy_nodes", "args": {"node_names": ["demo-01"]}, "note": "cleanup"},
            ],
            environment={
                "gpu": "NVIDIA GeForce RTX 4090",
                "torch": "2.6.0+cu124",
                "python": "3.11",
                "ultralytics": "8.4.30",
                "provider": "runpod",
            },
            results={
                "best_metric": {"map50_95_b": 0.733, "precision_b": 0.909},
                "wandb_url": "https://wandb.ai/user/project",
                "linked_run_ids": ["demo_123"],
            },
            gotchas=[
                {"issue": "torch cu130 too new", "fix": "pin torch 2.6.0+cu124"},
                {"issue": "W&B disabled by default", "fix": "yolo settings wandb=true"},
            ],
        )

        # Save
        save_result = recipe_save(original)
        assert save_result["saved"] is True

        # List
        list_result = recipe_list({})
        assert list_result["total"] == 1
        assert list_result["recipes"][0]["name"] == "roundtrip-test"

        # Get
        recipe = recipe_get({"name": "roundtrip-test"})
        assert recipe["name"] == "roundtrip-test"
        assert len(recipe["steps"]) == 6
        assert recipe["steps"][0]["tool"] == "provision_project"
        assert recipe["steps"][3]["args"]["overrides"]["MODEL"] == "yolo11n.pt"
        assert recipe["steps"][5]["tool"] == "destroy_nodes"
        assert recipe["environment"]["torch"] == "2.6.0+cu124"
        assert recipe["results"]["best_metric"]["map50_95_b"] == 0.733
        assert recipe["results"]["wandb_url"] == "https://wandb.ai/user/project"
        assert len(recipe["gotchas"]) == 2

    def test_multiple_recipes_isolated(self, cfg):
        recipe_save(_make_recipe_args(name="recipe-a", title="A", tags=["a"]))
        recipe_save(_make_recipe_args(name="recipe-b", title="B", tags=["b"]))
        recipe_save(_make_recipe_args(name="recipe-c", title="C", tags=["c"]))

        assert recipe_list({})["total"] == 3
        assert recipe_get({"name": "recipe-a"})["title"] == "A"
        assert recipe_get({"name": "recipe-b"})["title"] == "B"
        assert recipe_get({"name": "recipe-c"})["title"] == "C"

    def test_overwrite_preserves_other_recipes(self, cfg):
        recipe_save(_make_recipe_args(name="keep-me", title="Original"))
        recipe_save(_make_recipe_args(name="update-me", title="V1"))
        recipe_save(_make_recipe_args(name="update-me", title="V2"))

        assert recipe_list({})["total"] == 2
        assert recipe_get({"name": "keep-me"})["title"] == "Original"
        assert recipe_get({"name": "update-me"})["title"] == "V2"
