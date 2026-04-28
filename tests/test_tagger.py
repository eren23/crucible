"""Tests for crucible.runner.tagger — recipe + design auto-tagging."""
from __future__ import annotations

from crucible.runner.tagger import merge_auto_tags, tag_design, tag_recipe


# ---------------------------------------------------------------------------
# tag_recipe
# ---------------------------------------------------------------------------


def test_tag_recipe_empty():
    assert tag_recipe({}) == []


def test_tag_recipe_non_dict_input_safe():
    assert tag_recipe(None) == []  # type: ignore[arg-type]
    assert tag_recipe([]) == []  # type: ignore[arg-type]


def test_tag_recipe_preset_from_environment():
    rec = {"environment": {"PRESET": "smoke"}}
    tags = tag_recipe(rec)
    assert "preset:smoke" in tags


def test_tag_recipe_gpu_and_family():
    rec = {"environment": {"GPU_TYPE": "RTX 4090", "WANDB_PROJECT": "diff-xyz"}}
    tags = tag_recipe(rec)
    assert "gpu:rtx_4090" in tags
    assert "family:diff-xyz" in tags


def test_tag_recipe_outcome_from_results():
    rec = {"results": {"status": "success", "val_bpb": 1.21}}
    tags = tag_recipe(rec)
    assert "outcome:success" in tags
    assert "val_bpb_bucket:1.0-1.5" in tags


def test_tag_recipe_val_bpb_buckets_low():
    assert "val_bpb_bucket:<1.0" in tag_recipe({"results": {"val_bpb": 0.95}})


def test_tag_recipe_val_bpb_buckets_high():
    assert "val_bpb_bucket:3.0+" in tag_recipe({"results": {"val_bpb": 4.7}})


def test_tag_recipe_tools():
    rec = {
        "steps": [
            {"tool": "provision_nodes"},
            {"tool": "dispatch_experiments"},
            {"tool": "provision_nodes"},
        ]
    }
    tags = tag_recipe(rec)
    assert "tool:provision_nodes" in tags
    assert "tool:dispatch_experiments" in tags


def test_tag_recipe_caps_tools_to_8():
    rec = {"steps": [{"tool": f"tool_{i}"} for i in range(20)]}
    tool_tags = [t for t in tag_recipe(rec) if t.startswith("tool:")]
    assert len(tool_tags) == 8


def test_tag_recipe_project_spec_fallback_family():
    rec = {"project_spec": "my_project.yaml"}
    tags = tag_recipe(rec)
    assert "family:my_project" in tags


def test_tag_recipe_project_spec_with_yaml_chars_not_corrupted():
    """Regression: rstrip would have stripped any trailing y/a/m/l/. chars,
    turning 'play.yaml' into 'p'. removesuffix only strips the literal suffix."""
    rec = {"project_spec": "play.yaml"}
    tags = tag_recipe(rec)
    assert "family:play" in tags
    assert "family:p" not in tags


def test_tag_recipe_project_spec_no_yaml_suffix_preserved():
    rec = {"project_spec": "my-tray"}  # ends in 'y', no .yaml
    tags = tag_recipe(rec)
    assert "family:my-tray" in tags


def test_tag_recipe_project_spec_yml_extension():
    """Some projects use .yml, not .yaml — both should be stripped."""
    rec = {"project_spec": "my_project.yml"}
    tags = tag_recipe(rec)
    assert "family:my_project" in tags


def test_tag_recipe_project_spec_path_form():
    """Path-form project_spec should yield just the stem, not 'subdir/play'."""
    rec = {"project_spec": "subdir/play.yaml"}
    tags = tag_recipe(rec)
    assert "family:play" in tags
    assert not any("subdir" in t for t in tags)


def test_tag_design_modality_lm_with_vocabulary_size_alias():
    """Some configs use VOCABULARY_SIZE instead of VOCAB_SIZE — both should
    trigger modality:lm now that the alias was added."""
    d = {"config": {"VOCABULARY_SIZE": 50000}}
    tags = tag_design(d)
    assert "modality:lm" in tags


# ---------------------------------------------------------------------------
# tag_design
# ---------------------------------------------------------------------------


def test_tag_design_empty():
    assert tag_design({}) == []


def test_tag_design_preset_known():
    assert "preset:smoke" in tag_design({"base_preset": "smoke"})


def test_tag_design_preset_unknown_still_emitted():
    """Unknown presets get tagged anyway — keeps the tag space discoverable."""
    assert "preset:custom_thing" in tag_design({"base_preset": "custom_thing"})


def test_tag_design_backend_status_family():
    d = {"backend": "torch", "status": "active", "family": "looped"}
    tags = tag_design(d)
    assert "backend:torch" in tags
    assert "status:active" in tags
    assert "architecture:looped" in tags


def test_tag_design_modality_diffusion():
    d = {"config": {"DIFFUSION_STEPS": 1000, "NOISE_SCHED": "linear"}}
    tags = tag_design(d)
    assert "modality:diffusion" in tags


def test_tag_design_modality_world_model():
    d = {"config": {"JEPA_TARGET_DIM": 128, "FRAME_HORIZON": 8}}
    tags = tag_design(d)
    assert "modality:world_model" in tags


def test_tag_design_modality_lm():
    d = {"config": {"VOCAB_SIZE": 50000, "SEQ_LEN": 1024}}
    tags = tag_design(d)
    assert "modality:lm" in tags


def test_tag_design_architecture_from_config_when_no_family():
    d = {"config": {"MODEL_FAMILY": "convloop"}}
    tags = tag_design(d)
    assert "architecture:convloop" in tags


def test_tag_design_model_family_alone_does_not_imply_lm():
    """Regression: MODEL_FAMILY used to be in the LM signal list, so a
    diffusion config that set MODEL_FAMILY=ddpm got mistagged as modality:lm.
    A bare MODEL_FAMILY key with no other modality signal should produce NO
    modality tag — the architecture: tag carries the family info instead."""
    d = {"config": {"MODEL_FAMILY": "ddpm"}}
    tags = tag_design(d)
    assert "architecture:ddpm" in tags
    assert "modality:lm" not in tags
    # Also: a real diffusion config with explicit signals + MODEL_FAMILY
    # should be tagged as diffusion (signals match first).
    d2 = {"config": {"MODEL_FAMILY": "ddpm", "DIFFUSION_STEPS": 1000}}
    tags2 = tag_design(d2)
    assert "modality:diffusion" in tags2
    assert "modality:lm" not in tags2


# ---------------------------------------------------------------------------
# merge_auto_tags
# ---------------------------------------------------------------------------


def test_merge_keeps_user_order_first():
    user = ["alpha", "beta"]
    auto = ["preset:smoke", "alpha"]  # 'alpha' duplicate should not appear twice
    out = merge_auto_tags(user, auto)
    assert out == ["alpha", "beta", "preset:smoke"]


def test_merge_dedup_within_user():
    out = merge_auto_tags(["a", "a", "b"], [])
    assert out == ["a", "b"]


def test_merge_handles_empty():
    assert merge_auto_tags([], []) == []
    assert merge_auto_tags([], ["preset:smoke"]) == ["preset:smoke"]
    assert merge_auto_tags(["x"], []) == ["x"]
