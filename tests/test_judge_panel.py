"""Tests for JudgeConfig + JudgePanel — the LM-as-judge separation contract.

Mirrors GIANTS' rule: any LM-as-reward loop MUST declare a judge for
reward (used during selection/scoring) and a different judge for
evaluation (used for final ranking). Reward-hacking is the dominant
failure mode of LM-as-judge loops; separation is the standard mitigation.
"""
from __future__ import annotations

import warnings

import pytest

from crucible.core.config import JudgeConfig, JudgePanel
from crucible.core.errors import ConfigError


def _ok_panel() -> JudgePanel:
    return JudgePanel(
        reward_judge=JudgeConfig(model="gemini-2.5-flash", family="gemini"),
        eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        enforce_separation=True,
    )


class TestJudgeConfig:
    def test_fields(self) -> None:
        jc = JudgeConfig(model="claude-haiku-4-5", family="claude")
        assert jc.model == "claude-haiku-4-5"
        assert jc.family == "claude"
        assert jc.prompt_template == ""

    def test_optional_prompt_template(self) -> None:
        jc = JudgeConfig(
            model="gpt-5",
            family="openai",
            prompt_template="Score 1-10: {output}",
        )
        assert jc.prompt_template.startswith("Score")


class TestJudgePanelStructure:
    def test_required_reward_and_evaluation(self) -> None:
        panel = _ok_panel()
        assert panel.reward_judge.model == "gemini-2.5-flash"
        assert panel.eval_judge.model == "claude-opus-4-7"
        assert panel.audit_judge is None

    def test_audit_judge_optional(self) -> None:
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="gemini-2.5-flash", family="gemini"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            audit_judge=JudgeConfig(model="qwen3-14b", family="qwen"),
        )
        assert panel.audit_judge is not None
        assert panel.audit_judge.family == "qwen"

    def test_is_configured_true_when_models_set(self) -> None:
        assert _ok_panel().is_configured() is True

    def test_is_configured_false_when_empty(self) -> None:
        empty = JudgePanel(
            reward_judge=JudgeConfig(model="", family=""),
            eval_judge=JudgeConfig(model="", family=""),
        )
        assert empty.is_configured() is False


class TestAssertSeparated:
    def test_passes_when_properly_separated(self) -> None:
        _ok_panel().assert_separated()

    def test_rejects_same_model(self) -> None:
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        with pytest.raises(ConfigError, match="same model"):
            panel.assert_separated()

    def test_rejects_same_family(self) -> None:
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="claude-haiku-4-5", family="claude"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        with pytest.raises(ConfigError, match="same family"):
            panel.assert_separated()

    def test_audit_must_differ_from_reward_and_evaluation(self) -> None:
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="gemini-2.5-flash", family="gemini"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            audit_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        with pytest.raises(ConfigError, match="audit"):
            panel.assert_separated()

    def test_skip_when_panel_unconfigured(self) -> None:
        empty = JudgePanel(
            reward_judge=JudgeConfig(model="", family=""),
            eval_judge=JudgeConfig(model="", family=""),
        )
        empty.assert_separated()

    def test_configured_panel_requires_reward_family(self) -> None:
        # Models set but reward family blank → can't enforce family-level
        # separation. Configured panels must declare family.
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="gemini-2.5-flash", family=""),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        with pytest.raises(ConfigError, match="reward_judge.family"):
            panel.assert_separated()

    def test_configured_panel_requires_eval_family(self) -> None:
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="gemini-2.5-flash", family="gemini"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family=""),
        )
        with pytest.raises(ConfigError, match="eval_judge.family"):
            panel.assert_separated()

    def test_configured_audit_judge_requires_family(self) -> None:
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="gemini-2.5-flash", family="gemini"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            audit_judge=JudgeConfig(model="qwen3-14b", family=""),
        )
        with pytest.raises(ConfigError, match="audit_judge.family"):
            panel.assert_separated()

    def test_warning_only_mode_does_not_raise(self) -> None:
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            enforce_separation=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            panel.assert_separated()
        assert any("judge" in str(w.message).lower() for w in caught)

    def test_error_message_names_offending_fields(self) -> None:
        panel = JudgePanel(
            reward_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        with pytest.raises(ConfigError) as exc_info:
            panel.assert_separated()
        msg = str(exc_info.value)
        assert "reward_judge" in msg
        assert "eval_judge" in msg


class TestFromDict:
    def test_builds_from_minimal_dict(self) -> None:
        panel = JudgePanel.from_dict({
            "reward_judge": {"model": "gemini-2.5-flash", "family": "gemini"},
            "eval_judge": {"model": "claude-opus-4-7", "family": "claude"},
        })
        assert panel.reward_judge.family == "gemini"
        assert panel.eval_judge.family == "claude"
        assert panel.audit_judge is None
        assert panel.enforce_separation is True

    def test_includes_audit_when_present(self) -> None:
        panel = JudgePanel.from_dict({
            "reward_judge": {"model": "gemini-2.5-flash", "family": "gemini"},
            "eval_judge": {"model": "claude-opus-4-7", "family": "claude"},
            "audit_judge": {"model": "qwen3-14b", "family": "qwen"},
            "enforce_separation": False,
        })
        assert panel.audit_judge is not None
        assert panel.audit_judge.family == "qwen"
        assert panel.enforce_separation is False

    def test_empty_dict_yields_unconfigured_panel(self) -> None:
        panel = JudgePanel.from_dict({})
        assert panel.is_configured() is False


class TestProjectConfigIntegration:
    def test_judges_field_default_empty(self, tmp_path) -> None:
        from crucible.core.config import load_config

        cfg_path = tmp_path / "crucible.yaml"
        cfg_path.write_text("name: test\n", encoding="utf-8")
        cfg = load_config(cfg_path)
        assert cfg.judges.is_configured() is False

    def test_judges_field_loaded_from_yaml(self, tmp_path) -> None:
        from crucible.core.config import load_config

        cfg_path = tmp_path / "crucible.yaml"
        cfg_path.write_text(
            "name: test\n"
            "judges:\n"
            "  reward_judge:\n"
            "    model: gemini-2.5-flash\n"
            "    family: gemini\n"
            "  eval_judge:\n"
            "    model: claude-opus-4-7\n"
            "    family: claude\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_path)
        assert cfg.judges.is_configured() is True
        assert cfg.judges.reward_judge.family == "gemini"
        assert cfg.judges.eval_judge.family == "claude"
        cfg.judges.assert_separated()
