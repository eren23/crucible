"""Tests for the evaluators plugin family — Phase 3.3."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from crucible.core.evaluators import (
    EvalResult,
    EvalValidationResult,
    EvaluatorPlugin,
    discover_evaluator_plugins,
    instantiate_evaluator,
    list_evaluators,
    register_evaluator,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_lm_eval_harness_is_registered_as_builtin(self):
        names = list_evaluators()
        assert "lm_eval_harness" in names

    def test_instantiate_returns_plugin_subclass(self):
        ev = instantiate_evaluator("lm_eval_harness", {"tasks": ["hellaswag"]})
        from crucible.evaluators.lm_eval_harness import LMEvalHarnessEvaluator
        assert isinstance(ev, LMEvalHarnessEvaluator)
        assert ev.name == "lm_eval_harness"
        assert ev.config["tasks"] == ["hellaswag"]

    def test_unknown_evaluator_raises_key_error(self):
        with pytest.raises(KeyError, match="No evaluator registered"):
            instantiate_evaluator("does_not_exist")

    def test_register_then_list_then_get(self):
        class _Stub(EvaluatorPlugin):
            def validate(self):
                return EvalValidationResult(valid=True)
            def evaluate(self, checkpoint_path):
                return EvalResult(scores={"x": 1.0})

        register_evaluator("_test_stub", _Stub)
        try:
            assert "_test_stub" in list_evaluators()
            ev = instantiate_evaluator("_test_stub", {})
            assert isinstance(ev, _Stub)
        finally:
            # Best-effort cleanup — the registry has no unregister API.
            from crucible.core.evaluators import _EVALUATOR_REGISTRY
            _EVALUATOR_REGISTRY._registry.pop("_test_stub", None)
            _EVALUATOR_REGISTRY._meta.pop("_test_stub", None)


# ---------------------------------------------------------------------------
# LMEvalHarnessEvaluator — validate
# ---------------------------------------------------------------------------


class TestLMEvalHarnessValidate:
    def test_missing_tasks_fails_validation(self):
        ev = instantiate_evaluator("lm_eval_harness", {})
        out = ev.validate()
        assert out.valid is False
        assert any("tasks" in e for e in out.errors)

    def test_non_list_tasks_fails_validation(self):
        ev = instantiate_evaluator("lm_eval_harness", {"tasks": "hellaswag"})
        out = ev.validate()
        assert out.valid is False
        assert any("must be a list" in e for e in out.errors)

    def test_missing_binary_surfaces_actionable_error(self, monkeypatch):
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.shutil.which",
            lambda _name: None,
        )
        ev = instantiate_evaluator("lm_eval_harness", {"tasks": ["hellaswag"]})
        out = ev.validate()
        assert out.valid is False
        assert any("lm_eval CLI not found" in e for e in out.errors)
        assert any("pip install" in e for e in out.errors)

    def test_validate_passes_when_binary_present_and_tasks_set(self, monkeypatch):
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.shutil.which",
            lambda _name: "/fake/path/lm_eval",
        )
        ev = instantiate_evaluator("lm_eval_harness", {"tasks": ["hellaswag"]})
        out = ev.validate()
        assert out.valid is True
        assert out.errors == []


# ---------------------------------------------------------------------------
# LMEvalHarnessEvaluator — evaluate
# ---------------------------------------------------------------------------


class TestLMEvalHarnessEvaluate:
    def test_missing_checkpoint_returns_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.shutil.which",
            lambda _name: "/fake/lm_eval",
        )
        ev = instantiate_evaluator("lm_eval_harness", {"tasks": ["hellaswag"]})
        result = ev.evaluate(tmp_path / "nonexistent.bin")
        assert result.success is False
        assert "does not exist" in result.error

    def test_subprocess_failure_returns_failure(self, tmp_path, monkeypatch):
        import subprocess
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.shutil.which",
            lambda _name: "/fake/lm_eval",
        )
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=1,
                stdout="", stderr="ImportError: torch",
            )
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.subprocess.run", fake_run
        )
        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"")
        ev = instantiate_evaluator("lm_eval_harness", {"tasks": ["hellaswag"]})
        result = ev.evaluate(ckpt)
        assert result.success is False
        assert "exited with code 1" in result.error
        assert "ImportError: torch" in result.metadata["stderr"]

    def test_subprocess_timeout_returns_failure(self, tmp_path, monkeypatch):
        import subprocess
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.shutil.which",
            lambda _name: "/fake/lm_eval",
        )
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=10)
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.subprocess.run", fake_run
        )
        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"")
        ev = instantiate_evaluator("lm_eval_harness", {"tasks": ["hellaswag"]})
        result = ev.evaluate(ckpt)
        assert result.success is False
        assert "timed out" in result.error

    def test_successful_run_parses_json_results(self, tmp_path, monkeypatch):
        import subprocess
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.shutil.which",
            lambda _name: "/fake/lm_eval",
        )

        # Realistic lm_eval stdout ends with a "results" JSON block.
        fake_stdout = (
            "Some lm_eval logging banner\n"
            "Running on 1 task\n"
            "{\n  \"results\": {\n"
            "    \"hellaswag\": {\"acc\": 0.62, \"acc_norm\": 0.65},\n"
            "    \"arc_easy\": {\"acc\": 0.71, \"acc_norm\": 0.72}\n"
            "  }\n}\n"
        )

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=fake_stdout, stderr="",
            )
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.subprocess.run", fake_run
        )

        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"")
        ev = instantiate_evaluator(
            "lm_eval_harness",
            {"tasks": ["hellaswag", "arc_easy"], "num_fewshot": 5},
        )
        result = ev.evaluate(ckpt)
        assert result.success is True
        assert result.scores == {
            "hellaswag.acc": 0.62,
            "hellaswag.acc_norm": 0.65,
            "arc_easy.acc": 0.71,
            "arc_easy.acc_norm": 0.72,
        }
        assert result.metadata["tasks"] == ["hellaswag", "arc_easy"]
        assert result.metadata["num_fewshot"] == 5

    def test_command_construction(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "crucible.evaluators.lm_eval_harness.shutil.which",
            lambda _name: "/fake/lm_eval",
        )
        ev = instantiate_evaluator(
            "lm_eval_harness",
            {
                "tasks": ["hellaswag"],
                "batch_size": 16,
                "num_fewshot": 5,
                "limit": 100,
                "extra_args": ["--device", "cuda"],
            },
        )
        cmd = ev._build_command(tmp_path / "ckpt.bin")
        assert cmd[0] == "lm_eval"
        assert "--tasks" in cmd
        assert "hellaswag" in cmd
        assert "16" in cmd  # batch_size
        assert "5" in cmd   # num_fewshot
        assert "--limit" in cmd
        assert "100" in cmd
        assert cmd[-2:] == ["--device", "cuda"]


# ---------------------------------------------------------------------------
# MCP dispatch
# ---------------------------------------------------------------------------


def test_evaluator_list_via_mcp(monkeypatch):
    from crucible.mcp.tools import TOOL_DISPATCH
    # Bypass _get_config so we don't need a real project.
    from crucible.core.errors import CrucibleError
    def boom():
        raise CrucibleError("no project")
    monkeypatch.setattr("crucible.mcp.tools._get_config", boom)

    handler = TOOL_DISPATCH["evaluator_list"]
    out = handler({})
    assert out["count"] >= 1
    names = [e.get("name") for e in out["evaluators"]]
    assert "lm_eval_harness" in names


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_no_project_returns_list_without_error(self):
        result = discover_evaluator_plugins(project_root=None)
        assert isinstance(result, list)

    def test_finds_local_plugin(self, tmp_path):
        plugin_dir = tmp_path / ".crucible" / "plugins" / "evaluators"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "probe_discovered_eval.py").write_text(
            "from crucible.core.evaluators import (\n"
            "    EvalResult, EvalValidationResult, EvaluatorPlugin, register_evaluator,\n"
            ")\n"
            "class ProbeEvaluator(EvaluatorPlugin):\n"
            "    def validate(self):\n"
            "        return EvalValidationResult(valid=True)\n"
            "    def evaluate(self, checkpoint_path):\n"
            "        return EvalResult(scores={'probe.score': 1.0})\n"
            "register_evaluator('_probe_discovered_eval', ProbeEvaluator)\n",
            encoding="utf-8",
        )
        import sys
        from crucible.core.evaluators import _EVALUATOR_REGISTRY
        try:
            loaded = discover_evaluator_plugins(project_root=tmp_path)
            assert "probe_discovered_eval" in loaded
            assert "_probe_discovered_eval" in list_evaluators()
            ev = instantiate_evaluator("_probe_discovered_eval", {})
            assert ev.evaluate(tmp_path / "x").scores == {"probe.score": 1.0}
        finally:
            _EVALUATOR_REGISTRY._registry.pop("_probe_discovered_eval", None)
            _EVALUATOR_REGISTRY._meta.pop("_probe_discovered_eval", None)
            sys.modules.pop(
                "_crucible_plugin_evaluator_local_probe_discovered_eval", None
            )
