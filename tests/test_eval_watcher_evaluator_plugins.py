"""Tests for the H.4.D eval_watcher -> evaluators registry bridge.

Phase 3.3 shipped the evaluator plugin family; H.4 surfaced that the
eval_watcher daemon wasn't consuming it. The eval_suite block now
accepts ``{evaluator: <name>, config: {...}}`` entries that dispatch
through the Phase 3.3 registry instead of shelling out to a script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_eval_spec_dataclass_supports_evaluator_field():
    from crucible.runner.eval_watcher import EvalSpec
    spec = EvalSpec(evaluator="lm_eval_harness", config={"tasks": ["hellaswag"]})
    assert spec.evaluator == "lm_eval_harness"
    assert spec.config == {"tasks": ["hellaswag"]}
    spec2 = EvalSpec(script="probe.py", args=["--task", "foo"])
    assert spec2.script == "probe.py"
    assert spec2.args == ["--task", "foo"]


def test_read_eval_suite_parses_evaluator_entries(tmp_path, monkeypatch):
    import yaml
    proj_dir = tmp_path / ".crucible" / "projects"
    proj_dir.mkdir(parents=True)
    (proj_dir / "demo.yaml").write_text(yaml.safe_dump({
        "name": "demo",
        "eval_suite": [
            {"evaluator": "lm_eval_harness",
             "config": {"tasks": ["hellaswag"], "batch_size": 8}},
            {"script": "legacy.py", "args": ["--task", "x"]},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(
        "crucible.runner.eval_watcher._project_root", lambda: tmp_path,
    )
    from crucible.runner.eval_watcher import _read_eval_suite
    suite = _read_eval_suite("demo")
    assert len(suite) == 2
    assert suite[0].evaluator == "lm_eval_harness"
    assert suite[0].config == {"tasks": ["hellaswag"], "batch_size": 8}
    assert suite[0].script == ""
    assert suite[1].script == "legacy.py"
    assert suite[1].evaluator == ""


def test_run_one_eval_dispatches_to_evaluator_plugin(tmp_path):
    from crucible.core import evaluators as _evals
    from crucible.runner.eval_watcher import EvalSpec, _run_one_eval

    class _Stub(_evals.EvaluatorPlugin):
        def validate(self):
            return _evals.EvalValidationResult(valid=True)
        def evaluate(self, checkpoint_path):
            return _evals.EvalResult(
                scores={"task1.acc": 0.7, "task2.acc": 0.5},
                success=True,
                metadata={"version": "0.4.0"},
            )

    _evals.register_evaluator("_test_stub_eval", _Stub)
    try:
        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"")
        row = _run_one_eval(
            ckpt_path=ckpt, ckpt_sha="abc", label="proxy",
            spec=EvalSpec(evaluator="_test_stub_eval", config={}),
            env={},
        )
        assert row["ok"] is True
        assert row["result"] == {"task1.acc": 0.7, "task2.acc": 0.5}
        assert row["script"] == "evaluator:_test_stub_eval"
        assert row["metadata"]["version"] == "0.4.0"
    finally:
        _evals._EVALUATOR_REGISTRY._registry.pop("_test_stub_eval", None)
        _evals._EVALUATOR_REGISTRY._meta.pop("_test_stub_eval", None)


def test_run_one_eval_unknown_evaluator_returns_failure_row(tmp_path):
    from crucible.runner.eval_watcher import EvalSpec, _run_one_eval

    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"")
    row = _run_one_eval(
        ckpt_path=ckpt, ckpt_sha="abc", label="proxy",
        spec=EvalSpec(evaluator="does_not_exist", config={}),
        env={},
    )
    assert row["ok"] is False
    assert row["script"] == "evaluator:does_not_exist"
    assert "No evaluator registered" in row["stderr_tail"]


def test_run_one_eval_failed_evaluator_returns_failure_row(tmp_path):
    from crucible.core import evaluators as _evals
    from crucible.runner.eval_watcher import EvalSpec, _run_one_eval

    class _Failing(_evals.EvaluatorPlugin):
        def validate(self):
            return _evals.EvalValidationResult(valid=False, errors=["binary missing"])
        def evaluate(self, checkpoint_path):
            return _evals.EvalResult(
                scores={}, success=False, error="binary missing",
            )

    _evals.register_evaluator("_test_fail", _Failing)
    try:
        ckpt = tmp_path / "ckpt.bin"
        ckpt.write_bytes(b"")
        row = _run_one_eval(
            ckpt_path=ckpt, ckpt_sha="abc", label="proxy",
            spec=EvalSpec(evaluator="_test_fail", config={}),
            env={},
        )
        assert row["ok"] is False
        assert "binary missing" in row["stderr_tail"]
        assert row["result"] is None
    finally:
        _evals._EVALUATOR_REGISTRY._registry.pop("_test_fail", None)
        _evals._EVALUATOR_REGISTRY._meta.pop("_test_fail", None)
