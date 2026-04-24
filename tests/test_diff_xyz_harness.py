"""Tests for evaluation/diff_xyz/harness.py end-to-end flow with a dummy backend.

Full API-backed runs live in a separate integration test (not run in CI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.diff_xyz.dataset import DiffXYZSample
from evaluation.diff_xyz.harness import (
    SampleResult,
    _aggregate,
    _aggregate_per_lang,
    main,
    run_benchmark,
    score_sample,
)
from evaluation.diff_xyz.models import DummyBackend


def _make_sample(lang: str = "python", idx: int = 0) -> DiffXYZSample:
    return DiffXYZSample(
        repo=f"org/repo{idx}",
        commit=f"c{idx}",
        path="src/main.py",
        lang=lang,
        old_code="line_a\nline_b\nline_c",
        new_code="line_a\nline_B\nline_c",
        udiff="@@ -1,3 +1,3 @@\n line_a\n-line_b\n+line_B\n line_c\n",
        udiff_h="@@...@@\n line_a\n-line_b\n+line_B\n line_c\n",
        udiff_l="CON line_a\nDEL line_b\nADD line_B\nCON line_c\n",
        search_replace="<<<<<<< SEARCH\nline_b\n=======\nline_B\n>>>>>>> REPLACE\n",
        n_added=1, n_removed=1, n_hunks=1, change_kind="edit",
    )


# ---------------------------------------------------------------------------
# score_sample — task routing
# ---------------------------------------------------------------------------


class _FixedBackend:
    """Test backend that returns a fixed string regardless of prompt."""

    def __init__(self, output: str):
        self.output = output

    def generate(self, sys_p, usr_p, *, max_tokens=4096, temperature=0.0):
        del sys_p, usr_p, max_tokens, temperature
        return self.output


def test_score_sample_apply_perfect():
    sample = _make_sample()
    backend = _FixedBackend(sample.new_code)
    r = score_sample(backend, sample, idx=0, task="apply", fmt="udiff",
                     sys_mode="format", max_tokens=4096, temperature=0.0)
    assert r.em == 1.0
    assert r.iou == 1.0
    assert r.error == ""


def test_score_sample_apply_wrong_output():
    sample = _make_sample()
    backend = _FixedBackend("totally wrong output")
    r = score_sample(backend, sample, idx=0, task="apply", fmt="udiff",
                     sys_mode="format", max_tokens=4096, temperature=0.0)
    assert r.em == 0.0


def test_score_sample_anti_apply_perfect():
    sample = _make_sample()
    backend = _FixedBackend(sample.old_code)
    r = score_sample(backend, sample, idx=0, task="anti_apply", fmt="udiff",
                     sys_mode="format", max_tokens=4096, temperature=0.0)
    assert r.em == 1.0


def test_score_sample_diff_gen_perfect():
    sample = _make_sample()
    backend = _FixedBackend(sample.udiff)
    r = score_sample(backend, sample, idx=0, task="diff_gen", fmt="udiff",
                     sys_mode="format", max_tokens=4096, temperature=0.0)
    assert r.em == 1.0
    assert r.parsing_rate == 1.0
    assert r.applying_rate == 1.0
    assert r.f1_plus == 1.0
    assert r.f1_minus == 1.0


def test_score_sample_diff_gen_parse_fail():
    sample = _make_sample()
    backend = _FixedBackend("totally unparseable")
    r = score_sample(backend, sample, idx=0, task="diff_gen", fmt="udiff",
                     sys_mode="format", max_tokens=4096, temperature=0.0)
    assert r.parsing_rate == 0.0
    assert r.applying_rate == 0.0
    assert r.em == 0.0


def test_score_sample_strips_markdown_fence():
    sample = _make_sample()
    fenced = "```python\n" + sample.new_code + "\n```"
    backend = _FixedBackend(fenced)
    r = score_sample(backend, sample, idx=0, task="apply", fmt="udiff",
                     sys_mode="format", max_tokens=4096, temperature=0.0)
    assert r.em == 1.0


def test_score_sample_model_error_captured():
    from evaluation.diff_xyz.models import ModelError

    class _BrokenBackend:
        def generate(self, *a, **kw):
            raise ModelError("simulated rate limit")

    sample = _make_sample()
    r = score_sample(_BrokenBackend(), sample, idx=0, task="apply", fmt="udiff",
                     sys_mode="format", max_tokens=4096, temperature=0.0)
    assert r.em == 0.0
    assert "simulated rate limit" in r.error


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def _sr(em: float, iou: float, lang: str = "python", task: str = "apply") -> SampleResult:
    return SampleResult(
        idx=0, lang=lang, task=task, fmt="udiff",
        em=em, iou=iou, parsing_rate=0.0, applying_rate=0.0,
        f1_plus=0.0, f1_minus=0.0, response_chars=0, elapsed_s=0.1,
    )


def test_aggregate_apply_task():
    rows = [_sr(1.0, 1.0), _sr(0.0, 0.5), _sr(1.0, 0.75)]
    agg = _aggregate(rows, "apply")
    assert agg["EM"] == pytest.approx(2 / 3)
    assert agg["IoU"] == pytest.approx(0.75)
    # Apply task must not include diff_gen-only fields
    assert "parsing_rate" not in agg
    assert "f1_plus" not in agg


def test_aggregate_diff_gen_includes_rates():
    rows = [
        SampleResult(0, "py", "diff_gen", "udiff", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10, 0.1),
        SampleResult(1, "py", "diff_gen", "udiff", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.1),
    ]
    agg = _aggregate(rows, "diff_gen")
    assert agg["EM"] == 0.5
    assert agg["parsing_rate"] == 0.5
    assert agg["applying_rate"] == 0.5
    assert agg["f1_plus"] == 0.5


def test_aggregate_empty_returns_empty():
    assert _aggregate([], "apply") == {}


def test_aggregate_counts_errors():
    rows = [_sr(1.0, 1.0)]
    rows[0].error = "rate limited"
    agg = _aggregate(rows, "apply")
    assert agg["errors"] == 1


def test_aggregate_per_lang_buckets():
    rows = [
        _sr(1.0, 1.0, lang="python"),
        _sr(0.0, 0.5, lang="python"),
        _sr(1.0, 0.8, lang="rust"),
    ]
    per = _aggregate_per_lang(rows, "apply")
    assert set(per.keys()) == {"python", "rust"}
    assert per["python"]["EM"] == 0.5
    assert per["rust"]["EM"] == 1.0


# ---------------------------------------------------------------------------
# run_benchmark end-to-end (with load_samples mocked)
# ---------------------------------------------------------------------------


def _fake_load_samples(*args, **kwargs):
    return [_make_sample(lang="python", idx=i) for i in range(3)]


def test_run_benchmark_end_to_end(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "evaluation.diff_xyz.harness.load_samples", _fake_load_samples
    )
    # Oracle backend: returns the Apply-task's target (new_code) verbatim.
    # All 3 synthetic samples share the same new_code → _FixedBackend suffices.
    oracle = _FixedBackend(_make_sample().new_code)

    class _OracleFactory:
        @staticmethod
        def resolve(_spec):
            return oracle

    monkeypatch.setattr(
        "evaluation.diff_xyz.harness.resolve_backend", _OracleFactory.resolve
    )
    out = run_benchmark(
        model_spec="dummy:oracle",
        task="apply",
        fmt="udiff",
        sys_mode="format",
        limit=3,
        progress=False,
    )
    assert out["n_samples"] == 3
    assert out["metrics"]["EM"] == 1.0
    assert out["metrics"]["IoU"] == 1.0
    assert "per_lang" in out and "python" in out["per_lang"]


def test_run_benchmark_invalid_task_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "evaluation.diff_xyz.harness.load_samples", _fake_load_samples
    )
    with pytest.raises(ValueError, match="unknown task"):
        run_benchmark(
            model_spec="dummy:echo", task="bogus", fmt="udiff", progress=False,
        )


def test_run_benchmark_invalid_format_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "evaluation.diff_xyz.harness.load_samples", _fake_load_samples
    )
    with pytest.raises(ValueError, match="unknown format"):
        run_benchmark(
            model_spec="dummy:echo", task="apply", fmt="bogus", progress=False,
        )


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_writes_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(
        "evaluation.diff_xyz.harness.load_samples", _fake_load_samples
    )
    # Use echo backend + Apply task; score won't be 1.0 but JSON must be well-formed.
    out_file = tmp_path / "out.json"
    rc = main([
        "--model", "dummy:echo",
        "--task", "apply",
        "--format", "udiff",
        "--system-prompt", "format",
        "--out", str(out_file),
        "--no-progress",
    ])
    assert rc == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["task"] == "apply"
    assert data["n_samples"] == 3
    assert "EM" in data["metrics"]
    assert "IoU" in data["metrics"]
    assert data["model"] == "dummy:echo"


def test_cli_accepts_checkpoint_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # eval_watcher passes --checkpoint; harness must accept it silently.
    monkeypatch.setattr(
        "evaluation.diff_xyz.harness.load_samples", _fake_load_samples
    )
    out_file = tmp_path / "ckpt.json"
    rc = main([
        "--model", "dummy:echo",
        "--task", "apply",
        "--format", "udiff",
        "--checkpoint", "/tmp/fake.pt",
        "--out", str(out_file),
        "--no-progress",
    ])
    assert rc == 0
