"""Tests for eval_watcher._backfill_local_checkpoints."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def isolated_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run with cwd inside a fresh project so .crucible/ paths are scoped to tmp.

    Also resets the eval-watcher's in-memory seen-cache so tests that
    pre-seed `eval_watch.jsonl` on disk (bypassing `_append_log`) see
    their own data.
    """
    from crucible.runner import eval_watcher

    monkeypatch.chdir(tmp_path)
    eval_watcher._reset_seen_cache()
    return tmp_path


def _write_ckpt(path: Path, content: bytes = b"abc") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_backfill_empty_when_no_ckpts(isolated_project: Path):
    from crucible.runner.eval_watcher import EvalSpec, _backfill_local_checkpoints

    out = _backfill_local_checkpoints([EvalSpec(script="x.py", args=[])], env={})
    assert out == []


def test_backfill_runs_each_ckpt_against_each_spec(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.runner import eval_watcher
    from crucible.runner.eval_watcher import EvalSpec

    # Two checkpoints in the local cache.
    ckpt_dir = isolated_project / ".crucible" / "eval_watch_ckpts"
    _write_ckpt(ckpt_dir / "node1_step100.pt", b"ckpt-A")
    _write_ckpt(ckpt_dir / "node1_step200.pt", b"ckpt-B")

    suite = [EvalSpec(script="probe_a.py", args=[]), EvalSpec(script="probe_b.py", args=[])]

    captured: list[tuple[str, str, str]] = []

    def fake_run(ckpt_path, ckpt_sha, label, spec, env, timeout=1800):
        captured.append((ckpt_path.name, spec.script, ckpt_sha))
        return {
            "label": label, "script": spec.script, "ckpt_sha": ckpt_sha,
            "ok": True, "elapsed_s": 0.0, "result": {"score": 1.0},
            "stdout_tail": "", "stderr_tail": "", "ran_at": "now",
        }

    monkeypatch.setattr(eval_watcher, "_run_one_eval", fake_run)

    out = eval_watcher._backfill_local_checkpoints(suite, env={})
    # 2 ckpts × 2 scripts = 4 runs
    assert len(out) == 4
    assert all(row["backfilled"] is True for row in out)
    # Different ckpts → different SHAs
    shas = {sha for _, _, sha in captured}
    assert len(shas) == 2


def test_backfill_skips_already_logged_pairs(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If the JSONL already has (sha, script) for a pair, that pair is skipped."""
    from crucible.runner import eval_watcher
    from crucible.runner.eval_watcher import EvalSpec, _sha256_of_file

    ckpt_dir = isolated_project / ".crucible" / "eval_watch_ckpts"
    ckpt_path = ckpt_dir / "node1_step100.pt"
    _write_ckpt(ckpt_path, b"ckpt-X")

    sha = _sha256_of_file(ckpt_path)

    # Pre-seed the JSONL with one (sha, script) entry → that script should be skipped.
    log = isolated_project / ".crucible" / "eval_watch.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    log.write_text(_json.dumps({"ckpt_sha": sha, "script": "probe_a.py", "ok": True}) + "\n")

    suite = [EvalSpec(script="probe_a.py", args=[]), EvalSpec(script="probe_b.py", args=[])]

    runs: list[str] = []

    def fake_run(ckpt_path, ckpt_sha, label, spec, env, timeout=1800):
        runs.append(spec.script)
        return {
            "label": label, "script": spec.script, "ckpt_sha": ckpt_sha,
            "ok": True, "elapsed_s": 0.0, "result": None,
            "stdout_tail": "", "stderr_tail": "", "ran_at": "now",
        }

    monkeypatch.setattr(eval_watcher, "_run_one_eval", fake_run)

    out = eval_watcher._backfill_local_checkpoints(suite, env={})
    # probe_a was pre-logged → only probe_b runs
    assert runs == ["probe_b.py"]
    assert len(out) == 1


def test_seen_runs_skips_failed_rows_too(isolated_project: Path):
    """A row with ok=False still marks the (sha, script) as seen — failed
    runs must not be retried automatically (the user should clear the row to
    rerun)."""
    from crucible.runner.eval_watcher import _seen_runs
    import json as _json

    log = isolated_project / ".crucible" / "eval_watch.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        _json.dumps({"ckpt_sha": "abc", "script": "probe.py", "ok": False}) + "\n"
        + _json.dumps({"ckpt_sha": "def", "script": "probe.py", "ok": True}) + "\n"
    )

    seen = _seen_runs()
    assert ("abc", "probe.py") in seen
    assert ("def", "probe.py") in seen


def test_seen_runs_caches_after_first_load(isolated_project: Path, monkeypatch: pytest.MonkeyPatch):
    """`_seen_runs()` should cache after first call; subsequent calls must
    not re-open the file. Critical for daemons that poll every 5 minutes
    against a 100MB+ JSONL.
    """
    from crucible.runner import eval_watcher
    import json as _json

    log = isolated_project / ".crucible" / "eval_watch.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(_json.dumps({"ckpt_sha": "h", "script": "p.py", "ok": True}) + "\n")

    # First call populates the cache.
    seen1 = eval_watcher._seen_runs()
    assert ("h", "p.py") in seen1

    # Now force the file to a state that would FAIL if re-read (e.g., delete it).
    log.unlink()
    seen2 = eval_watcher._seen_runs()
    # Cached lookup still wins despite the file being gone.
    assert ("h", "p.py") in seen2


def test_append_log_keeps_cache_in_sync(isolated_project: Path):
    """After the cache is loaded, `_append_log` must update it incrementally
    so a subsequent `_seen_runs()` reflects the new row without re-reading."""
    from crucible.runner import eval_watcher

    # Prime the cache with an empty log.
    assert eval_watcher._seen_runs() == set()
    # Append a row through the public path — cache should pick it up.
    eval_watcher._append_log({"ckpt_sha": "abc", "script": "x.py", "ok": True})
    seen = eval_watcher._seen_runs()
    assert ("abc", "x.py") in seen


def test_seen_runs_does_not_age_out_old_entries(isolated_project: Path):
    """Regression: an earlier _read_log_tail(n=10000) cap silently dropped
    old (sha, script) pairs once the log grew past 10k rows. Now _seen_runs
    reads the full log and old entries stay tracked.
    """
    from crucible.runner.eval_watcher import _seen_runs
    import json as _json

    log = isolated_project / ".crucible" / "eval_watch.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # First entry is the one that would have aged out under the old 10k cap.
    rows.append(_json.dumps({"ckpt_sha": "old_hash", "script": "old_script.py", "ok": True}))
    # Pad with 10500 dummy rows to push past the prior tail window.
    for i in range(10500):
        rows.append(_json.dumps({"ckpt_sha": f"h{i}", "script": "filler.py", "ok": True}))
    log.write_text("\n".join(rows) + "\n")

    seen = _seen_runs()
    assert ("old_hash", "old_script.py") in seen, (
        "_seen_runs lost the oldest entry — backfill would re-run it"
    )
