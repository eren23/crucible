"""Regression tests: harness_runner output satisfies Crucible OutputParser.

These tests execute the tap's harness_runner.py in a subprocess against a
small inline MemorySystem and assert that the captured stdout parses as
``completed`` with all domain metrics populated.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from crucible.runner.output_parser import OutputParser


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / ".crucible" / "taps" / "meta-harness" / "launchers" / "harness_runner" / "harness_runner.py"


CANDIDATE_SOURCE = """
class MemorySystem:
    def predict(self, example):
        return example.get("target", 0)
    def learn_from_batch(self, batch):
        pass
"""


@pytest.mark.skipif(not RUNNER_PATH.exists(), reason="meta-harness tap not present")
def test_runner_output_parses_as_completed(tmp_path: Path) -> None:
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    (candidates_dir / "cand1.py").write_text(CANDIDATE_SOURCE)

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--candidate", "cand1",
            "--candidates-dir", str(candidates_dir),
            "--class-name", "MemorySystem",
            "--steps", "1",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    parsed = OutputParser().parse(result.stdout)
    assert parsed is not None
    assert parsed["status"] == "completed", f"stdout: {result.stdout}"
    res = parsed["result"]
    assert "val_bpb" in res
    assert "accuracy" in res
    assert "tokens_per_example" in res
    assert parsed["model_bytes"] == 0
