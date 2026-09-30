"""AxSandboxRunner against a fake ax-run: arguments, env filtering, log and exit parsing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from crucible.researcher.code_mutation import (
    AxSandboxRunner,
    SandboxConfig,
    SandboxError,
    SandboxRunner,
    make_sandbox,
)

FAKE_AX_RUN = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
out = Path(os.environ["AX_RUN_OUT"])
(out / "call.json").write_text(json.dumps({"argv": sys.argv[1:], "timeout": os.environ["AX_RUN_TIMEOUT"],
                                           "git": Path(sys.argv[1], ".git").is_dir()}))
(out / "axr-x.log").write_text("step:1/1 val_loss:1.0 val_bpb:0.25\\n")
print("ax-run: axr-x finished, exit code " + os.environ.get("FAKE_EXIT", "0"))
sys.exit(int(os.environ.get("FAKE_EXIT", "0")) and 1)
"""


def _runner(tmp_path: Path) -> tuple[AxSandboxRunner, Path]:
    fake = tmp_path / "ax-run"
    fake.write_text(FAKE_AX_RUN.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    fake.chmod(0o755)
    record = tmp_path / "record"
    record.mkdir()
    # The fake writes into AX_RUN_OUT, a temp dir the runner deletes; copy the call out first.
    wrapper = tmp_path / "ax-run-wrap"
    wrapper.write_text(f'#!/bin/sh\n"{fake}" "$@"; rc=$?; cp "$AX_RUN_OUT/call.json" "{record}/"; exit $rc\n')
    wrapper.chmod(0o755)
    return AxSandboxRunner(tmp_path, ax_run=str(wrapper)), record


def test_run_passes_no_egress_and_filters_host_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "v")
    runner, record = _runner(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = SandboxConfig(timeout_seconds=42, inherit_env_keys=("PATH", "PYTHONPATH", "MY_KEY", "UNSET_KEY"),
                           cwd_subdir="sub dir")
    result = runner.run(workspace, ["python3", "score.py", "--x", "a b"], config)

    assert result["ok"] is True and result["returncode"] == 0
    assert "val_bpb:0.25" in result["stdout"]
    call = json.loads((record / "call.json").read_text())
    argv = call["argv"]
    assert argv[0] == str(workspace) and argv[2] == "--exec"
    assert "cd 'sub dir' && exec python3 score.py --x 'a b'" in argv[1]
    assert "--no-egress" in argv
    assert argv[argv.index("--env") + 1] == "MY_KEY" and argv.count("--env") == 1
    assert call["timeout"] == "42" and call["git"] is True


def test_run_reports_command_failure_and_network_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    runner, record = _runner(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = runner.run(workspace, ["python3", "score.py"], SandboxConfig(allow_network=True))
    assert result["ok"] is False and result["returncode"] == 3
    assert "--no-egress" not in json.loads((record / "call.json").read_text())["argv"]


def test_make_sandbox_selects_by_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_SANDBOX", raising=False)
    assert type(make_sandbox(tmp_path)) is SandboxRunner
    monkeypatch.setenv("CRUCIBLE_SANDBOX", "ax")
    fake, _ = _runner(tmp_path)
    runner = make_sandbox(tmp_path, ax_run=fake.ax_run)
    assert isinstance(runner, AxSandboxRunner)
    assert ".env" in runner.rsync_excludes


def test_missing_ax_run_fails_loudly_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_SANDBOX", "ax")
    with pytest.raises(SandboxError, match="Install ax-lab"):
        make_sandbox(tmp_path, ax_run=str(tmp_path / "no-such-ax-run"))
