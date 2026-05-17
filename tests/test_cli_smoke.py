"""CLI smoke tests — Part G promised these, never landed.

Covers the entry-point dispatch path that has zero unit-test coverage. Each
test invokes the real `crucible` CLI via subprocess so the argparse wiring,
TOOL_DISPATCH table, and config bootstrapping are all exercised — the kind
of seam that the Part G `tool_router` schema-skew bug lived in.

We don't try for deep behavior coverage here; that lives in the per-module
unit tests. Smoke tests assert: (1) the command runs, (2) exit code is 0,
(3) the output has the rough shape we expect. Network-bound subcommands
(hub_sync, hf_publish, runpod_*) are excluded — they need creds and live
endpoints.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _cli(*args: str, cwd: Path | None = None, env_extra: dict[str, str] | None = None,
         timeout: int = 30) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "crucible.cli.main", *args],
        cwd=str(cwd) if cwd else str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# A minimal-but-valid crucible.yaml — enough that load_config doesn't trip.
MINIMAL_YAML = """\
name: cli-smoke
version: "0.3.0"
provider:
  type: ssh
  ssh_key: ~/.ssh/id_ed25519
data:
  source: huggingface
  repo_id: test/ds
  local_root: ./data
  manifest: manifest.json
training:
  - backend: torch
    script: train.py
presets:
  smoke:
    MAX_WALLCLOCK_SECONDS: "60"
wandb:
  required: false
hf_collab:
  enabled: false
researcher:
  budget_hours: 0.1
  max_iterations: 1
  program_file: program.md
results_file: experiments.jsonl
fleet_results_file: experiments_fleet.jsonl
logs_dir: logs
nodes_file: nodes.json
"""


@pytest.fixture
def smoke_project(tmp_path: Path) -> Path:
    (tmp_path / "crucible.yaml").write_text(MINIMAL_YAML, encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "experiments.jsonl").touch()
    (tmp_path / "experiments_fleet.jsonl").touch()
    (tmp_path / "nodes.json").write_text("[]", encoding="utf-8")
    return tmp_path


class TestRootCli:
    def test_help(self):
        r = _cli("--help")
        assert r.returncode == 0
        assert "Crucible" in r.stdout
        assert "fleet" in r.stdout
        assert "research" in r.stdout
        assert "mcp" in r.stdout

    def test_version(self):
        r = _cli("--version")
        assert r.returncode == 0
        # Version string is printed; exact value isn't pinned here.
        assert r.stdout.strip()

    def test_no_args_shows_usage(self):
        r = _cli()
        # argparse prints usage; exit code may be 0 or 2 depending on argparse
        # version. We just want a non-crashing exit + usage line.
        assert r.returncode in (0, 2)
        text = r.stdout + r.stderr
        assert "usage:" in text.lower() or "crucible" in text


class TestSubcommandHelp:
    @pytest.mark.parametrize(
        "subcommand",
        [
            "fleet", "run", "analyze", "research", "data",
            "store", "hub", "tap", "track", "project", "note",
            "recipe", "serve", "mcp", "trace", "models", "notebook",
        ],
    )
    def test_help_doesnt_crash(self, subcommand: str):
        r = _cli(subcommand, "--help")
        assert r.returncode == 0, f"{subcommand} --help failed: {r.stderr}"
        assert "usage:" in r.stdout.lower()


class TestMcpCall:
    def test_help(self):
        r = _cli("mcp", "call", "--help")
        assert r.returncode == 0
        assert "tool_name" in r.stdout
        assert "--args" in r.stdout

    def test_tool_router_against_tmp_project(self, smoke_project: Path):
        r = _cli("mcp", "call", "tool_router", cwd=smoke_project)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        # Parse the JSON tail (stdout may have INFO log lines before the JSON).
        payload = _extract_json_tail(r.stdout)
        assert "recommended_tool" in payload
        assert "rationale" in payload
        assert "state" in payload

    def test_runs_search_empty_project(self, smoke_project: Path):
        r = _cli(
            "mcp", "call", "runs_search",
            "--args", '{"limit": 10}',
            cwd=smoke_project,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        payload = _extract_json_tail(r.stdout)
        # Empty project → empty rows list (not an error).
        assert "rows" in payload or "results" in payload or "runs" in payload

    def test_unknown_tool_fails_with_clear_message(self, smoke_project: Path):
        r = _cli(
            "mcp", "call", "this_tool_does_not_exist",
            cwd=smoke_project,
        )
        # Either exits non-zero or returns an error payload. Either is fine
        # — we just want the failure to surface clearly, not swallowed.
        text = (r.stdout + r.stderr).lower()
        assert r.returncode != 0 or "error" in text or "unknown" in text


class TestResearchCli:
    def test_run_help_lists_verbs(self):
        r = _cli("research", "run", "--help")
        assert r.returncode == 0
        # Phase 1 orchestrator-contract verbs.
        for verb in ("start", "submit", "status", "cancel"):
            assert verb in r.stdout.lower()

    def test_run_start_help(self):
        r = _cli("research", "run", "start", "--help")
        assert r.returncode == 0
        assert "iterations" in r.stdout.lower()
        assert "tier" in r.stdout.lower()
        assert "budget-usd" in r.stdout.lower()


class TestRecipeCli:
    def test_list_runs(self):
        # Run from a tmp dir so we don't read the user's real recipes.
        r = _cli("recipe", "list", cwd=Path("/tmp"))
        # Exit code 0 even when no recipes exist (prints "no recipes" or empty
        # table). Just assert it didn't crash.
        assert r.returncode == 0, f"stderr: {r.stderr}"


class TestModelsCli:
    def test_list_families(self):
        r = _cli("models", "--help")
        assert r.returncode == 0


class TestTrackCli:
    def test_help(self):
        r = _cli("track", "--help")
        assert r.returncode == 0
        assert "track" in r.stdout.lower()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _extract_json_tail(stdout: str) -> dict:
    """The CLI prints INFO log lines before the JSON payload. Parse the
    last JSON object in the stream (or raise with helpful context)."""
    text = stdout.strip()
    # Walk back from end to find the start of the trailing JSON object.
    if not text.endswith("}"):
        raise AssertionError(f"stdout does not end with JSON: {text[-200:]!r}")
    depth = 0
    start = None
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start is None:
        raise AssertionError(f"could not find JSON start: {text[:200]!r} ... {text[-200:]!r}")
    return json.loads(text[start:])
