"""Code mutation edge cases — sandbox + safety stress."""
from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path

import pytest

from crucible.researcher.code_mutation import (
    AstLocalEditPolicy,
    AstSafetyChecker,
    LlmDiffPolicy,
    MutationProposal,
    SandboxConfig,
    SandboxRunner,
    ScorerConfig,
    apply_unified_diff,
    check_workspace_confinement,
    llm_diff_request_prompt,
    parse_diff_targets,
    validate_scorer_cmd,
)


@pytest.fixture
def tiny_project(tmp_path):
    """Mirrors the fixture in test_code_mutation_phase5.py."""
    project = tmp_path / "tiny"
    (project / "src").mkdir(parents=True)
    (project / "src" / "model.py").write_text("OUTPUT = 0.5\n")
    (project / "run_scorer.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\n"
        "from model import OUTPUT\n"
        "print(f'val_bpb:{OUTPUT}')\n"
    )
    return project


# ---------------------------------------------------------------------------
# Diff parsing / scope edges
# ---------------------------------------------------------------------------


class TestDiffParseEdges:
    def test_empty_diff(self):
        assert parse_diff_targets("") == []

    def test_whitespace_diff(self):
        assert parse_diff_targets("   \n\n   ") == []

    def test_diff_with_no_hunk_headers(self):
        # Header lines only — no @@.
        diff = "--- a/x.py\n+++ b/x.py\n"
        # Targets still parsed even without hunks.
        assert parse_diff_targets(diff) == ["x.py"]

    def test_diff_with_backslash_path(self):
        # Backslash in path (Windows-style). On POSIX we don't try to
        # interpret it; just preserve so check_workspace_confinement
        # can reject.
        diff = "--- a/x\\..\\y.py\n+++ b/x\\..\\y.py\n@@ -1 +1 @@\n-x\n+y\n"
        targets = parse_diff_targets(diff)
        assert targets and "\\" in targets[0]

    def test_diff_with_unicode_target(self):
        diff = "--- a/résumé.py\n+++ b/résumé.py\n@@ -1 +1 @@\n-x\n+y\n"
        assert parse_diff_targets(diff) == ["résumé.py"]

    def test_diff_with_tabs_in_target(self):
        # Targets are space-terminated by the regex; a tab is whitespace too.
        diff = "--- a/x.py\t2026-01-01\n+++ b/x.py\t2026-01-02\n@@ -1 +1 @@\n-x\n+y\n"
        assert parse_diff_targets(diff) == ["x.py"]


class TestWorkspaceConfinement:
    def test_dotdot_segment_rejected(self, tmp_path):
        bad = check_workspace_confinement(["src/../etc/passwd"], tmp_path)
        assert bad == ["src/../etc/passwd"]

    def test_normal_path_accepted(self, tmp_path):
        assert check_workspace_confinement(["src/x.py", "y.py"], tmp_path) == []

    def test_symlink_escape_rejected(self, tmp_path):
        # Make a symlink inside tmp_path pointing at /etc and try to
        # resolve a path that goes through it.
        link = tmp_path / "outlink"
        link.symlink_to("/etc")
        bad = check_workspace_confinement(["outlink/passwd"], tmp_path)
        assert bad == ["outlink/passwd"]


# ---------------------------------------------------------------------------
# apply_unified_diff edges
# ---------------------------------------------------------------------------


class TestApplyDiffEdges:
    def test_empty_diff_rejected(self, tmp_path):
        from crucible.researcher.code_mutation import DiffApplyError
        with pytest.raises(DiffApplyError):
            apply_unified_diff(tmp_path, "")

    def test_garbage_diff_rejected(self, tmp_path):
        from crucible.researcher.code_mutation import DiffApplyError
        with pytest.raises(DiffApplyError):
            apply_unified_diff(tmp_path, "not a diff at all")

    def test_diff_against_missing_file(self, tmp_path):
        from crucible.researcher.code_mutation import DiffApplyError
        diff = (
            "--- a/missing.py\n+++ b/missing.py\n"
            "@@ -1 +1 @@\n-x\n+y\n"
        )
        with pytest.raises(DiffApplyError):
            apply_unified_diff(tmp_path, diff)


# ---------------------------------------------------------------------------
# Sandbox excludes + env propagation
# ---------------------------------------------------------------------------


class TestSandboxExcludes:
    def test_data_directory_excluded_from_clone(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / "src.py").write_text("x = 1\n")
        (project / "data").mkdir()
        (project / "data" / "big.bin").write_bytes(b"x" * 1000)
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sb")
        with sandbox.cloned_workspace() as ws:
            assert (ws / "src.py").exists()
            assert not (ws / "data" / "big.bin").exists()

    def test_pyc_excluded(self, tmp_path):
        project = tmp_path / "proj2"
        (project / "pkg").mkdir(parents=True)
        (project / "pkg" / "x.py").write_text("pass\n")
        (project / "pkg" / "x.pyc").write_bytes(b"\x00\x01")
        (project / "pkg" / "__pycache__").mkdir()
        (project / "pkg" / "__pycache__" / "y.pyc").write_bytes(b"\x00\x01")
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sb2")
        with sandbox.cloned_workspace() as ws:
            assert (ws / "pkg" / "x.py").exists()
            assert not (ws / "pkg" / "x.pyc").exists()
            assert not (ws / "pkg" / "__pycache__").exists()


class TestSandboxEnvIsolation:
    def test_wandb_key_does_not_leak_by_default(self, tmp_path, monkeypatch):
        # Set a fake WANDB key in the parent env. Default
        # inherit_env_keys does NOT include it.
        monkeypatch.setenv("WANDB_API_KEY", "sekret_wandb")
        project = tmp_path / "p"
        project.mkdir()
        (project / "leak.py").write_text(
            "import os\n"
            "print('WANDB_LEAK=' + str(os.environ.get('WANDB_API_KEY')))\n"
        )
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sb")
        with sandbox.cloned_workspace() as ws:
            res = sandbox.run(ws, ["python3", "leak.py"], SandboxConfig())
        # Default whitelist doesn't include WANDB_API_KEY so child sees None.
        assert "WANDB_LEAK=None" in res["stdout"], res["stdout"]
        assert "sekret_wandb" not in res["stdout"]

    def test_explicit_inherit_opt_in_does_leak(self, tmp_path, monkeypatch):
        # Operator who explicitly opts in to WANDB_API_KEY accepts that
        # the scorer sees it. This is the inverse of the previous test —
        # confirms the default exclusion is real protection, not vestigial.
        monkeypatch.setenv("WANDB_API_KEY", "sekret_wandb2")
        project = tmp_path / "p2"
        project.mkdir()
        (project / "leak.py").write_text(
            "import os\n"
            "print('WANDB_LEAK=' + str(os.environ.get('WANDB_API_KEY')))\n"
        )
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sb2")
        config = SandboxConfig(inherit_env_keys=("PATH", "WANDB_API_KEY"))
        with sandbox.cloned_workspace() as ws:
            res = sandbox.run(ws, ["python3", "leak.py"], config)
        assert "WANDB_LEAK=sekret_wandb2" in res["stdout"]

    def test_blackhole_proxy_set_when_network_disallowed(self, tmp_path):
        project = tmp_path / "p3"
        project.mkdir()
        (project / "dump.py").write_text(
            "import os\nfor k in ('HTTP_PROXY','HTTPS_PROXY','NO_PROXY'):\n"
            "    print(f'{k}={os.environ.get(k)}')\n"
        )
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sb3")
        config = SandboxConfig(allow_network=False)
        with sandbox.cloned_workspace() as ws:
            res = sandbox.run(ws, ["python3", "dump.py"], config)
        assert "HTTP_PROXY=http://127.0.0.1:1" in res["stdout"]
        assert "HTTPS_PROXY=http://127.0.0.1:1" in res["stdout"]


# ---------------------------------------------------------------------------
# Sandbox stability — concurrency + leak
# ---------------------------------------------------------------------------


class TestSandboxStability:
    def test_concurrent_clones_disjoint(self, tmp_path):
        """Two threads clone + run in parallel — no clobbering."""
        project = tmp_path / "p"
        project.mkdir()
        (project / "s.py").write_text("print('val_bpb:1.0')\n")
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sb")
        results = []
        lock = threading.Lock()

        def worker():
            with sandbox.cloned_workspace() as ws:
                res = sandbox.run(ws, ["python3", "s.py"], SandboxConfig(timeout_seconds=15))
                with lock:
                    results.append(res)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 4
        assert all(r["ok"] for r in results), [r for r in results if not r["ok"]]

    def test_no_fd_leak_over_many_runs(self, tmp_path):
        """50 sandbox runs in a loop — file descriptor count should
        not grow unboundedly. We allow up to +10 FD slack for caches."""
        project = tmp_path / "p"
        project.mkdir()
        (project / "s.py").write_text("print('val_bpb:1.0')\n")
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sb")

        def fd_count() -> int:
            # /proc/self/fd doesn't exist on macOS; fall back to lsof.
            proc_fd = Path("/proc/self/fd")
            if proc_fd.exists():
                return len(list(proc_fd.iterdir()))
            # macOS path — count open FDs for this pid via `lsof -p`.
            import subprocess
            try:
                out = subprocess.run(
                    ["lsof", "-p", str(os.getpid())],
                    capture_output=True, text=True, timeout=5,
                )
                return len(out.stdout.splitlines())
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return -1  # cannot measure; test becomes a smoke run

        gc.collect()
        before = fd_count()
        for _ in range(50):
            with sandbox.cloned_workspace() as ws:
                sandbox.run(ws, ["python3", "s.py"], SandboxConfig(timeout_seconds=10))
        gc.collect()
        after = fd_count()
        if before == -1 or after == -1:
            pytest.skip("cannot measure FDs on this platform")
        assert after - before < 15, (
            f"FD count grew {before} → {after} over 50 runs — likely leak"
        )

    def test_timeout_cleans_up_workspace(self, tmp_path):
        project = tmp_path / "p"
        project.mkdir()
        (project / "slow.py").write_text("import time; time.sleep(30)\n")
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sb")
        before_dirs = set(p.name for p in (tmp_path / "sb").iterdir() if (tmp_path / "sb").exists()) if (tmp_path / "sb").exists() else set()
        with sandbox.cloned_workspace() as ws:
            res = sandbox.run(ws, ["python3", "slow.py"], SandboxConfig(timeout_seconds=1))
            assert not res["ok"]
            assert "timeout" in res["stderr"]
        # After context exit, the clone dir is gone.
        if (tmp_path / "sb").exists():
            after_dirs = set(p.name for p in (tmp_path / "sb").iterdir())
            assert ws.name not in after_dirs


# ---------------------------------------------------------------------------
# llm_diff_request_prompt edges
# ---------------------------------------------------------------------------


class TestLlmDiffPromptEdges:
    def test_truncation_for_large_source(self, tiny_project):
        # Inflate the target file past the 16000-char truncation cap.
        target = tiny_project / "src" / "model.py"
        target.write_text("x = 1\n" + ("# pad\n" * 5000))
        env = llm_diff_request_prompt(
            target_file="src/model.py",
            intent="...",
            project_root=tiny_project,
        )
        assert "[NOTE: file truncated" in env["user"]

    def test_no_truncation_note_for_small_source(self, tiny_project):
        env = llm_diff_request_prompt(
            target_file="src/model.py", intent="...", project_root=tiny_project,
        )
        assert "[NOTE: file truncated" not in env["user"]

    def test_mutation_scope_appears_in_prompt(self, tiny_project):
        env = llm_diff_request_prompt(
            target_file="src/model.py", intent="...", project_root=tiny_project,
            mutation_scope=["src/"],
        )
        assert "mutation_scope" in env["user"]
        assert "src/" in env["user"]


# ---------------------------------------------------------------------------
# AST swappers — quirks
# ---------------------------------------------------------------------------


class TestAstSwappers:
    def test_swap_literal_strict_type_match(self, tiny_project):
        # swap_literal old=1 (int) MUST NOT replace True (which equals 1
        # in Python by value but is type bool).
        target = tiny_project / "src" / "model.py"
        target.write_text("a = True\nb = 1\n")
        policy = AstLocalEditPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["python3", "run_scorer.py"]),
        )
        diff = policy.generate_diff(MutationProposal(
            name="swap1", target_file="src/model.py",
            diff=json.dumps({"kind": "swap_literal", "old": 1, "new": 2}),
        ))
        # b = 1 → b = 2; a = True must NOT become a = 2. (a = True may
        # appear as a context line — what matters is the +/- pairs.)
        assert "+b = 2" in diff
        assert "-b = 1" in diff
        assert "+a = 2" not in diff  # never replaced True with 2
        assert "-a = True" not in diff  # True line was not removed

    def test_swap_attribute_only_matches_attribute_node(self, tiny_project):
        target = tiny_project / "src" / "model.py"
        target.write_text("import torch.nn as nn\nx = nn.GELU()\ny = 'GELU'\n")
        policy = AstLocalEditPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["python3", "run_scorer.py"]),
        )
        diff = policy.generate_diff(MutationProposal(
            name="swap_attr", target_file="src/model.py",
            diff=json.dumps({"kind": "swap_attribute", "old": "GELU", "new": "SiLU"}),
        ))
        # nn.GELU → nn.SiLU; the string literal 'GELU' must NOT change.
        assert "nn.SiLU()" in diff
        assert "'SiLU'" not in diff


# ---------------------------------------------------------------------------
# validate_scorer_cmd edges
# ---------------------------------------------------------------------------


class TestScorerCmdEdges:
    def test_python_with_version_suffix_accepted(self, tmp_path):
        (tmp_path / "x.py").write_text("pass\n")
        # python3.12 / python3.11 / python should all work via the
        # basename startswith("python3.") branch.
        assert validate_scorer_cmd(["python3.12", "x.py"], tmp_path) is None

    def test_absolute_python_binary_path(self, tmp_path):
        (tmp_path / "x.py").write_text("pass\n")
        # Full path /usr/bin/python3 → basename is python3, accepted.
        assert validate_scorer_cmd(["/usr/bin/python3", "x.py"], tmp_path) is None

    def test_empty_cmd_rejected(self, tmp_path):
        assert validate_scorer_cmd([], tmp_path) is not None

    def test_unknown_basename_accepted(self, tmp_path):
        # `make` is not in the deny list; not a Python interpreter; so
        # validate_scorer_cmd returns None (pass-through). The operator
        # is on the hook for what their custom scorer does.
        assert validate_scorer_cmd(["make", "test"], tmp_path) is None


# ---------------------------------------------------------------------------
# AST safety edges
# ---------------------------------------------------------------------------


class TestAstSafetyEdges:
    def test_empty_source_clean(self):
        report = AstSafetyChecker().check_source("", "ok.py")
        assert report.ok

    def test_from_import_star_does_not_crash(self):
        report = AstSafetyChecker().check_source("from os.path import *\n", "edge.py")
        # os.path is not in sensitive modules; the import should pass.
        # But `from os.path import *` references "os.path" in module
        # — check it's clean.
        # `os` IS in sensitive modules. Walk both — but our module check
        # uses node.module which is "os.path" not "os", so this should be clean.
        # That's a known limitation — flagged here as documentation.
        # The test asserts current behavior (whichever way it goes).
        # Either ok or flagged — both are valid outcomes.
        assert isinstance(report.problems, list)

    def test_lambda_in_decorator(self):
        # Decorators are calls — should not crash the walker.
        src = "import functools\n@functools.lru_cache(maxsize=128)\ndef f(): pass\n"
        report = AstSafetyChecker().check_source(src, "ok.py")
        assert isinstance(report.problems, list)  # walker survives

    def test_nested_function_with_dynamic_call_flagged(self):
        src = "def outer():\n    def inner():\n        return getattr(x, 'y')\n    return inner\n"
        report = AstSafetyChecker().check_source(src, "bad.py")
        assert any("dynamic-call" in p for p in report.problems)


# ---------------------------------------------------------------------------
# execute_mutation full-loop edges
# ---------------------------------------------------------------------------


class TestExecuteMutationEdges:
    def test_diff_with_no_targets_returns_error(self, tiny_project, tmp_path):
        sandbox = SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb")
        policy = LlmDiffPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["python3", "run_scorer.py"]),
            sandbox=sandbox,
        )
        # An LLM that returns malformed diff text — validate() catches.
        proposal = MutationProposal(
            name="empty", target_file="src/model.py",
            diff="just prose, no diff headers",
        )
        result = policy.apply(proposal)
        assert not result.success
        assert "no +++ targets" in (result.error or "") or "validate" in (result.error or "")

    def test_traversal_caught_with_no_scope(self, tiny_project, tmp_path):
        """With mutation_scope=None, scope check is skipped but
        check_workspace_confinement still rejects traversal — that's
        the C1 last-line defense."""
        sandbox = SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb")
        policy = LlmDiffPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["python3", "run_scorer.py"]),
            sandbox=sandbox,
            allow_non_py=True,
        )
        proposal = MutationProposal(
            name="traverse", target_file="src/model.py",
            diff=(
                "--- a/src/../etc/passwd\n"
                "+++ b/src/../etc/passwd\n"
                "@@ -1 +1 @@\n-x\n+y\n"
            ),
            mutation_scope=None,
        )
        result = policy.apply(proposal)
        assert not result.success
        assert "workspace-confinement" in (result.error or "") or "does not match" in (result.error or "")
