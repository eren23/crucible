"""Phase 5.1 code-mutation tests — real policies, sandbox, safety filter."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from crucible.researcher.code_mutation import (
    AstLocalEditPolicy,
    AstSafetyChecker,
    AstSafetyReport,
    CodeMutationError,
    DiffApplyError,
    LlmDiffPolicy,
    MutationProposal,
    MutationResult,
    SandboxConfig,
    SandboxRunner,
    ScorerConfig,
    apply_unified_diff,
    build_code_mutation_policy,
    check_scope,
    execute_mutation,
    llm_diff_parse_response,
    llm_diff_request_prompt,
    parse_diff_targets,
    score_stdout,
)


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


class TestParseDiffTargets:
    def test_extracts_b_prefixed_targets(self):
        diff = (
            "--- a/src/model.py\n"
            "+++ b/src/model.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        assert parse_diff_targets(diff) == ["src/model.py"]

    def test_skips_dev_null_deletions(self):
        diff = (
            "--- a/old.py\n"
            "+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n"
            "-x = 1\n"
        )
        assert parse_diff_targets(diff) == []

    def test_multifile_returns_all(self):
        diff = (
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
            "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n"
        )
        assert parse_diff_targets(diff) == ["a.py", "b.py"]


class TestCheckScope:
    def test_none_is_unrestricted(self):
        assert check_scope(["src/x.py", "etc/passwd"], None) == []

    def test_empty_denies_everything(self):
        assert check_scope(["src/x.py"], []) == ["src/x.py"]

    def test_prefix_match(self):
        assert check_scope(["src/x.py", "tests/y.py", "secrets.env"], ["src/", "tests/"]) == ["secrets.env"]

    def test_bare_prefix_does_not_match_sibling(self):
        # M2 fix: ["src"] should NOT match src_backup/ files.
        violations = check_scope(["src_backup/model.py"], ["src"])
        assert violations == ["src_backup/model.py"]

    def test_bare_prefix_matches_dir_contents(self):
        # M2 fix: ["src"] should match files under src/ exactly like ["src/"].
        assert check_scope(["src/model.py"], ["src"]) == []

    def test_single_file_scope_matches_exact(self):
        # M2 fix: ["baseline.py"] should match exact filename target.
        # This is the demo use case that the over-eager normalisation
        # initially broke.
        assert check_scope(["baseline.py"], ["baseline.py"]) == []
        # ...and still reject other targets.
        assert check_scope(["other.py"], ["baseline.py"]) == ["other.py"]


# ---------------------------------------------------------------------------
# AST safety filter
# ---------------------------------------------------------------------------


class TestAstSafetyChecker:
    def test_clean_source_passes(self):
        src = "def f(x):\n    return x + 1\n"
        report = AstSafetyChecker().check_source(src, "ok.py")
        assert report.ok is True

    def test_subprocess_call_blocked(self):
        src = "import subprocess\ndef f():\n    subprocess.run(['ls'])\n"
        report = AstSafetyChecker().check_source(src, "bad.py")
        assert not report.ok
        assert any("shell-escape" in p for p in report.problems)

    def test_network_call_blocked(self):
        src = "import requests\ndef f():\n    return requests.get('http://x')\n"
        report = AstSafetyChecker().check_source(src, "bad.py")
        assert not report.ok
        assert any("network" in p for p in report.problems)

    def test_dunder_attribute_blocked(self):
        src = "class A: pass\n_ = A.__bases__\n"
        report = AstSafetyChecker().check_source(src, "bad.py")
        assert any("dunder" in p for p in report.problems)

    def test_sensitive_module_import_blocked(self):
        src = "import crucible.core.redact\n"
        report = AstSafetyChecker().check_source(src, "bad.py")
        assert any("sensitive module" in p for p in report.problems)

    def test_allow_calls_overrides_deny(self):
        # M1: subprocess is now in the sensitive-module deny list, so
        # `import subprocess` is rejected at import time regardless of
        # allow_calls. The allowlist still works for non-sensitive
        # patterns: extra_deny then allow_calls for the same call.
        src = "import time\ntime.sleep(1)\n"
        checker = AstSafetyChecker(
            extra_deny_calls={("time", "sleep")},
            allow_calls={("time", "sleep")},
        )
        report = checker.check_source(src, "ok.py")
        assert report.ok

    def test_dynamic_call_builtins_blocked(self):
        # M1: getattr / __import__ / eval-family bypass the per-call
        # deny lookup by constructing calls dynamically. Reference of
        # the Name itself should be flagged.
        eval_src = "_ = " + "eval" + "('1+1')\n"
        for src in (
            "_ = getattr(object(), 'x')\n",
            eval_src,
            "_ = __import__('subprocess')\n",
        ):
            report = AstSafetyChecker().check_source(src, "bad.py")
            assert not report.ok, f"expected reject for: {src!r}"
            assert any("dynamic-call" in p for p in report.problems)

    def test_alias_import_of_sensitive_module_blocked(self):
        # M1: importing subprocess (under any alias) is now flagged
        # because subprocess is in _DEFAULT_SENSITIVE_MODULES.
        src = "import subprocess as _sp\n"
        report = AstSafetyChecker().check_source(src, "bad.py")
        assert not report.ok
        assert any("sensitive module" in p for p in report.problems)

    def test_syntax_error_reported(self):
        report = AstSafetyChecker().check_source("def f(:", "broken.py")
        assert not report.ok
        assert "syntax error" in report.problems[0]


# ---------------------------------------------------------------------------
# Diff application (git apply)
# ---------------------------------------------------------------------------


class TestApplyUnifiedDiff:
    def test_clean_apply(self, tmp_path):
        (tmp_path / "m.py").write_text("x = 1\n")
        diff = (
            "--- a/m.py\n"
            "+++ b/m.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        apply_unified_diff(tmp_path, diff)
        assert (tmp_path / "m.py").read_text() == "x = 2\n"

    def test_conflict_raises(self, tmp_path):
        (tmp_path / "m.py").write_text("totally different\n")
        diff = (
            "--- a/m.py\n"
            "+++ b/m.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        with pytest.raises(DiffApplyError):
            apply_unified_diff(tmp_path, diff)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoreStdout:
    def test_last_match_wins(self):
        out = "step:1/10 val_bpb:5.0\nstep:5/10 val_bpb:3.2\nstep:10/10 val_bpb:2.1\n"
        assert score_stdout(out, r"val_bpb:([0-9.]+)") == 2.1

    def test_no_match_returns_none(self):
        assert score_stdout("nothing here", r"val_bpb:([0-9.]+)") is None


# ---------------------------------------------------------------------------
# SandboxRunner — uses real subprocess, so keep minimal
# ---------------------------------------------------------------------------


class TestSandboxRunner:
    def test_clone_creates_isolated_copy(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "hello.py").write_text("print('hello')\n")
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sandbox")
        with sandbox.cloned_workspace() as ws:
            assert (ws / "hello.py").read_text() == "print('hello')\n"
            (ws / "hello.py").write_text("mutated\n")
            assert (ws / "hello.py").read_text() == "mutated\n"
        # original unchanged
        assert (project / "hello.py").read_text() == "print('hello')\n"

    def test_run_captures_stdout_and_returncode(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "s.py").write_text("print('val_bpb:2.5')\n")
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sandbox")
        with sandbox.cloned_workspace() as ws:
            res = sandbox.run(ws, ["python3", "s.py"], SandboxConfig())
        assert res["ok"]
        assert "val_bpb:2.5" in res["stdout"]

    def test_run_honors_timeout(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "slow.py").write_text("import time; time.sleep(10)\n")
        sandbox = SandboxRunner(project, sandbox_root=tmp_path / "sandbox")
        with sandbox.cloned_workspace() as ws:
            res = sandbox.run(ws, ["python3", "slow.py"], SandboxConfig(timeout_seconds=1))
        assert not res["ok"]
        assert "timeout" in res["stderr"]


# ---------------------------------------------------------------------------
# AstLocalEditPolicy — end-to-end on a tiny project
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_project(tmp_path):
    """A minimal project where a Python script prints a scorable value.

    Mutation target: src/model.py contains `OUTPUT = 0.5`. The
    AstLocalEditPolicy swap_literal mutation will change 0.5 → 0.2,
    and the scorer reads the printed value via the val_bpb pattern.
    """
    project = tmp_path / "tiny"
    (project / "src").mkdir(parents=True)
    (project / "src" / "model.py").write_text("OUTPUT = 0.5\n")
    (project / "run_scorer.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\n"
        "from model import OUTPUT\n"
        "print(f'val_bpb:{OUTPUT}')\n"
    )
    return project


class TestAstLocalEditPolicy:
    def test_swap_literal_end_to_end(self, tiny_project, tmp_path):
        scorer = ScorerConfig(cmd=["python3", "run_scorer.py"])
        policy = AstLocalEditPolicy(
            project_root=tiny_project,
            scorer=scorer,
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
            sandbox_config=SandboxConfig(timeout_seconds=30),
        )
        proposal = MutationProposal(
            name="swap_to_0.2",
            target_file="src/model.py",
            diff=json.dumps({"kind": "swap_literal", "old": 0.5, "new": 0.2}),
            hypothesis="lower OUTPUT improves score",
        )
        problems = policy.validate(proposal)
        assert problems == [], problems
        result = policy.apply(proposal)
        assert result.success, f"expected success, got error: {result.error}"
        assert result.score == pytest.approx(0.2)
        # original tree untouched
        assert (tiny_project / "src" / "model.py").read_text() == "OUTPUT = 0.5\n"

    def test_no_op_edit_is_rejected(self, tiny_project, tmp_path):
        scorer = ScorerConfig(cmd=["python3", "run_scorer.py"])
        policy = AstLocalEditPolicy(
            project_root=tiny_project,
            scorer=scorer,
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
        )
        proposal = MutationProposal(
            name="noop",
            target_file="src/model.py",
            diff=json.dumps({"kind": "swap_literal", "old": 999.0, "new": 1000.0}),
        )
        result = policy.apply(proposal)
        assert not result.success
        assert "no changes" in (result.error or "")

    def test_validate_rejects_non_py(self, tiny_project):
        policy = AstLocalEditPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["true"]),
        )
        proposal = MutationProposal(
            name="x",
            target_file="README.md",
            diff=json.dumps({"kind": "swap_identifier", "old": "a", "new": "b"}),
        )
        problems = policy.validate(proposal)
        assert any(".py files" in p for p in problems)

    def test_validate_rejects_missing_context(self):
        policy = AstLocalEditPolicy()
        proposal = MutationProposal(name="x", target_file="y.py", diff="{}")
        problems = policy.validate(proposal)
        assert any("project_root" in p for p in problems)


# ---------------------------------------------------------------------------
# LlmDiffPolicy — orchestrator contract
# ---------------------------------------------------------------------------


class TestLlmDiffContract:
    def test_request_prompt_envelope_shape(self, tiny_project):
        env = llm_diff_request_prompt(
            target_file="src/model.py",
            intent="lower OUTPUT to improve val_bpb",
            project_root=tiny_project,
            mutation_scope=["src/"],
        )
        assert {"system", "user", "schema", "target_file", "mutation_scope"} <= env.keys()
        assert "OUTPUT = 0.5" in env["user"]
        assert env["schema"]["required"] == ["diff", "hypothesis", "rationale"]

    def test_request_prompt_missing_file(self, tmp_path):
        with pytest.raises(CodeMutationError):
            llm_diff_request_prompt(
                target_file="nope.py",
                intent="x",
                project_root=tmp_path,
            )

    def test_parse_response_builds_proposal(self):
        diff = (
            "--- a/src/model.py\n"
            "+++ b/src/model.py\n"
            "@@ -1 +1 @@\n"
            "-OUTPUT = 0.5\n"
            "+OUTPUT = 0.2\n"
        )
        proposal = llm_diff_parse_response(
            {"diff": diff, "hypothesis": "lower OUTPUT", "rationale": "test", "name": "drop_out"},
            target_file="src/model.py",
            mutation_scope=["src/"],
        )
        assert proposal.name == "drop_out"
        assert proposal.target_file == "src/model.py"
        assert proposal.diff == diff

    def test_parse_response_rejects_scope_violation(self):
        # M6 now catches target_file mismatch *before* scope violation,
        # so a diff against a file outside target_file raises target-
        # mismatch. Test scope-violation via allow_multi_file=True path.
        diff = (
            "--- a/src/model.py\n+++ b/src/model.py\n@@ -1 +1 @@\n-x\n+y\n"
            "--- a/etc/passwd\n+++ b/etc/passwd\n@@ -1 +1 @@\n-x\n+y\n"
        )
        with pytest.raises(CodeMutationError, match="mutation_scope"):
            llm_diff_parse_response(
                {"diff": diff, "hypothesis": "h", "rationale": "r"},
                target_file="src/model.py",
                mutation_scope=["src/"],
                allow_multi_file=True,
            )

    def test_parse_response_rejects_absolute_path(self):
        # m2: absolute paths in diff headers are stripped at parse time,
        # so the diff appears empty (no surviving targets).
        diff = "--- a//etc/passwd\n+++ b//etc/passwd\n@@ -1 +1 @@\n-x\n+y\n"
        with pytest.raises(CodeMutationError, match="no \\+\\+\\+ targets"):
            llm_diff_parse_response(
                {"diff": diff, "hypothesis": "h", "rationale": "r"},
                target_file="src/model.py",
            )

    def test_parse_response_rejects_other_file(self):
        # M6: single-file diff that touches a different relative file fails.
        diff = "--- a/src/other.py\n+++ b/src/other.py\n@@ -1 +1 @@\n-x\n+y\n"
        with pytest.raises(CodeMutationError, match="does not match"):
            llm_diff_parse_response(
                {"diff": diff, "hypothesis": "h", "rationale": "r"},
                target_file="src/model.py",
            )

    def test_parse_response_rejects_multi_file_by_default(self):
        diff = (
            "--- a/src/model.py\n+++ b/src/model.py\n@@ -1 +1 @@\n-x\n+y\n"
            "--- a/src/other.py\n+++ b/src/other.py\n@@ -1 +1 @@\n-x\n+y\n"
        )
        with pytest.raises(CodeMutationError, match="touches 2 files"):
            llm_diff_parse_response(
                {"diff": diff, "hypothesis": "h", "rationale": "r"},
                target_file="src/model.py",
            )

    def test_parse_response_rejects_missing_field(self):
        with pytest.raises(CodeMutationError, match="missing required"):
            llm_diff_parse_response(
                {"diff": "x", "hypothesis": "h"},
                target_file="src/model.py",
            )


class TestLlmDiffPolicy:
    def test_end_to_end(self, tiny_project, tmp_path):
        scorer = ScorerConfig(cmd=["python3", "run_scorer.py"])
        policy = LlmDiffPolicy(
            project_root=tiny_project,
            scorer=scorer,
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
            sandbox_config=SandboxConfig(timeout_seconds=30),
        )
        diff = (
            "--- a/src/model.py\n"
            "+++ b/src/model.py\n"
            "@@ -1 +1 @@\n"
            "-OUTPUT = 0.5\n"
            "+OUTPUT = 0.3\n"
        )
        proposal = MutationProposal(
            name="llm_lower",
            target_file="src/model.py",
            diff=diff,
            hypothesis="lower OUTPUT improves score",
            rationale="test",
            mutation_scope=["src/"],
        )
        result = policy.apply(proposal)
        assert result.success, f"err: {result.error}"
        assert result.score == pytest.approx(0.3)

    def test_safety_filter_rejects_subprocess(self, tiny_project, tmp_path):
        # Diff that injects a subprocess call into the target file.
        # AST safety filter should reject before sandbox runs.
        scorer = ScorerConfig(cmd=["python3", "run_scorer.py"])
        policy = LlmDiffPolicy(
            project_root=tiny_project,
            scorer=scorer,
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
            sandbox_config=SandboxConfig(timeout_seconds=30),
        )
        diff = (
            "--- a/src/model.py\n"
            "+++ b/src/model.py\n"
            "@@ -1 +1,3 @@\n"
            "-OUTPUT = 0.5\n"
            "+import subprocess\n"
            "+subprocess.run(['ls'])\n"
            "+OUTPUT = 0.5\n"
        )
        proposal = MutationProposal(
            name="injection",
            target_file="src/model.py",
            diff=diff,
            mutation_scope=["src/"],
        )
        result = policy.apply(proposal)
        assert not result.success
        assert "safety filter" in (result.error or "")


# ---------------------------------------------------------------------------
# build_code_mutation_policy
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_builds_ast_local_edit_with_context(self, tiny_project):
        policy = build_code_mutation_policy(
            "ast_local_edit",
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["true"]),
        )
        assert isinstance(policy, AstLocalEditPolicy)
        assert policy.project_root == tiny_project.resolve()
        assert policy.scorer is not None

    def test_builds_llm_diff_with_context(self, tiny_project):
        policy = build_code_mutation_policy(
            "llm_diff",
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["true"]),
        )
        assert isinstance(policy, LlmDiffPolicy)

    def test_unknown_policy_raises(self, tiny_project):
        with pytest.raises(CodeMutationError):
            build_code_mutation_policy(
                "does_not_exist",
                project_root=tiny_project,
                scorer=ScorerConfig(cmd=["true"]),
            )

    def test_stub_fallback_for_kwarg_ignoring_policy(self, tiny_project):
        # StubCodeMutationPolicy has no kwargs in __init__; builder
        # should warn and fall back to bare constructor.
        policy = build_code_mutation_policy(
            "stub",
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["true"]),
        )
        # Stub doesn't raise on construction.
        from crucible.researcher.code_mutation import StubCodeMutationPolicy
        assert isinstance(policy, StubCodeMutationPolicy)


# ---------------------------------------------------------------------------
# MCP tool surface
# ---------------------------------------------------------------------------


class TestMcpCodeMutationTools:
    def test_list_reflects_phase5_policies(self, monkeypatch, tmp_path):
        from crucible.core.errors import CrucibleError
        def boom():
            raise CrucibleError("no project")
        monkeypatch.setattr("crucible.mcp.tools._get_config", boom)
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_list"]({})
        names = {p.get("name") for p in out["policies"]}
        assert {"ast_local_edit", "llm_diff", "code_mutation", "stub"} <= names
        assert "Phase 5.1" in out["note"]

    def test_propose_returns_envelope(self, tiny_project, monkeypatch):
        # Stub _get_config to point at tiny_project.
        class _FakeConfig:
            project_root = tiny_project
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_propose"]({
            "target_file": "src/model.py",
            "intent": "lower OUTPUT",
            "mutation_scope": ["src/"],
        })
        assert "system" in out and "user" in out and "schema" in out
        assert "OUTPUT = 0.5" in out["user"]

    def test_propose_rejects_missing_args(self, monkeypatch):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_propose"]({"intent": "x"})
        assert "error" in out and "target_file" in out["error"]

    def test_apply_llm_diff_end_to_end(self, tiny_project, tmp_path, monkeypatch):
        class _FakeConfig:
            project_root = tiny_project
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH
        diff = (
            "--- a/src/model.py\n"
            "+++ b/src/model.py\n"
            "@@ -1 +1 @@\n"
            "-OUTPUT = 0.5\n"
            "+OUTPUT = 0.42\n"
        )
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "policy": "llm_diff",
            "target_file": "src/model.py",
            "mutation_scope": ["src/"],
            "llm_response": {
                "diff": diff,
                "hypothesis": "lower output",
                "rationale": "test",
                "name": "drop42",
            },
            "scorer": {"cmd": ["python3", "run_scorer.py"]},
            "sandbox_timeout": 30,
        })
        assert out["success"], out.get("error")
        assert out["score"] == pytest.approx(0.42)

    def test_apply_ast_local_edit(self, tiny_project, monkeypatch):
        class _FakeConfig:
            project_root = tiny_project
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "policy": "ast_local_edit",
            "proposal": {
                "name": "ast_swap",
                "target_file": "src/model.py",
                "diff": json.dumps({"kind": "swap_literal", "old": 0.5, "new": 0.7}),
            },
            "scorer": {"cmd": ["python3", "run_scorer.py"]},
            "sandbox_timeout": 30,
        })
        assert out["success"], out.get("error")
        assert out["score"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Tree integration
# ---------------------------------------------------------------------------


class TestCodeMutationTreeBridge:
    def test_expand_and_record(self, tiny_project, tmp_path):
        from crucible.researcher.code_mutation_tree import expand_tree_with_mutations
        from crucible.researcher.search_tree import SearchTree

        tree_dir = tmp_path / "tree"
        tree = SearchTree.create(
            tree_dir=tree_dir,
            name="mut_test",
            roots=[{"name": "root", "config": {}}],
            primary_metric="val_bpb",
            metric_direction="minimize",
            max_expansions_per_node=5,
        )
        root_id = next(iter(tree.nodes))

        scorer = ScorerConfig(cmd=["python3", "run_scorer.py"])
        policy = AstLocalEditPolicy(
            project_root=tiny_project,
            scorer=scorer,
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
            sandbox_config=SandboxConfig(timeout_seconds=30),
        )
        proposals = [
            MutationProposal(
                name=f"swap_{v}",
                target_file="src/model.py",
                diff=json.dumps({"kind": "swap_literal", "old": 0.5, "new": v}),
                hypothesis=f"try {v}",
            )
            for v in (0.3, 0.7)
        ]
        summaries = expand_tree_with_mutations(tree, root_id, proposals, policy)
        assert len(summaries) == 2
        assert all(s.success for s in summaries), [s.error for s in summaries]
        scores = sorted(s.score for s in summaries)
        assert scores == pytest.approx([0.3, 0.7])

        # Tree nodes recorded with the score under the primary metric.
        for s in summaries:
            node = tree.get_node(s.node_id)
            assert node["status"] == "completed"
            assert node["generation_method"] == "code_mutation"
            assert node["result"]["val_bpb"] == pytest.approx(s.score)


# ---------------------------------------------------------------------------
# Regression tests for security/correctness fixes
# ---------------------------------------------------------------------------


class TestPathTraversalDefense:
    """C1 fix: diff targets with `..` segments must not escape workspace."""

    def test_dotdot_in_target_caught(self, tiny_project, tmp_path):
        from crucible.researcher.code_mutation import (
            LlmDiffPolicy,
            MutationProposal,
            SandboxConfig,
            SandboxRunner,
            ScorerConfig,
        )
        policy = LlmDiffPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["python3", "run_scorer.py"]),
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
            sandbox_config=SandboxConfig(timeout_seconds=30),
        )
        diff = (
            "--- a/src/../../../etc/passwd\n"
            "+++ b/src/../../../etc/passwd\n"
            "@@ -1 +1 @@\n-x\n+y\n"
        )
        proposal = MutationProposal(
            name="traverse", target_file="src/model.py",
            diff=diff,  # multi-file would also fail M6, set scope=None
            mutation_scope=None,
        )
        # LlmDiffPolicy.validate now also rejects non-.py — workaround
        # via allow_non_py=True so we reach the workspace-confinement
        # check at apply time (which is what C1 is about).
        policy.allow_non_py = True
        result = policy.apply(proposal)
        assert not result.success
        assert (
            "workspace-confinement" in (result.error or "")
            or "mutation_scope" in (result.error or "")
        )


class TestScorerCmdAllowlist:
    """C2 fix: scorer.cmd basename refuses shell/env/network tools."""

    def test_env_basename_rejected(self, tmp_path):
        from crucible.researcher.code_mutation import (
            SandboxRunner,
            ScorerConfig,
            validate_scorer_cmd,
        )
        problem = validate_scorer_cmd(["env"], tmp_path)
        assert problem is not None and "env" in problem

    def test_shell_basename_rejected(self, tmp_path):
        from crucible.researcher.code_mutation import validate_scorer_cmd
        for shell in (["sh", "-c", "echo $HOME"], ["bash", "-c", "..."], ["zsh"]):
            problem = validate_scorer_cmd(shell, tmp_path)
            assert problem is not None, f"{shell} should be rejected"

    def test_curl_basename_rejected(self, tmp_path):
        from crucible.researcher.code_mutation import validate_scorer_cmd
        for net in (["curl", "http://x"], ["wget", "http://x"], ["nc", "x", "1"]):
            problem = validate_scorer_cmd(net, tmp_path)
            assert problem is not None

    def test_python_bare_rejected(self, tmp_path):
        from crucible.researcher.code_mutation import validate_scorer_cmd
        assert validate_scorer_cmd(["python3"], tmp_path) is not None
        assert "no script arg" in validate_scorer_cmd(["python3"], tmp_path)

    def test_python_dash_c_rejected(self, tmp_path):
        from crucible.researcher.code_mutation import validate_scorer_cmd
        problem = validate_scorer_cmd(["python3", "-c", "print(1)"], tmp_path)
        assert problem is not None and "flag" in problem

    def test_python_with_relative_py_script_allowed(self, tmp_path):
        from crucible.researcher.code_mutation import validate_scorer_cmd
        (tmp_path / "run.py").write_text("print('ok')\n")
        assert validate_scorer_cmd(["python3", "run.py"], tmp_path) is None

    def test_python_with_escaping_script_rejected(self, tmp_path):
        from crucible.researcher.code_mutation import validate_scorer_cmd
        problem = validate_scorer_cmd(["python3", "/etc/passwd"], tmp_path)
        assert problem is not None and "workspace" in problem.lower()

    def test_default_inherit_env_does_not_include_home(self):
        from crucible.researcher.code_mutation import _DEFAULT_INHERIT_ENV_KEYS
        assert "HOME" not in _DEFAULT_INHERIT_ENV_KEYS
        assert "WANDB_API_KEY" not in _DEFAULT_INHERIT_ENV_KEYS
        assert "RUNPOD_API_KEY" not in _DEFAULT_INHERIT_ENV_KEYS
        assert "HF_TOKEN" not in _DEFAULT_INHERIT_ENV_KEYS


class TestNonPyMutationGate:
    """M7 fix: LlmDiffPolicy refuses non-.py target files by default."""

    def test_makefile_rejected_by_default(self, tiny_project, tmp_path):
        (tiny_project / "Makefile").write_text("all:\n\techo hi\n")
        from crucible.researcher.code_mutation import (
            LlmDiffPolicy, MutationProposal, SandboxConfig, SandboxRunner, ScorerConfig,
        )
        policy = LlmDiffPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["python3", "run_scorer.py"]),
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
        )
        diff = "--- a/Makefile\n+++ b/Makefile\n@@ -1,2 +1,2 @@\n all:\n-\techo hi\n+\techo bye\n"
        proposal = MutationProposal(
            name="m", target_file="Makefile", diff=diff,
            mutation_scope=None,
        )
        result = policy.apply(proposal)
        assert not result.success
        assert "non-.py" in (result.error or "")

    def test_makefile_allowed_with_opt_in(self, tiny_project, tmp_path):
        (tiny_project / "Makefile").write_text("all:\n\techo hi\n")
        from crucible.researcher.code_mutation import (
            LlmDiffPolicy, MutationProposal, SandboxConfig, SandboxRunner, ScorerConfig,
        )
        policy = LlmDiffPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["python3", "run_scorer.py"]),
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
            allow_non_py=True,
        )
        diff = "--- a/Makefile\n+++ b/Makefile\n@@ -1,2 +1,2 @@\n all:\n-\techo hi\n+\techo bye\n"
        proposal = MutationProposal(
            name="m", target_file="Makefile", diff=diff,
            mutation_scope=None,
        )
        result = policy.apply(proposal)
        # Apply may still fail (scorer doesn't care about Makefile change)
        # but the validate step should pass (no non-.py rejection).
        assert "non-.py" not in (result.error or "")


class TestSandboxAssertNotInvokedOnSafetyReject:
    """m1 fix: when AST safety rejects, the scorer must not run."""

    def test_no_returncode_on_safety_reject(self, tiny_project, tmp_path):
        from crucible.researcher.code_mutation import (
            LlmDiffPolicy, MutationProposal, SandboxConfig, SandboxRunner, ScorerConfig,
        )
        policy = LlmDiffPolicy(
            project_root=tiny_project,
            scorer=ScorerConfig(cmd=["python3", "run_scorer.py"]),
            sandbox=SandboxRunner(tiny_project, sandbox_root=tmp_path / "sb"),
            sandbox_config=SandboxConfig(timeout_seconds=30),
        )
        diff = (
            "--- a/src/model.py\n+++ b/src/model.py\n"
            "@@ -1 +1,3 @@\n"
            "-OUTPUT = 0.5\n"
            "+import subprocess\n"
            "+subprocess.run(['ls'])\n"
            "+OUTPUT = 0.5\n"
        )
        proposal = MutationProposal(
            name="injection", target_file="src/model.py",
            diff=diff, mutation_scope=["src/"],
        )
        result = policy.apply(proposal)
        assert not result.success
        # Mechanism check: error was the safety filter, not a scorer
        # crash; artifacts must not contain returncode/stdout (sandbox
        # never ran).
        assert "safety filter" in (result.error or "")
        assert "returncode" not in result.artifacts
        assert "stdout_tail" not in result.artifacts
