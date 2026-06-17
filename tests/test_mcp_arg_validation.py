"""Hostile-args matrix for the new MCP tools shipped in Phase 5.1.

Every tool must:
- Return a structured ``{"error": ...}`` dict on bad / missing args.
- Never raise — that would kill the MCP JSON-RPC reply.
- Return a dict that ``json.dumps`` can serialise.

The matrix is intentionally repetitive — one assertion per failure
mode so a regression points at exactly the right tool/arg.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def fake_config(tmp_path, monkeypatch):
    """Stub `_get_config` to return a config rooted at tmp_path."""
    class _Cfg:
        project_root = tmp_path
    monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _Cfg())
    return _Cfg()


def _assert_error_dict(result, *substrings):
    assert isinstance(result, dict)
    assert "error" in result, f"expected error dict, got {result!r}"
    err = str(result["error"])
    for s in substrings:
        assert s in err, f"error string {err!r} missing {s!r}"
    # Must be JSON-serialisable for the MCP channel.
    json.dumps(result)


# ---------------------------------------------------------------------------
# code_mutation_propose
# ---------------------------------------------------------------------------


class TestCodeMutationProposeArgs:
    def test_no_target_file(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_propose"]({})
        _assert_error_dict(out, "target_file")

    def test_no_intent(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_propose"]({"target_file": "x.py"})
        _assert_error_dict(out, "intent")

    def test_nonexistent_target(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_propose"]({
            "target_file": "does_not_exist.py", "intent": "x",
        })
        _assert_error_dict(out, "not found")

    def test_target_with_scope(self, fake_config):
        (fake_config.project_root / "src").mkdir(parents=True, exist_ok=True)
        (fake_config.project_root / "src" / "x.py").write_text("pass\n")
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_propose"]({
            "target_file": "src/x.py", "intent": "lower bpb",
            "mutation_scope": ["src/"],
        })
        assert "system" in out and "user" in out and "schema" in out
        json.dumps(out)


# ---------------------------------------------------------------------------
# code_mutation_apply
# ---------------------------------------------------------------------------


class TestCodeMutationApplyArgs:
    def test_no_scorer(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({})
        _assert_error_dict(out, "scorer")

    def test_scorer_without_cmd(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({"scorer": {}})
        _assert_error_dict(out, "scorer.cmd")

    def test_scorer_cmd_wrong_type(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "scorer": {"cmd": "python3 scorer.py"},  # should be list
        })
        # Either rejected for cmd type or coerced — both acceptable;
        # the contract is "returns dict, no raise".
        assert isinstance(out, dict)
        json.dumps(out)

    def test_llm_diff_missing_response(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "policy": "llm_diff",
            "scorer": {"cmd": ["python3", "x.py"]},
        })
        _assert_error_dict(out, "llm_response")

    def test_llm_diff_missing_target_file(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "policy": "llm_diff",
            "scorer": {"cmd": ["python3", "x.py"]},
            "llm_response": {"diff": "x", "hypothesis": "h", "rationale": "r"},
        })
        _assert_error_dict(out, "target_file")

    def test_ast_local_edit_missing_proposal(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "policy": "ast_local_edit",
            "scorer": {"cmd": ["python3", "x.py"]},
        })
        _assert_error_dict(out, "proposal")

    def test_ast_local_edit_proposal_missing_name(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "policy": "ast_local_edit",
            "scorer": {"cmd": ["python3", "x.py"]},
            "proposal": {"target_file": "x.py", "diff": "{}"},
        })
        _assert_error_dict(out, "name")

    def test_unknown_policy(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "policy": "bogus",
            "scorer": {"cmd": ["python3", "x.py"]},
            "proposal": {"name": "x", "target_file": "x.py", "diff": "{}"},
        })
        _assert_error_dict(out)
        # bogus policy means build_code_mutation_policy raises.
        err = str(out["error"])
        assert "bogus" in err or "No code_mutation policy" in err

    def test_proposal_block_not_a_dict(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_apply"]({
            "policy": "ast_local_edit",
            "scorer": {"cmd": ["python3", "x.py"]},
            "proposal": "not a dict",
        })
        _assert_error_dict(out, "proposal")


# ---------------------------------------------------------------------------
# hypothesis_tournament_*
# ---------------------------------------------------------------------------


class TestTournamentCreateArgs:
    def test_no_name(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({})
        _assert_error_dict(out, "name")

    def test_no_hypotheses(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({"name": "x"})
        _assert_error_dict(out, "hypotheses")

    def test_hypotheses_not_a_list(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": "x", "hypotheses": "not a list",
        })
        _assert_error_dict(out, "hypotheses")

    def test_hypotheses_empty_list(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": "x", "hypotheses": [],
        })
        _assert_error_dict(out, "hypotheses")

    def test_traversal_in_name(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": "../escape", "hypotheses": [{"id": "h0"}],
        })
        _assert_error_dict(out, "invalid tournament name")


class TestTournamentPairArgs:
    def test_no_name(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_pair"]({})
        _assert_error_dict(out, "name")

    def test_nonexistent_tournament(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_pair"]({"name": "ghost"})
        _assert_error_dict(out, "no tournament")

    def test_unknown_policy(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": "x", "hypotheses": [{"id": "a"}, {"id": "b"}],
        })
        out = TOOL_DISPATCH["hypothesis_tournament_pair"]({
            "name": "x", "policy": "bogus",
        })
        _assert_error_dict(out, "unknown policy")


class TestTournamentSubmitArgs:
    def test_no_name(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_submit"]({})
        _assert_error_dict(out, "name")

    def test_no_winner(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_submit"]({
            "name": "x", "loser_id": "h1",
        })
        _assert_error_dict(out, "winner_id")

    def test_no_loser(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_submit"]({
            "name": "x", "winner_id": "h0",
        })
        _assert_error_dict(out, "loser_id")

    def test_nonexistent_tournament(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_submit"]({
            "name": "ghost", "winner_id": "h0", "loser_id": "h1",
        })
        _assert_error_dict(out, "no tournament")


class TestTournamentRankArgs:
    def test_no_name(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_rank"]({})
        _assert_error_dict(out, "name")

    def test_nonexistent(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_rank"]({"name": "ghost"})
        _assert_error_dict(out, "no tournament")


# ---------------------------------------------------------------------------
# hypothesis_cluster
# ---------------------------------------------------------------------------


class TestHypothesisClusterArgs:
    def test_hypotheses_not_a_list(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_cluster"]({"hypotheses": "not a list"})
        _assert_error_dict(out, "hypotheses")

    def test_empty_list_returns_zero_clusters(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_cluster"]({"hypotheses": []})
        assert out.get("count") == 0
        assert out.get("clusters") == []
        assert out.get("keepers") == []
        json.dumps(out)

    def test_unknown_backend(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        # backend=kmeans without sklearn → ProximityError → error dict
        out = TOOL_DISPATCH["hypothesis_cluster"]({
            "hypotheses": [{"id": "h1", "payload": {"x": "a"}}, {"id": "h2", "payload": {"x": "b"}}],
            "backend": "kmeans",
        })
        # Either sklearn installed (kmeans works) or absent (error). Both fine.
        assert isinstance(out, dict)
        json.dumps(out)


# ---------------------------------------------------------------------------
# research_meta_review
# ---------------------------------------------------------------------------


class TestResearchMetaReviewArgs:
    def test_no_track_name(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["research_meta_review"]({})
        _assert_error_dict(out, "track_name")

    def test_minimal_args_works(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["research_meta_review"]({"track_name": "alpha"})
        assert "system" in out and "user" in out and "schema" in out
        assert out["track_name"] == "alpha"
        json.dumps(out)

    def test_unknown_tournament_name_is_silently_ignored(self, fake_config):
        # Spec: meta_review with a tournament_name that doesn't exist
        # should just return zero tournament_entries — not error.
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["research_meta_review"]({
            "track_name": "alpha",
            "tournament_name": "no_such_tournament",
        })
        assert "system" in out
        assert out["sources_used"]["tournament_entries"] == 0


# ---------------------------------------------------------------------------
# code_mutation_list — should never need args
# ---------------------------------------------------------------------------


class TestCodeMutationListArgs:
    def test_empty_args_lists_policies(self, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["code_mutation_list"]({})
        assert out["count"] >= 4  # stub, code_mutation, ast_local_edit, llm_diff
        names = {p["name"] for p in out["policies"]}
        assert {"ast_local_edit", "llm_diff", "stub"} <= names
        json.dumps(out)
