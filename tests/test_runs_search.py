"""Tests for runs_search MCP — Phase 2.5.

Covers the predicate parser (AST whitelist), dotted-path resolution,
config-namespace folding, sort/limit/select behavior, and the safety
gates that keep an attacker-supplied expression from executing
arbitrary Python.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from crucible.runner.search import (
    SearchError,
    compile_predicate,
    search_runs,
)

# ---------------------------------------------------------------------------
# Predicate parser
# ---------------------------------------------------------------------------


class TestCompilePredicate:
    def test_empty_predicate_matches_all(self):
        pred = compile_predicate("")
        assert pred({"val_loss": 1.0}) is True
        assert pred({}) is True

    def test_simple_lt(self):
        pred = compile_predicate("val_loss < 2.0")
        assert pred({"val_loss": 1.0}) is True
        assert pred({"val_loss": 3.0}) is False
        # Missing field → None → comparison is False, doesn't raise.
        assert pred({}) is False

    def test_and_or_not(self):
        pred = compile_predicate(
            "val_loss < 2.0 and (status == 'completed' or name == 'smoke')"
        )
        assert pred({"val_loss": 1.0, "status": "completed", "name": "x"}) is True
        assert pred({"val_loss": 1.0, "status": "failed", "name": "smoke"}) is True
        assert pred({"val_loss": 3.0, "status": "completed", "name": "x"}) is False

    def test_dotted_access(self):
        pred = compile_predicate("result.val_loss < 2.0")
        assert pred({"result": {"val_loss": 1.0}}) is True
        assert pred({"result": {"val_loss": 3.0}}) is False

    def test_config_namespace_fold(self):
        """Top-level config keys are reachable without the prefix."""
        pred = compile_predicate("model_dim > 128")
        row = {"config": {"model_dim": 256}}
        assert pred(row) is True
        row2 = {"config": {"model_dim": 64}}
        assert pred(row2) is False

    def test_config_uppercase_fallback(self):
        """Env-var style (MODEL_DIM) keys match lowercase identifiers."""
        pred = compile_predicate("model_family == 'baseline'")
        assert pred({"config": {"MODEL_FAMILY": "baseline"}}) is True
        assert pred({"config": {"MODEL_FAMILY": "other"}}) is False

    def test_in_and_not_in(self):
        pred = compile_predicate("status in ['completed', 'finished']")
        assert pred({"status": "completed"}) is True
        assert pred({"status": "failed"}) is False
        not_pred = compile_predicate("status not in ['failed', 'error']")
        assert not_pred({"status": "completed"}) is True

    def test_chained_comparison(self):
        pred = compile_predicate("0 < val_loss < 2.0")
        assert pred({"val_loss": 1.0}) is True
        assert pred({"val_loss": 0.0}) is False
        assert pred({"val_loss": 3.0}) is False

    def test_null_safe_comparison(self):
        """Comparing a missing field never raises; just returns False."""
        pred = compile_predicate("val_loss < 2.0")
        assert pred({}) is False
        assert pred({"val_loss": None}) is False

    def test_string_compare_numeric_coercion(self):
        """Env-var values are strings; the predicate engine coerces both."""
        pred = compile_predicate("model_dim > 100")
        assert pred({"config": {"model_dim": "256"}}) is True
        assert pred({"config": {"model_dim": "64"}}) is False


# ---------------------------------------------------------------------------
# Safety: parser whitelist
# ---------------------------------------------------------------------------


class TestParserSafety:
    """The predicate parser must reject anything outside its whitelist."""

    def test_rejects_function_call(self):
        with pytest.raises(SearchError, match="Unsupported"):
            compile_predicate("len(name) > 5")

    def test_rejects_arithmetic(self):
        # We don't allow BinOp in the whitelist — adding two columns is
        # out of scope for a search predicate.
        with pytest.raises(SearchError):
            compile_predicate("val_loss + 1 < 2")

    def test_rejects_subscript(self):
        with pytest.raises(SearchError):
            compile_predicate("name[0] == 'x'")

    def test_rejects_assignment(self):
        with pytest.raises(SearchError):
            compile_predicate("val_loss = 1")

    def test_rejects_double_underscore_attr(self):
        """An attribute chain still has to resolve via _resolve, which
        only reads dict keys. __class__/__getattribute__ access on a
        dict value yields None, not the class — defense in depth."""
        pred = compile_predicate("val_loss.__class__ == 'X'")
        # Resolves to None because dict doesn't have __class__ as a key.
        assert pred({"val_loss": 1.0}) is False

    def test_syntax_error_surfaces_typed_exception(self):
        with pytest.raises(SearchError, match="syntax error"):
            compile_predicate("val_loss < <")


# ---------------------------------------------------------------------------
# search_runs end-to-end
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, project_root: Path):
        self.project_root = project_root


def _write_results(tmp_path: Path, rows: list[dict[str, Any]]) -> None:
    import json
    path = tmp_path / "experiments.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def project_with_runs(tmp_path: Path, monkeypatch):
    rows = [
        {"id": "r1", "name": "smoke_a", "status": "completed",
         "config": {"model_dim": "256", "LR": "0.001"},
         "result": {"val_loss": 1.2, "steps_completed": 500},
         "backend": "torch", "model_bytes": 15000000},
        {"id": "r2", "name": "smoke_b", "status": "completed",
         "config": {"model_dim": "512", "LR": "0.001"},
         "result": {"val_loss": 0.9, "steps_completed": 500},
         "backend": "torch", "model_bytes": 18000000},
        {"id": "r3", "name": "failed_c", "status": "failed",
         "config": {"model_dim": "128", "LR": "0.01"},
         "result": None, "backend": "torch", "model_bytes": 0},
        {"id": "r4", "name": "smoke_d", "status": "completed",
         "config": {"model_dim": "256", "LR": "0.0005"},
         "result": {"val_loss": 1.5, "steps_completed": 500},
         "backend": "torch", "model_bytes": 14000000},
    ]
    _write_results(tmp_path, rows)
    monkeypatch.chdir(tmp_path)

    # Use a minimally-faked config to bypass the full yaml loader.
    from crucible.analysis import results as r_module

    def _fake_load(cfg=None, *, source="local"):
        if source != "local":
            return []
        import json
        return [json.loads(l) for l in (tmp_path / "experiments.jsonl").read_text().splitlines() if l]

    def _fake_merged(cfg=None):
        return _fake_load(cfg)

    monkeypatch.setattr(r_module, "load_results", _fake_load)
    monkeypatch.setattr(r_module, "merged_results", _fake_merged)
    return _FakeConfig(tmp_path)


class TestSearchRuns:
    def test_no_filter_returns_all(self, project_with_runs):
        out = search_runs(project_with_runs)
        assert out["matched"] == 4
        assert out["returned"] == 4

    def test_predicate_filter(self, project_with_runs):
        out = search_runs(
            project_with_runs,
            where="result.val_loss < 1.3 and status == 'completed'",
        )
        # r1 (val_loss=1.2) and r2 (val_loss=0.9) qualify.
        assert out["matched"] == 2
        names = [r["name"] for r in out["rows"]]
        assert set(names) == {"smoke_a", "smoke_b"}

    def test_order_by_ascending(self, project_with_runs):
        out = search_runs(
            project_with_runs,
            where="status == 'completed'",
            order_by="result.val_loss",
            direction="asc",
        )
        names = [r["name"] for r in out["rows"]]
        assert names == ["smoke_b", "smoke_a", "smoke_d"]

    def test_order_by_descending(self, project_with_runs):
        out = search_runs(
            project_with_runs,
            where="status == 'completed'",
            order_by="result.val_loss",
            direction="desc",
        )
        names = [r["name"] for r in out["rows"]]
        assert names == ["smoke_d", "smoke_a", "smoke_b"]

    def test_limit(self, project_with_runs):
        out = search_runs(
            project_with_runs,
            where="status == 'completed'",
            order_by="result.val_loss",
            limit=2,
        )
        assert out["matched"] == 3
        assert out["returned"] == 2

    def test_select_projects_columns(self, project_with_runs):
        out = search_runs(
            project_with_runs,
            where="status == 'completed'",
            order_by="result.val_loss",
            limit=2,
            select=["name", "result.val_loss"],
        )
        assert all(set(r.keys()) == {"name", "val_loss"} for r in out["rows"])
        assert out["rows"][0]["name"] == "smoke_b"
        assert out["rows"][0]["val_loss"] == 0.9

    def test_config_namespace_in_predicate(self, project_with_runs):
        out = search_runs(
            project_with_runs,
            where="model_dim > 256 and status == 'completed'",
        )
        # Only r2 has model_dim=512.
        assert out["matched"] == 1
        assert out["rows"][0]["name"] == "smoke_b"

    def test_unknown_source_raises(self, project_with_runs):
        with pytest.raises(SearchError, match="Unknown source"):
            search_runs(project_with_runs, source="bogus")

    def test_dispatch_through_mcp(self, project_with_runs, monkeypatch):
        from crucible.mcp.tools import TOOL_DISPATCH
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config", lambda: project_with_runs
        )
        handler = TOOL_DISPATCH["runs_search"]
        out = handler({"where": "status == 'completed'", "limit": 10})
        assert out["matched"] == 3


# ---------------------------------------------------------------------------
# strict_fields (G.4 seam 4)
# ---------------------------------------------------------------------------


class TestStrictFields:
    """When strict_fields=True, a predicate identifier that doesn't appear
    in any row raises SearchError with a nearest-match suggestion
    instead of silently matching zero rows."""

    def test_unknown_field_raises_with_suggestion(self, project_with_runs):
        # Typo: val_los instead of val_loss (under result.).
        with pytest.raises(SearchError, match="unknown field"):
            search_runs(
                project_with_runs,
                where="val_los < 2.0",
                strict_fields=True,
            )

    def test_unknown_field_suggests_nearest(self, project_with_runs):
        try:
            search_runs(
                project_with_runs,
                where="moddel_dim > 256",
                strict_fields=True,
            )
        except SearchError as exc:
            assert "moddel_dim" in str(exc)
            assert "model_dim" in str(exc), (
                "should suggest the real field name as a typo fix"
            )

    def test_known_field_passes_strict_mode(self, project_with_runs):
        # status, model_dim, result.val_loss all exist in fixture rows.
        out = search_runs(
            project_with_runs,
            where="status == 'completed' and model_dim > 100 and result.val_loss < 2.0",
            strict_fields=True,
        )
        assert out["matched"] >= 1

    def test_strict_fields_default_off_preserves_silent_typo(self, project_with_runs):
        """Backward compat: default behavior unchanged."""
        out = search_runs(
            project_with_runs,
            where="nonexistent_field == 'x'",
        )
        assert out["matched"] == 0
        assert "error" not in out

    def test_top_level_dotted_paths_validate(self, project_with_runs):
        """result.val_loss is a real nested key; strict mode accepts it."""
        out = search_runs(
            project_with_runs,
            where="result.val_loss < 5.0",
            strict_fields=True,
        )
        assert out["matched"] >= 1

    def test_dotted_path_into_nonexistent_field_raises(self, project_with_runs):
        with pytest.raises(SearchError, match="unknown field"):
            search_runs(
                project_with_runs,
                where="nope.something == 'x'",
                strict_fields=True,
            )
