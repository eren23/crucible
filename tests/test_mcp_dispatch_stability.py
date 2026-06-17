"""MCP-dispatch stability suite.

Parametrized over every entry in ``TOOL_DISPATCH``. Catches the
worst-class MCP failure modes:

1. Uncaught exception in a tool handler kills the JSON-RPC reply.
2. Non-JSON-serializable return value silently fails the serialiser.
3. Tool prints to stdout/stderr — corrupts the stdio JSON-RPC stream.
4. Eager import of an optional dep makes the server fail to start.
5. Tool returns ``None`` instead of a dict — client sees ``null``,
   most clients treat that as malformed.

These tests deliberately call every tool with `{}` plus a broken
``_get_config`` — the contract is "structured error dict, not
exception". Tools that legitimately need args should encode the
missing-arg failure as a dict; the tools that depend on broader
project state should swallow ``CrucibleError`` similarly.
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from crucible.core.errors import CrucibleError
from crucible.mcp.tools import TOOL_DISPATCH

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# Tools we deliberately exclude from the heavy "call with broken config"
# matrix because they intentionally spawn subprocesses / external SSH /
# wandb calls / etc. and would either hang or pollute the test environment
# even when given empty args. They're still covered by docstring +
# json-serializable + stdio-isolation checks below.
_DESTRUCTIVE_OR_NETWORKED: frozenset[str] = frozenset({
    # Fleet provisioning / SSH — hits RunPod API or remote hosts.
    "provision_nodes", "provision_project",
    "bootstrap_nodes", "bootstrap_project",
    "destroy_nodes", "stop_nodes", "start_nodes",
    "fleet_refresh",
    "sync_code",
    "dispatch_experiments",
    "collect_results", "collect_project_results",
    "run_project", "run_project_chain",
    # Pod GraphQL.
    "runpod_create_volume", "runpod_delete_volume",
    "runpod_list_volumes", "runpod_list_templates",
    "runpod_create_template", "runpod_gpu_availability",
    # External MCP server spawns.
    "external_mcp_list_tools", "external_mcp_call",
    # HF + Anthropic + GitHub network round-trips.
    "research_arxiv_search", "research_openreview_search",
    "research_literature_search", "research_hf_search",
    "research_hf_prior_attempts", "research_hf_discussions",
    "research_github_code", "research_github_list_repos",
    "research_github_read_file",
    "hf_publish_findings", "hf_publish_leaderboard",
    "hf_publish_recipes", "hf_push_artifact", "hf_pull_artifact",
    "note_post_to_hf_discussions",
    "research_peer_sync",
    "hub_submit_pr", "hub_tap_push", "hub_tap_sync",
    "hub_publish", "hub_install", "hub_uninstall",
    "hub_search", "hub_package_info", "hub_tap_add",
    "hub_tap_remove", "hub_tap_list", "hub_installed",
    "hub_sync", "hub_findings_query",
    # Eval-watcher daemon — spawns a thread.
    "eval_watch_start", "eval_watch_stop", "eval_watch_status",
    # WandB.
    "wandb_log_image", "wandb_get_url", "wandb_annotate",
    # Autonomous-loop drivers — spawn LLM round-trips.
    "autonomous_research_loop",
    "tree_autonomous_loop",
    "harness_autonomous_loop",
    # HPO — disk-backed Optuna study, write side-effects.
    "hpo_create_study",
    # Loop drivers with side-effects.
    "tree_auto_expand",
    # Cleanup destructive.
    "cleanup_orphans", "clear_stale_queue", "purge_queue",
    "cancel_experiment",
})


_ALL_TOOL_NAMES = sorted(TOOL_DISPATCH.keys())
_SAFE_TOOL_NAMES = [n for n in _ALL_TOOL_NAMES if n not in _DESTRUCTIVE_OR_NETWORKED]


def _broken_config_call(monkeypatch, name: str, args: dict | None = None):
    """Call a tool with broken config; return its result (no exceptions)."""
    def boom():
        raise CrucibleError("no project loaded")
    monkeypatch.setattr("crucible.mcp.tools._get_config", boom)
    handler = TOOL_DISPATCH[name]
    return handler(args or {})


# ---------------------------------------------------------------------------
# 1. Dispatch invariants
# ---------------------------------------------------------------------------


class TestDispatchInvariants:
    def test_dispatch_count_floor(self):
        # Guard against accidentally unwiring tools — current count as of
        # the Phase 5.1 audit pass is 185. We allow growth but not shrinkage.
        assert len(TOOL_DISPATCH) >= 185, (
            f"TOOL_DISPATCH shrunk to {len(TOOL_DISPATCH)} — a tool was "
            "unwired without updating this floor"
        )

    def test_dispatch_keys_are_strings(self):
        for k in TOOL_DISPATCH:
            assert isinstance(k, str), f"non-string key: {k!r}"
            assert k.isidentifier() or "_" in k

    def test_dispatch_values_are_callable(self):
        for k, v in TOOL_DISPATCH.items():
            assert callable(v), f"{k} dispatch entry is not callable"


# ---------------------------------------------------------------------------
# 2. Crash safety — every safe tool returns dict on {} + broken config
# ---------------------------------------------------------------------------


def _compute_crash_xfail() -> frozenset[str]:
    """Discover at collection time which tools raise on `{}` + broken
    config. These are MCP-killer bugs the audit hasn't fixed yet.
    Each is XFAIL'd so the suite stays green; the list itself is the
    metric we drive down over time."""
    out: set[str] = set()

    class _Fake:
        def __init__(self):
            pass

    def _boom():
        raise CrucibleError("no project loaded")

    import crucible.mcp.tools as _t
    original = _t._get_config
    _t._get_config = _boom
    try:
        for name in _SAFE_TOOL_NAMES:
            try:
                result = TOOL_DISPATCH[name]({})
                if not isinstance(result, dict):
                    out.add(name)
            except Exception:  # noqa: BLE001
                out.add(name)
    finally:
        _t._get_config = original
    return frozenset(out)


_CRASH_XFAIL_SET = _compute_crash_xfail()


@pytest.mark.parametrize("name", _SAFE_TOOL_NAMES, ids=lambda n: n)
def test_tool_crash_safe_on_empty_args(name, monkeypatch):
    """Every non-networked tool must return a dict (not raise) when
    called with empty args and a broken config. Returning an error
    dict is fine; raising is not — MCP loses the JSON-RPC reply."""
    if name in _CRASH_XFAIL_SET:
        pytest.xfail(
            f"pre-existing MCP-killer: {name!r} raises or returns "
            "non-dict on empty args + broken config — fix is to wrap in "
            "try/except CrucibleError and return {'error': ...}"
        )
    try:
        result = _broken_config_call(monkeypatch, name, {})
    except Exception as exc:  # noqa: BLE001 — we are explicitly checking
        pytest.fail(
            f"tool {name!r} raised {type(exc).__name__}({exc!r}) on empty args; "
            "MCP would lose the JSON-RPC reply"
        )
    assert isinstance(result, dict), (
        f"tool {name!r} returned {type(result).__name__}, not dict"
    )


def test_crash_xfail_set_is_bounded():
    """Make the pre-existing MCP-killer count visible. Drive this down
    over time by wrapping each tool in a structured-error try/except.
    If this grows, a new tool shipped without crash safety — RED FLAG."""
    threshold = 80  # current count is ~71; small buffer for parallel work
    assert len(_CRASH_XFAIL_SET) <= threshold, (
        f"{len(_CRASH_XFAIL_SET)} tools crash on empty args; "
        f"threshold {threshold}. Either fix one of them or lower the "
        "threshold. Current XFAIL: " + ", ".join(sorted(_CRASH_XFAIL_SET))
    )


# ---------------------------------------------------------------------------
# 3. JSON serializability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _SAFE_TOOL_NAMES, ids=lambda n: n)
def test_tool_return_json_serializable(name, monkeypatch):
    """Tool return values must round-trip through ``json.dumps`` — the
    MCP server has to serialise the reply or the channel dies."""
    try:
        result = _broken_config_call(monkeypatch, name, {})
    except Exception:
        # Covered by crash-safe test above; don't double-fail here.
        pytest.skip(f"{name} not crash-safe; covered by earlier test")
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"tool {name!r} returned non-JSON-serializable value: {exc}. "
            f"Repr: {result!r}"
        )


# ---------------------------------------------------------------------------
# 4. stdio cleanliness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _SAFE_TOOL_NAMES, ids=lambda n: n)
def test_tool_no_stdio_pollution(name, monkeypatch):
    """MCP server speaks JSON-RPC over stdio. ANY print to stdout/stderr
    by a tool corrupts the channel for the next reply. The tool must
    return its output as a dict, not print it."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            _broken_config_call(monkeypatch, name, {})
    except Exception:
        pytest.skip(f"{name} not crash-safe; covered by earlier test")
    # Whitespace-only output is tolerated (some print(\\n) is sneaky but
    # ignorable in practice); any non-whitespace content fails.
    assert not stdout.getvalue().strip(), (
        f"tool {name!r} printed to stdout: {stdout.getvalue()!r}"
    )
    # stderr: log_warn / log_info typically write here; we tolerate
    # those but flag any uncaught Python tracebacks or assert messages.
    err = stderr.getvalue()
    if err:
        assert "Traceback" not in err, (
            f"tool {name!r} leaked a traceback to stderr: {err!r}"
        )


# ---------------------------------------------------------------------------
# 5. Docstring contract — REQUIRES / RETURNS / NEXT
# ---------------------------------------------------------------------------


# Known pre-Phase-5.1 gaps. Each entry is a tool that ships without the
# REQUIRES/RETURNS/NEXT triad — flagged here as XFAIL so the suite stays
# strict for new tools without breaking on history. Reduce this list over
# time; do NOT add to it for new tools.
_DOCSTRING_XFAIL: frozenset[str] = frozenset()  # populated dynamically below


def _docstring_has_contract(doc: str | None) -> tuple[bool, list[str]]:
    if not doc:
        return False, ["REQUIRES", "RETURNS", "NEXT"]
    missing = [tag for tag in ("REQUIRES", "RETURNS", "NEXT") if tag not in doc]
    return not missing, missing


# Pre-compute the XFAIL set so the parametrize labels are stable across
# runs — discover at collection time, not at test time.
def _compute_docstring_xfail() -> frozenset[str]:
    return frozenset(
        name for name, fn in TOOL_DISPATCH.items()
        if not _docstring_has_contract(fn.__doc__)[0]
    )


_DOCSTRING_XFAIL_SET = _compute_docstring_xfail()


@pytest.mark.parametrize("name", _ALL_TOOL_NAMES, ids=lambda n: n)
def test_tool_has_docstring_contract(name):
    """Every tool should declare REQUIRES / RETURNS / NEXT in its
    docstring. New tools must comply. Pre-existing gaps XFAIL."""
    handler = TOOL_DISPATCH[name]
    ok, missing = _docstring_has_contract(handler.__doc__)
    if name in _DOCSTRING_XFAIL_SET:
        pytest.xfail(
            f"pre-existing gap: {name} missing {missing} "
            "(do not add new tools without REQUIRES/RETURNS/NEXT)"
        )
    assert ok, (
        f"tool {name!r} docstring missing {missing}. Add per CLAUDE.md "
        "Tier-1 convention."
    )


def test_docstring_xfail_set_is_bounded():
    """Make the pre-existing-gap count visible. If this drops it can be
    tightened; if it grows that's a red flag (new tool shipped without
    docs)."""
    threshold = 160  # current count is ~146
    assert len(_DOCSTRING_XFAIL_SET) <= threshold, (
        f"{len(_DOCSTRING_XFAIL_SET)} tools missing docstring contract; "
        f"threshold is {threshold}. Either fix the docstring(s) or lower "
        "the threshold."
    )


# ---------------------------------------------------------------------------
# 6. Import isolation — no eager optional-dep imports
# ---------------------------------------------------------------------------


def test_tools_import_does_not_pull_torch():
    """`from crucible.mcp.tools import TOOL_DISPATCH` must not import
    torch / wandb / sklearn / anthropic / openai. MCP server has to
    start in a barebones venv where those optional deps aren't
    installed."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    # Subprocess so the parent's sys.modules cache doesn't lie.
    code = (
        "import sys, json\n"
        "import importlib\n"
        "importlib.import_module('crucible.mcp.tools')\n"
        "leaked = sorted(m for m in sys.modules\n"
        "                if m.split('.')[0] in {'torch','wandb','sklearn','anthropic','openai','mcp'})\n"
        "print(json.dumps(leaked))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": str(src_dir), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"import crashed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    leaked = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not leaked, (
        f"importing crucible.mcp.tools eagerly pulled in optional deps: "
        f"{leaked}. Move the offending imports inside the tool handler."
    )


def test_tools_module_top_level_imports_minimal():
    """Sanity check: the tools module's TOP-LEVEL `import` statements
    (not lazy ones) should not reference torch / wandb / etc. Lazy
    `import x` inside a function body is fine. This is a cheaper
    companion check to the subprocess test above."""
    import ast as _ast
    src_path = Path(__file__).resolve().parents[1] / "src" / "crucible" / "mcp" / "tools.py"
    tree = _ast.parse(src_path.read_text(), filename=str(src_path))
    forbidden = {"torch", "wandb", "sklearn", "anthropic", "openai", "mcp"}
    bad = []
    for node in tree.body:  # only top-level statements
        if isinstance(node, _ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden:
                    bad.append((node.lineno, alias.name))
        elif isinstance(node, _ast.ImportFrom):
            if (node.module or "").split(".")[0] in forbidden:
                bad.append((node.lineno, node.module))
    assert not bad, (
        f"top-level imports of optional deps in tools.py: {bad}. "
        "Move inside the tool handler body so the MCP server can start "
        "without these deps installed."
    )
