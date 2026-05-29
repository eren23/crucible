"""Recipe lint — every `tool:` named in a recipe must exist in TOOL_DISPATCH.

This is the regression test for the C4 audit finding: both
`co-scientist-style-tournament.yaml` and
`flagship-param-golf-discovery.yaml` shipped calls to
`autonomous_research_loop(action='request_prompt')` which doesn't
exist. An orchestrator following those recipes would hit a runtime
error on the first round.

The test walks every recipe YAML in `docs/recipes/` and
`.crucible/recipes/`, collects every `tool: <name>` value, and asserts
each name appears in `TOOL_DISPATCH`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# A loose grammar: capture every `tool: <identifier>` value. Recipes
# are not strictly valid YAML in places (multi-line gotchas embed
# colons, etc.) so we don't try to YAML-parse the whole file — we
# pattern-match the lines that name MCP tools, which is what we
# actually need to lint.
_TOOL_LINE_RE = re.compile(r"^\s*tool:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", re.MULTILINE)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIRS = [
    PROJECT_ROOT / "docs" / "recipes",
    PROJECT_ROOT / ".crucible" / "recipes",
]


def _all_recipes() -> list[Path]:
    out: list[Path] = []
    for d in RECIPE_DIRS:
        if d.exists():
            out.extend(sorted(d.glob("*.yaml")))
    return out


def test_at_least_one_recipe_exists():
    """Sanity: the recipe dirs aren't empty."""
    recipes = _all_recipes()
    assert recipes, f"no recipes under {RECIPE_DIRS}"


@pytest.mark.parametrize("recipe_path", _all_recipes(), ids=lambda p: p.name)
def test_recipe_tool_names_dispatchable(recipe_path):
    """Every `tool: <name>` in a recipe must be a real TOOL_DISPATCH key."""
    from crucible.mcp.tools import TOOL_DISPATCH

    raw = recipe_path.read_text()
    referenced = sorted(set(_TOOL_LINE_RE.findall(raw)))
    if not referenced:
        pytest.skip(f"{recipe_path.name} has no `tool:` lines")
    unknown = [name for name in referenced if name not in TOOL_DISPATCH]
    assert not unknown, (
        f"{recipe_path.name} references tool names that are not in "
        f"TOOL_DISPATCH: {unknown}. Either add the tool or fix the recipe."
    )
