"""CLI handlers for session recipes (playbooks).

Thin wrappers around the existing recipe_list / recipe_get MCP tools,
exposed as ``crucible recipe list`` and ``crucible recipe get NAME`` so
the recipe library is discoverable without dropping into the MCP
client. Surfaced in Phase 2.6 as part of the discoverability surge.
"""
from __future__ import annotations

import argparse
import json
import sys


def handle_recipe(args: argparse.Namespace) -> None:
    cmd = getattr(args, "recipe_command", None)
    if cmd == "list":
        _list_recipes(args)
    elif cmd == "get":
        _get_recipe(args)
    else:
        print("Usage: crucible recipe {list|get NAME}", file=sys.stderr)
        sys.exit(2)


def _list_recipes(args: argparse.Namespace) -> None:
    from crucible.mcp.tools import recipe_list

    out = recipe_list({
        "tag": getattr(args, "tag", None) or "",
        "tags": list(getattr(args, "tags", []) or []),
    })
    if "error" in out:
        print(out["error"], file=sys.stderr)
        sys.exit(1)

    recipes = out.get("recipes", [])
    if not recipes:
        print("No recipes found in .crucible/recipes/. "
              "Use `recipe_save` (MCP) after a successful session to capture one.")
        return

    name_w = max(len(r.get("name", "")) for r in recipes)
    name_w = max(name_w, len("name"))
    print(f"{'name':<{name_w}}  title")
    print(f"{'-' * name_w}  {'-' * 40}")
    for r in recipes:
        name = r.get("name", "?")
        title = (r.get("title") or "").strip().replace("\n", " ")
        tags = ", ".join(r.get("tags", []))
        line = f"{name:<{name_w}}  {title}"
        if tags:
            line += f"  [tags: {tags}]"
        print(line)


def _get_recipe(args: argparse.Namespace) -> None:
    from crucible.mcp.tools import recipe_get

    out = recipe_get({"name": args.name})
    if "error" in out:
        print(out["error"], file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, indent=2, default=str))
