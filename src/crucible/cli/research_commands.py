"""CLI handlers for autonomous research commands."""
from __future__ import annotations

import argparse
import json
import sys

from crucible.core.config import load_config
from crucible.core.errors import CrucibleError


def handle_research(args: argparse.Namespace) -> None:
    try:
        _handle_research(args)
    except CrucibleError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _handle_research(args: argparse.Namespace) -> None:
    config = load_config()
    cmd = getattr(args, "research_command", None)

    if cmd == "run":
        # Orchestrator-contract autonomous loop: start a session, print the
        # first prompt to stdout, and exit. The orchestrator (Claude Code,
        # human, etc.) drives submits via MCP `autonomous_research_loop`.
        # CLI does not call any LLM; that's the whole point.
        from crucible.researcher import autonomous_session as autos

        sub = getattr(args, "run_subcommand", None) or "start"
        if sub == "start":
            out = autos.action_start(
                config,
                iterations=getattr(args, "iterations", 5),
                tier=getattr(args, "tier", "proxy"),
                focus_family=getattr(args, "focus_family", "") or "",
                budget_usd=getattr(args, "budget_usd", None),
            )
            print(json.dumps(out, indent=2, default=str))
            print(
                "\n# Next: call your LLM with the system/user prompts, parse per "
                "schema, then\n#   crucible research run submit --session-id "
                f"{out['session_id']} --response-file resp.json",
                file=sys.stderr,
            )
        elif sub == "submit":
            session_id = getattr(args, "session_id", None)
            response_file = getattr(args, "response_file", None)
            if not session_id or not response_file:
                print(
                    "Error: --session-id and --response-file are required for submit",
                    file=sys.stderr,
                )
                sys.exit(2)
            with open(response_file, encoding="utf-8") as f:
                response = json.load(f)
            out = autos.action_submit(
                config, session_id=session_id, response=response
            )
            print(json.dumps(out, indent=2, default=str))
        elif sub == "status":
            session_id = getattr(args, "session_id", None)
            if not session_id:
                print("Error: --session-id is required", file=sys.stderr)
                sys.exit(2)
            print(json.dumps(autos.action_status(config, session_id=session_id), indent=2, default=str))
        elif sub == "cancel":
            session_id = getattr(args, "session_id", None)
            if not session_id:
                print("Error: --session-id is required", file=sys.stderr)
                sys.exit(2)
            print(json.dumps(
                autos.action_cancel(config, session_id=session_id, reason=getattr(args, "reason", "")),
                indent=2,
                default=str,
            ))
        else:
            print("Usage: crucible research run {start|submit|status|cancel} ...", file=sys.stderr)
            sys.exit(2)

    elif cmd == "status":
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / "research_state.jsonl"
        if not state_path.exists():
            print("No research state found. Run 'crucible research run start' first.")
            return
        state = ResearchState(state_path)
        print(f"Budget remaining: {state.budget_remaining:.2f} compute-hours")
        print(f"Hypotheses: {len(state.hypotheses)} total, {len(state.pending_hypotheses())} pending")
        print(f"History: {len(state.history)} experiments completed")
        if state.beliefs:
            print("\nCurrent beliefs:")
            for b in state.beliefs:
                print(f"  - {b}")

    elif cmd == "import":
        importer_kind = getattr(args, "import_kind", None)
        if importer_kind != "autoresearch":
            print(
                "Usage: crucible research import autoresearch <source-dir> [--name NAME] [--force]",
                file=sys.stderr,
            )
            sys.exit(2)
        from crucible.runner.autoresearch_adapter import import_autoresearch

        source = getattr(args, "source", None)
        if not source:
            print(
                "Error: 'source' positional argument is required for import autoresearch",
                file=sys.stderr,
            )
            sys.exit(2)
        result = import_autoresearch(
            source,
            project_root=config.project_root,
            name=getattr(args, "name", "") or "",
            force=getattr(args, "force", False),
            primary_metric=getattr(args, "primary_metric", "") or "val_loss",
            direction=getattr(args, "direction", "") or "minimize",
        )
        print(json.dumps(result, indent=2, default=str))
        print(
            f"\n# Next: launch a session against the imported project\n"
            f"#   crucible research run start --iterations 3 --tier proxy\n"
            f"# Or run a single smoke experiment first:\n"
            f"#   crucible run experiment --project {result['name']} --preset smoke",
            file=sys.stderr,
        )

    else:
        print("Usage: crucible research {run|status|import}", file=sys.stderr)
