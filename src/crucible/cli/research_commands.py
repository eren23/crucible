"""CLI handlers for autonomous research commands."""
from __future__ import annotations

import argparse
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

    if cmd == "start":
        from crucible.researcher.loop import AutonomousResearcher

        researcher = AutonomousResearcher(
            config=config,
            budget_hours=getattr(args, "budget_hours", 10.0),
            max_iterations=getattr(args, "max_iterations", 20),
            tier=getattr(args, "tier", "proxy"),
            backend=getattr(args, "backend", "torch"),
            dry_run=getattr(args, "dry_run", False),
        )
        researcher.run()

    elif cmd == "status":
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / "research_state.jsonl"
        if not state_path.exists():
            print("No research state found. Run 'crucible research start' first.")
            return
        state = ResearchState(state_path)
        print(f"Budget remaining: {state.budget_remaining:.2f} compute-hours")
        print(f"Hypotheses: {len(state.hypotheses)} total, {len(state.pending_hypotheses())} pending")
        print(f"History: {len(state.history)} experiments completed")
        if state.beliefs:
            print("\nCurrent beliefs:")
            for b in state.beliefs:
                print(f"  - {b}")

    else:
        print("Usage: crucible research {start|status}", file=sys.stderr)
