"""CLI handlers for experiment execution commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crucible.core.config import load_config
from crucible.core.errors import CrucibleError


def handle_run(args: argparse.Namespace) -> None:
    try:
        _handle_run(args)
    except CrucibleError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _handle_run(args: argparse.Namespace) -> None:
    config = load_config()
    cmd = getattr(args, "run_command", None)

    if cmd == "experiment":
        from crucible.runner.experiment import run_experiment

        overrides: dict[str, str] = {}
        for item in getattr(args, "overrides", None) or []:
            if "=" in item:
                k, v = item.split("=", 1)
                overrides[k] = v

        result = run_experiment(
            config=overrides,
            name=getattr(args, "name", None) or "cli_experiment",
            preset=getattr(args, "preset", "smoke"),
            backend=getattr(args, "backend", "torch"),
            timeout_seconds=getattr(args, "timeout", None),
            project_root=config.project_root,
            project_config=config,
        )
        status = result.get("status", "unknown")
        res = result.get("result", {})
        print(f"Status: {status}")
        if res:
            for k, v in res.items():
                print(f"  {k}: {v}")

    elif cmd == "enqueue":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        spec_path = Path(args.spec)
        if not spec_path.is_absolute():
            spec_path = config.project_root / spec_path
        if not spec_path.exists():
            print(f"Spec file not found: {spec_path}", file=sys.stderr)
            sys.exit(1)

        added = fleet.enqueue(spec_path=spec_path, limit=getattr(args, "limit", 0))
        print(f"Enqueued {len(added)} experiments.")
        for item in added:
            print(f"  {item['experiment_name']} ({item['tier']})")

    elif cmd == "dispatch":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        dispatched = fleet.dispatch(max_assignments=getattr(args, "limit", 8))
        running = [r for r in dispatched if r.get("lease_state") == "running"]
        queued = [r for r in dispatched if r.get("lease_state") == "queued"]
        print(f"Dispatched: {len(running)} running, {len(queued)} still queued.")

    elif cmd == "collect":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        fleet.collect()
        print("Results collected.")

    elif cmd == "day":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        day_dir = fleet.run_day(
            count=getattr(args, "count", 4),
            name_prefix=getattr(args, "name_prefix", "crucible-day"),
        )
        print(f"Day run complete. Output: {day_dir}")

    elif cmd == "night":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        spec_path = getattr(args, "spec", None)
        if spec_path:
            fleet.run_night(
                spec_path=Path(spec_path),
                count=getattr(args, "count", 2),
                name_prefix=getattr(args, "name_prefix", "crucible-night"),
            )
        else:
            print("Error: --spec required for night runs", file=sys.stderr)
            sys.exit(1)

    else:
        print("Usage: crucible run {experiment|queue|enqueue|dispatch|collect|day|night}", file=sys.stderr)
