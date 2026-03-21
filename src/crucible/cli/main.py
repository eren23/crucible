"""Top-level Crucible CLI with subcommand groups."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    from crucible.core.errors import CrucibleError

    try:
        _main()
    except CrucibleError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _main() -> None:
    parser = argparse.ArgumentParser(
        prog="crucible",
        description="Crucible — ML research platform for fleet orchestration, autonomous experimentation, and model development.",
    )
    parser.add_argument("--version", action="version", version="crucible 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── init ──
    subparsers.add_parser("init", help="Initialize a new crucible.yaml in the current directory")

    # ── fleet ──
    fleet_parser = subparsers.add_parser("fleet", help="Fleet management")
    fleet_sub = fleet_parser.add_subparsers(dest="fleet_command")
    fleet_sub.add_parser("status", help="Show node inventory and health")

    prov = fleet_sub.add_parser("provision", help="Provision compute nodes")
    prov.add_argument("--count", type=int, default=2, help="Number of nodes to create")
    prov.add_argument("--name-prefix", type=str, default="crucible", help="Node name prefix")
    prov.add_argument("--interruptible", action="store_true", help="Use spot instances")

    fleet_sub.add_parser("destroy", help="Tear down nodes").add_argument(
        "--node", type=str, action="append", dest="nodes", help="Node names to destroy"
    )
    boot = fleet_sub.add_parser("bootstrap", help="Bootstrap nodes (sync code + data)")
    boot.add_argument("--train-shards", type=int, default=1, help="Number of training data shards to download")
    boot.add_argument("--skip-install", action="store_true", help="Skip pip install step")
    boot.add_argument("--skip-data", action="store_true", help="Skip data download step")
    fleet_sub.add_parser("sync", help="Push local code to nodes")
    mon = fleet_sub.add_parser("monitor", help="Live node status")
    mon.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds")

    # ── run ──
    run_parser = subparsers.add_parser("run", help="Experiment execution")
    run_sub = run_parser.add_subparsers(dest="run_command")

    exp = run_sub.add_parser("experiment", help="Run a single experiment")
    exp.add_argument("--preset", type=str, default="smoke", help="Preset name")
    exp.add_argument("--backend", type=str, default="torch", help="Training backend")
    exp.add_argument("--name", type=str, help="Experiment name")
    exp.add_argument("--timeout", type=int, help="Timeout in seconds")
    exp.add_argument("--set", action="append", dest="overrides", metavar="KEY=VALUE", help="Config overrides")

    queue = run_sub.add_parser("queue", help="Run tiered queue locally")
    queue.add_argument("--tier", type=str, default="proxy", help="Tier to run")
    queue.add_argument("--backend", type=str, default="torch", help="Training backend")

    enq = run_sub.add_parser("enqueue", help="Add experiments to fleet queue")
    enq.add_argument("--spec", type=str, required=True, help="Path to spec JSON file")
    enq.add_argument("--limit", type=int, default=0, help="Max experiments to enqueue")

    run_sub.add_parser("dispatch", help="Assign queued runs to nodes").add_argument(
        "--limit", type=int, default=8, help="Max assignments"
    )
    run_sub.add_parser("collect", help="Gather results from fleet nodes")

    day = run_sub.add_parser("day", help="Full multi-wave day orchestration")
    day.add_argument("--count", type=int, default=4, help="Number of nodes")
    day.add_argument("--name-prefix", type=str, default="crucible-day", help="Node name prefix")

    night = run_sub.add_parser("night", help="Overnight breadth run")
    night.add_argument("--spec", type=str, help="Path to night spec JSON")
    night.add_argument("--count", type=int, default=2, help="Number of nodes")

    # ── analyze ──
    analyze_parser = subparsers.add_parser("analyze", help="Results analysis")
    analyze_sub = analyze_parser.add_subparsers(dest="analyze_command")

    rank = analyze_sub.add_parser("rank", help="Ranked results table")
    rank.add_argument("--top", type=int, default=20, help="Number of results to show")
    rank.add_argument("--tag", type=str, help="Filter by tag")

    analyze_sub.add_parser("sensitivity", help="Parameter sensitivity analysis")
    analyze_sub.add_parser("pareto", help="Pareto frontier")

    export = analyze_sub.add_parser("export", help="Export top configs as JSON")
    export.add_argument("--top", type=int, default=5, help="Number of configs to export")
    export.add_argument("--out", type=str, default="exported_configs", help="Output directory")

    analyze_sub.add_parser("summary", help="Generate markdown summary")

    # ── research ──
    research_parser = subparsers.add_parser("research", help="Autonomous research loop")
    research_sub = research_parser.add_subparsers(dest="research_command")

    start = research_sub.add_parser("start", help="Launch autonomous researcher")
    start.add_argument("--budget-hours", type=float, default=10.0, help="Total compute-hours budget")
    start.add_argument("--max-iterations", type=int, default=20, help="Max research iterations")
    start.add_argument("--tier", type=str, default="proxy", help="Experiment tier")
    start.add_argument("--backend", type=str, default="torch", help="Training backend")
    start.add_argument("--dry-run", action="store_true", help="Print without executing")

    research_sub.add_parser("status", help="Show research state")

    # ── data ──
    data_parser = subparsers.add_parser("data", help="Data management")
    data_sub = data_parser.add_subparsers(dest="data_command")

    dl = data_sub.add_parser("download", help="Download dataset shards")
    dl.add_argument("--variant", type=str, default="sp1024", help="Dataset variant")
    dl.add_argument("--train-shards", type=int, default=80, help="Number of training shards")

    data_sub.add_parser("sync", help="Push data to fleet nodes")
    data_sub.add_parser("status", help="Show local data availability")

    # ── tui ──
    subparsers.add_parser("tui", help="Interactive experiment design browser")

    # ── store ──
    store_parser = subparsers.add_parser("store", help="Version store management")
    store_sub = store_parser.add_subparsers(dest="store_command")

    store_list = store_sub.add_parser("list", help="List versioned resources")
    store_list.add_argument("resource_type", choices=["designs", "context"], help="Resource type to list")

    store_show = store_sub.add_parser("show", help="Show current version of a resource")
    store_show.add_argument("resource_path", help="Resource path: {type}/{name}")

    store_history = store_sub.add_parser("history", help="Show version history")
    store_history.add_argument("resource_path", help="Resource path: {type}/{name}")

    store_sub.add_parser("commit", help="Git commit all uncommitted versions")

    # ── mcp ──
    mcp_parser = subparsers.add_parser("mcp", help="MCP server")
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command")
    mcp_sub.add_parser("serve", help="Start MCP server (stdio)")

    # ── models ──
    models_parser = subparsers.add_parser("models", help="Model zoo")
    models_sub = models_parser.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="List registered model families")

    # Parse and dispatch
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _dispatch(args)


def _dispatch(args: argparse.Namespace) -> None:
    """Route parsed args to the appropriate handler."""

    if args.command == "init":
        _cmd_init()
    elif args.command == "fleet":
        from crucible.cli.fleet_commands import handle_fleet

        handle_fleet(args)
    elif args.command == "run":
        from crucible.cli.run_commands import handle_run

        handle_run(args)
    elif args.command == "analyze":
        from crucible.cli.analyze_commands import handle_analyze

        handle_analyze(args)
    elif args.command == "research":
        from crucible.cli.research_commands import handle_research

        handle_research(args)
    elif args.command == "data":
        from crucible.cli.data_commands import handle_data

        handle_data(args)
    elif args.command == "tui":
        from crucible.tui.app import main as tui_main

        tui_main()
    elif args.command == "store":
        from crucible.cli.store_commands import handle_store

        handle_store(args)
    elif args.command == "mcp":
        from crucible.cli.mcp_commands import handle_mcp

        handle_mcp(args)
    elif args.command == "models":
        from crucible.cli.analyze_commands import handle_models

        handle_models(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


def _cmd_init() -> None:
    """Initialize a crucible.yaml in the current directory."""
    from crucible.core.config import generate_default_config

    target = Path.cwd() / "crucible.yaml"
    if target.exists():
        print(f"crucible.yaml already exists at {target}")
        sys.exit(1)
    target.write_text(generate_default_config(), encoding="utf-8")
    print(f"Created {target}")


if __name__ == "__main__":
    main()
