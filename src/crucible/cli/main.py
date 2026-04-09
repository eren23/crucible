"""Top-level Crucible CLI with subcommand groups."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crucible import __version__


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
    parser.add_argument("--version", action="version", version=f"crucible {__version__}")

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

    # ── hub ──
    hub_parser = subparsers.add_parser("hub", help="Cross-project knowledge hub")
    hub_sub = hub_parser.add_subparsers(dest="hub_command")
    hub_sub.add_parser("init", help="Initialize ~/.crucible-hub")
    hub_sub.add_parser("link", help="Register current project in hub")
    hub_sub.add_parser("sync", help="Sync hub via git")
    hub_sub.add_parser("status", help="Show hub status")

    # ── tap ──
    tap_parser = subparsers.add_parser("tap", help="Community plugin taps")
    tap_sub = tap_parser.add_subparsers(dest="tap_command")
    ta = tap_sub.add_parser("add", help="Add a plugin tap")
    ta.add_argument("url")
    ta.add_argument("--name", default="")
    tr = tap_sub.add_parser("remove", help="Remove a tap")
    tr.add_argument("name")
    tap_sub.add_parser("list", help="List configured taps")
    tsy = tap_sub.add_parser("sync", help="Sync tap repos")
    tsy.add_argument("--name", default="")
    tsr = tap_sub.add_parser("search", help="Search for plugins")
    tsr.add_argument("query")
    tsr.add_argument("--type", default="")
    ti = tap_sub.add_parser("install", help="Install a plugin")
    ti.add_argument("name")
    ti.add_argument("--tap", default="")
    tu = tap_sub.add_parser("uninstall", help="Uninstall a plugin")
    tu.add_argument("name")
    tins = tap_sub.add_parser("installed", help="List installed plugins")
    tins.add_argument("--type", default="")
    tpub = tap_sub.add_parser("publish", help="Publish a local plugin")
    tpub.add_argument("name")
    tpub.add_argument("--type", required=True)
    tpub.add_argument("--tap", required=True)
    tinfo = tap_sub.add_parser("info", help="Show plugin details")
    tinfo.add_argument("name")
    tpush = tap_sub.add_parser("push", help="Push tap repo to remote")
    tpush.add_argument("tap")
    tpr = tap_sub.add_parser("submit-pr", help="Open PR from tap fork to upstream")
    tpr.add_argument("tap")
    tpr.add_argument("--title", default="")
    tpr.add_argument("--body", default="")
    tval = tap_sub.add_parser(
        "validate",
        help="Validate every plugin.yaml in a tap directory against the schema",
    )
    tval.add_argument("path", help="Path to the tap repo root")
    tval.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Exit non-zero even if only warnings are found (CI-friendly)",
    )

    # ── track ──
    track_parser = subparsers.add_parser("track", help="Research track management")
    track_sub = track_parser.add_subparsers(dest="track_command")
    tc = track_sub.add_parser("create", help="Create a research track")
    tc.add_argument("name")
    tc.add_argument("--description", default="")
    tc.add_argument("--tags", nargs="*", default=[])
    ts = track_sub.add_parser("switch", help="Switch active track")
    ts.add_argument("name")
    track_sub.add_parser("list", help="List tracks")
    tshow = track_sub.add_parser("show", help="Show track details")
    tshow.add_argument("name", nargs="?", default="")

    # ── project ──
    project_parser = subparsers.add_parser(
        "project",
        help="Project spec management (create from template, validate, list)",
    )
    project_sub = project_parser.add_subparsers(dest="project_command")
    pnew = project_sub.add_parser(
        "new", help="Create a new project spec from a template"
    )
    pnew.add_argument("name", help="Project name (becomes .crucible/projects/<name>.yaml)")
    pnew.add_argument(
        "--template",
        "-t",
        default="generic",
        help="Template to use (default: generic). Run 'crucible project templates' to list options.",
    )
    pnew.add_argument(
        "--set",
        action="append",
        dest="set_vars",
        metavar="KEY=VALUE",
        help="Set a template variable (repeatable). Example: --set REPO_URL=https://...",
    )
    pnew.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite an existing spec with the same name",
    )
    pnew.add_argument(
        "--no-prompt",
        action="store_true",
        help="Fail instead of prompting for missing required variables (CI-friendly)",
    )
    project_sub.add_parser("list", help="List existing project specs")
    project_sub.add_parser(
        "templates", help="List available built-in project templates"
    )
    pvalidate = project_sub.add_parser(
        "validate", help="Load and validate an existing project spec"
    )
    pvalidate.add_argument("name", help="Project name to validate")

    # ── note ──
    note_parser = subparsers.add_parser("note", help="Experiment notes")
    note_sub = note_parser.add_subparsers(dest="note_command")
    na = note_sub.add_parser("add", help="Add note to a run")
    na.add_argument("run_id")
    na.add_argument("--text", required=True)
    na.add_argument("--stage", default="")
    na.add_argument("--tags", nargs="*", default=[])
    nl = note_sub.add_parser("list", help="List notes for a run")
    nl.add_argument("run_id")
    ns = note_sub.add_parser("search", help="Search notes across runs")
    ns.add_argument("--query", required=True)

    # ── serve ──
    serve = subparsers.add_parser("serve", help="Start API server")
    serve.add_argument("--port", type=int, default=8741)
    serve.add_argument("--host", default="0.0.0.0")

    # ── mcp ──
    mcp_parser = subparsers.add_parser("mcp", help="MCP server")
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command")
    mcp_serve = mcp_sub.add_parser("serve", help="Start MCP server (stdio)")
    mcp_serve.add_argument("--trace", action="store_true", help="Enable session tracing to .crucible/traces/")
    mcp_serve.add_argument("--trace-id", type=str, default=None, help="Custom session ID for the trace")

    # ── trace ──
    trace_parser = subparsers.add_parser("trace", help="Session trace viewing and export")
    trace_sub = trace_parser.add_subparsers(dest="trace_command")
    trace_sub.add_parser("list", help="List all recorded traces")
    trace_show = trace_sub.add_parser("show", help="Print trace entries for a session")
    trace_show.add_argument("session_id", help="Session ID to display")
    trace_export = trace_sub.add_parser("export", help="Export trace as shareable markdown")
    trace_export.add_argument("session_id", help="Session ID to export")
    trace_export.add_argument("--output", "-o", type=str, default=None, help="Output file path (default: stdout)")

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
    elif args.command == "hub":
        from crucible.cli.hub_commands import handle_hub

        handle_hub(args)
    elif args.command == "tap":
        from crucible.cli.tap_commands import handle_tap

        handle_tap(args)
    elif args.command == "track":
        from crucible.cli.track_commands import handle_track

        handle_track(args)
    elif args.command == "note":
        from crucible.cli.note_commands import handle_note

        handle_note(args)
    elif args.command == "project":
        from crucible.cli.project_commands import handle_project

        handle_project(args)
    elif args.command == "serve":
        from crucible.api.server import main as api_main

        api_main(port=args.port, host=args.host)
    elif args.command == "mcp":
        from crucible.cli.mcp_commands import handle_mcp

        handle_mcp(args)
    elif args.command == "trace":
        from crucible.cli.trace_commands import handle_trace

        handle_trace(args)
    elif args.command == "models":
        from crucible.cli.analyze_commands import handle_models

        handle_models(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


def _cmd_init() -> None:
    """Initialize a crucible.yaml and .crucible/ directory structure."""
    from crucible.core.config import generate_default_config

    cwd = Path.cwd()
    target = cwd / "crucible.yaml"

    # Create crucible.yaml
    if target.exists():
        print(f"crucible.yaml already exists at {target}")
    else:
        target.write_text(generate_default_config(), encoding="utf-8")
        print(f"Created {target}")

    # Create .crucible/ directory structure
    dirs_to_create = [
        cwd / ".crucible",
        cwd / ".crucible" / "plugins" / "optimizers",
        cwd / ".crucible" / "plugins" / "callbacks",
        cwd / ".crucible" / "plugins" / "schedulers",
        cwd / ".crucible" / "projects",
        cwd / ".crucible" / "designs",
        cwd / ".crucible" / "traces",
    ]
    created = []
    for d in dirs_to_create:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d.relative_to(cwd)))

    if created:
        print("Created directories:")
        for name in created:
            print(f"  {name}/")
    else:
        print(".crucible/ directory structure already exists")


if __name__ == "__main__":
    main()
