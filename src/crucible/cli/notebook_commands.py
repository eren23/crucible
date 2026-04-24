"""CLI handlers for ``crucible notebook`` — project-spec → notebook export.

Subcommands:
  - export:    render a project spec as a standalone .py or .ipynb
  - runtimes:  list available runtime profiles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def handle_notebook(args: argparse.Namespace) -> None:
    """Dispatch ``crucible notebook <subcommand>``."""
    cmd = getattr(args, "notebook_command", None)
    if cmd == "export":
        _cmd_export(args)
    elif cmd == "runtimes":
        _cmd_runtimes(args)
    else:
        print(
            "Usage: crucible notebook {export|runtimes}",
            file=sys.stderr,
        )
        sys.exit(1)


def _cmd_export(args: argparse.Namespace) -> None:
    from crucible.notebook import export_project
    from crucible.notebook.exporter import NotebookExportError

    overrides: dict[str, str] = {}
    for raw in getattr(args, "overrides", None) or []:
        if "=" not in raw:
            print(f"Error: --set expects KEY=VALUE, got: {raw!r}", file=sys.stderr)
            sys.exit(2)
        key, _, value = raw.partition("=")
        overrides[key.strip()] = value

    try:
        result = export_project(
            project=args.project,
            runtime=args.runtime,
            preset=args.preset,
            out_path=args.out or None,
            variant=args.variant or None,
            overrides=overrides,
            inline_plugins=bool(getattr(args, "inline_plugins", False)),
            crucible_install=args.crucible_install or None,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except NotebookExportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ exported {result.project} → {result.out_path}")
    print(f"  runtime:  {result.runtime}")
    print(f"  preset:   {result.preset}")
    if result.variant:
        print(f"  variant:  {result.variant}")
    print(f"  cells:    {result.cells}")
    print(f"  size:     {result.size_bytes} bytes")
    if result.source_path and result.source_path != result.out_path:
        print(f"  source:   {result.source_path}")
    if result.open_in_colab_url:
        print(f"  colab:    {result.open_in_colab_url}")


def _cmd_runtimes(args: argparse.Namespace) -> None:
    del args
    from crucible.notebook.runtimes import list_runtimes

    rows = list_runtimes()
    print(f"Available runtimes ({len(rows)}):")
    for row in rows:
        print(f"  {row['name']:16} {row['gpu']:24} {row['description']}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the notebook subcommand group on the top-level CLI parser."""
    notebook_parser = subparsers.add_parser(
        "notebook", help="Notebook export (Colab/Jupyter)"
    )
    notebook_sub = notebook_parser.add_subparsers(dest="notebook_command")

    export = notebook_sub.add_parser(
        "export", help="Export a project spec as a standalone notebook"
    )
    export.add_argument("--project", required=True, help="Project spec name")
    export.add_argument(
        "--runtime", default="colab-h100",
        help="Runtime profile (default: colab-h100). List with `crucible notebook runtimes`.",
    )
    export.add_argument("--preset", default="smoke", help="Preset name (default: smoke)")
    export.add_argument("--variant", default="", help="Variant name from spec.variants")
    export.add_argument(
        "--set", action="append", dest="overrides",
        metavar="KEY=VALUE", help="Extra env overrides (repeatable)",
    )
    export.add_argument(
        "--crucible-install", default="",
        help="pip spec for Crucible itself (default: git+main)",
    )
    export.add_argument(
        "--inline-plugins", action="store_true",
        help="Reserved; inline plugin .py into cells instead of pip-install",
    )
    export.add_argument(
        "--out", default="",
        help="Output path (.py or .ipynb). Default: <project>.ipynb",
    )

    notebook_sub.add_parser("runtimes", help="List available runtime profiles")


def _default_out_path(project: str) -> Path:
    return Path(f"{project}.ipynb")
